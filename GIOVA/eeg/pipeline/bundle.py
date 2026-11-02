"""GIOVA's own functions copied, with everything they use, into one script.

A standalone script must compute with the same code as the Workbench, or its
numbers drift from the sheet's. So the code is not rewritten for the script:
`bundle` reads the package's own source, starts from the functions a pipeline
calls, follows every name they use to the module that defines it, and gathers
those definitions as they are written - comments and docstrings included - in
an order where each comes after what it needs. Only what is used is copied:
asking for the Welch PSD brings `welch_psd` and `welch_sizes`, not the rest of
`core.spectral`.

What a definition needs from outside the package (numpy, scipy, MNE, ...) stays
an import: at the top of the script for the modules' own top-level imports,
where it was written for the imports inside a function (so a library a step
does not use is never loaded). Imports from the package itself are dropped; the
definition they named is copied instead. A package module used as a whole
(`from . import boards`, then `boards.csv_board`) becomes a namespace of the
members used. Two definitions of the same name from different modules cannot
both live in one script: one reached only through such a namespace is renamed
(`mne_io_load`), anything else is an error naming both, so a clash is seen
while the export is written, never as a wrong number later.

Pure stdlib - no Qt.
"""
import ast
import builtins
from dataclasses import dataclass, field
import io
from pathlib import Path
import symtable
import tokenize

PACKAGE = "eeg"
#: Names every script has without importing them.
BUILTINS = set(dir(builtins)) | {"__file__", "__name__"}


class BundleError(Exception):
    pass


@dataclass
class Definition:
    """One top-level statement of a module: a function, a class or an assignment."""
    module: str
    name: str
    node: ast.AST
    #: names it needs from the module's top level, and (module, name) of the package's
    #: function-level imports it makes
    names: set = field(default_factory=set)
    inner: list = field(default_factory=list)
    #: package modules it uses through an alias: alias -> (module, {attribute})
    aliases: dict = field(default_factory=dict)
    #: line numbers (1-based) of function-level package imports, dropped from the copy
    drop: set = field(default_factory=set)


class Module:
    """A package module's source, top-level definitions and imports."""

    def __init__(self, name, path):
        self.name, self.path = name, path
        self.source = path.read_text(encoding="utf-8")
        self.lines = self.source.splitlines()
        self.tree = ast.parse(self.source, str(path))
        self.tables = {}
        for child in symtable.symtable(self.source, str(path), "exec").get_children():
            self.tables.setdefault((child.get_name(), child.get_lineno()), child)
        #: local name -> ("external", import statement) or ("package", module, name) or ("module", module)
        self.imports = {}
        #: name -> Definition
        self.definitions = {}
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for local, target in _import_targets(node, self.name):
                    self.imports[local] = target
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.definitions[node.name] = Definition(self.name, node.name, node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name in _stored(target):
                        self.definitions[name] = Definition(self.name, name, node)

    def analyse(self, definition):
        """Fill in what `definition` uses (once)."""
        if definition.names or definition.inner or definition.aliases:
            return definition
        node = definition.node
        used = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            table = self.tables.get((node.name, node.lineno))
            if table is None:
                raise BundleError("{}.{}: no symbol table".format(self.name, node.name))
            used |= _globals(table)
            # evaluated where the definition stands, in the module's scope
            outer = list(node.decorator_list)
            if isinstance(node, ast.ClassDef):
                outer += node.bases + [k.value for k in node.keywords]
            else:
                arguments = node.args
                outer += arguments.defaults + [d for d in arguments.kw_defaults if d is not None]
                outer += [a.annotation for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs
                          if a.annotation is not None]
                outer += [a.annotation for a in (arguments.vararg, arguments.kwarg) if a and a.annotation]
                if node.returns is not None:
                    outer.append(node.returns)
            for expression in outer:
                used |= _loaded(expression)
            for inner in ast.walk(node):
                if isinstance(inner, ast.ImportFrom) and _is_package(inner, self.name):
                    definition.drop.update(range(inner.lineno, inner.end_lineno + 1))
                    for local, target in _import_targets(inner, self.name):
                        if target[0] == "module":
                            definition.aliases[local] = (target[1], _attributes(node, local))
                        elif local != target[2]:
                            raise BundleError("{}.{}: 'import {} as {}' inside a function; import it by its "
                                              "own name".format(self.name, node.name, target[2], local))
                        else:
                            definition.inner.append((target[1], target[2]))
                elif isinstance(inner, ast.Import) and any(a.name.split(".")[0] == PACKAGE for a in inner.names):
                    raise BundleError("{}.{}: 'import {}...' inside a function".format(self.name, node.name, PACKAGE))
        else:
            value = node.value
            if value is not None:
                used |= _loaded(value)
            if isinstance(node, ast.AnnAssign):
                used |= _loaded(node.annotation)
        for name in sorted(used - BUILTINS):
            if name in self.imports and self.imports[name][0] == "module":
                definition.aliases[name] = (self.imports[name][1], _attributes(node, name))
            elif name not in definition.aliases:
                definition.names.add(name)
        definition.names -= set(definition.aliases)
        if not (definition.names or definition.inner or definition.aliases):
            definition.names.add("")          # analysed: uses nothing
        return definition

    def text(self, definition, renames=None):
        """The definition's source as written, the comment lines right above it included; `renames`
        {old: new} renames those names wherever the code uses them (not attributes, keywords or text)."""
        node = definition.node
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while first > 1 and self.lines[first - 2].lstrip().startswith("#"):
            first -= 1
        lines = [line for number, line in enumerate(self.lines[first - 1:node.end_lineno], first)
                 if number not in definition.drop]
        text = "\n".join(lines)
        return _renamed(text, renames) if renames else text


def _renamed(text, renames):
    """`text` with each name in `renames` replaced where it is a name in the code."""
    tokens = list(tokenize.generate_tokens(io.StringIO(text + "\n").readline))
    edits, depth = [], 0
    for k, token in enumerate(tokens):
        if token.type == tokenize.OP:
            depth += token.string in "([{"
            depth -= token.string in ")]}"
        if token.type != tokenize.NAME or token.string not in renames:
            continue
        before = next((t for t in reversed(tokens[:k]) if t.type not in (tokenize.NL, tokenize.COMMENT)), None)
        after = next((t for t in tokens[k + 1:] if t.type not in (tokenize.NL, tokenize.COMMENT)), None)
        if before is not None and before.string == ".":
            continue                                    # an attribute of something else
        if depth and after is not None and after.string == "=":
            continue                                    # a keyword argument
        edits.append((token.start, token.end, renames[token.string]))
    lines = text.split("\n")
    for (row, start), (_, end), new in sorted(edits, reverse=True):
        line = lines[row - 1]
        lines[row - 1] = line[:start] + new + line[end:]
    return "\n".join(lines)


def _stored(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for element in target.elts for n in _stored(element)]
    return []


def _loaded(expression):
    """Names an expression reads from its scope (a lambda's or comprehension's own names left out)."""
    loads, bound = set(), set()
    for inner in ast.walk(expression):
        if isinstance(inner, ast.Name):
            (loads if isinstance(inner.ctx, ast.Load) else bound).add(inner.id)
        elif isinstance(inner, ast.arg):
            bound.add(inner.arg)
    return loads - bound


def _attributes(node, alias):
    """Which attributes of `alias` a definition uses; `alias` itself passed around is not supported."""
    found, bare = set(), 0
    for inner in ast.walk(node):
        if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name) and inner.value.id == alias:
            found.add(inner.attr)
        elif isinstance(inner, ast.Name) and inner.id == alias and isinstance(inner.ctx, ast.Load):
            bare += 1
    if bare > len([n for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                   and n.value.id == alias]):
        raise BundleError("'{}' is used as a module object, not only as {}.name".format(alias, alias))
    return found


def _globals(table):
    """Names a function's or class's body (and everything nested in it) reads from the module."""
    names = {s.get_name() for s in table.get_symbols() if s.is_global() and not s.is_imported()}
    if table.get_type() == "class":
        names -= {s.get_name() for s in table.get_symbols() if s.is_local()}
    for child in table.get_children():
        names |= _globals(child)
    return names


def _is_package(node, module):
    return node.level > 0 or (node.module or "").split(".")[0] == PACKAGE


def _resolve(node, module):
    """The absolute module an ImportFrom names."""
    if node.level == 0:
        return node.module
    base = module.split(".")
    # a module's own package is everything but its last part (a package's __init__ is not bundled)
    base = base[:len(base) - node.level]
    return ".".join(base + ([node.module] if node.module else []))


def _import_targets(node, module):
    """(local name, target) for each name an import statement binds."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] == PACKAGE:
                yield alias.asname or alias.name, ("module", alias.name)
            else:
                local = alias.asname or alias.name.split(".")[0]
                text = "import {}".format(alias.name) + (" as {}".format(alias.asname) if alias.asname else "")
                yield local, ("external", text)
        return
    if not _is_package(node, module):
        for alias in node.names:
            text = "from {} import {}".format(node.module, alias.name) + (
                " as {}".format(alias.asname) if alias.asname else "")
            yield alias.asname or alias.name, ("external", text)
        return
    source = _resolve(node, module)
    for alias in node.names:
        local = alias.asname or alias.name
        if _module_path(source + "." + alias.name) is not None:
            yield local, ("module", source + "." + alias.name)
        else:
            yield local, ("package", source, alias.name)


_ROOT = None


def package_root():
    """The folder holding the `eeg` package's readable source."""
    global _ROOT
    if _ROOT is None:
        from .export import package_folder
        _ROOT = package_folder()
    return _ROOT


def _module_path(name):
    path = package_root().joinpath(*name.split("."))
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    return None


@dataclass
class Bundle:
    #: import statements, sorted: "import ..." first, then "from ... import ..."
    imports: list
    #: [(module, source text)] in the order they must appear
    definitions: list
    #: every name the bundle defines or imports at the script's top level
    names: set = field(default_factory=set)

    def text(self):
        parts, last = [], None
        for module, text in self.definitions:
            if module != last:
                parts.append("# " + "-" * 70 + "\n# from GIOVA's {}\n# {}".format(
                    module.replace(".", "/") + ".py", "-" * 70))
                last = module
            parts.append(text)
        return "\n\n\n".join(parts)


class Bundler:
    def __init__(self):
        self.modules = {}
        #: name in the script -> (module, original name), ("import", statement) or ("namespace", alias)
        self.owners = {}
        #: ("def", key) and ("namespace", alias, attributes, first) in the order they are written
        self.order = []
        self.imports = set()
        self.done = set()
        #: key -> its name in the script, when two modules define the same name
        self.renames = {}
        #: the definitions the caller asked for: they keep their names
        self.roots = set()
        #: alias -> (module, {attribute: key})
        self.namespaces = {}
        #: alias -> attributes already written into the script
        self.published = {}
        #: (module, name it uses) -> key of the definition it means
        self.meaning = {}

    def module(self, name):
        if name not in self.modules:
            path = _module_path(name)
            if path is None:
                raise BundleError("No module {}".format(name))
            self.modules[name] = Module(name, path)
        return self.modules[name]

    def find(self, module, name, seen=()):
        """(module, name) of the definition `name` means in `module`, following the package's imports."""
        mod = self.module(module)
        if name in mod.definitions:
            return module, name
        target = mod.imports.get(name)
        if target is None or target[0] != "package":
            raise BundleError("{}: '{}' is not a definition of the package".format(module, name))
        if (module, name) in seen:
            raise BundleError("Import loop for {}".format(name))
        return self.find(target[1], target[2], seen + ((module, name),))

    def add(self, module, name):
        """Bundle `module.name` and what it needs; returns its key (module, name) where it is defined."""
        key = self.find(module, name)
        if key in self.done:
            return key
        mod = self.module(key[0])
        definition = mod.analyse(mod.definitions[key[1]])
        # every name a multi-name assignment binds is the same statement
        for other, d in mod.definitions.items():
            if d.node is definition.node:
                self.done.add((key[0], other))
                self._claim((key[0], other))
        for used in sorted(definition.names - {""}):
            self._use(mod, used)
        for source, used in definition.inner:
            self.meaning[(mod.name, used)] = self.add(source, used)
        for alias, (target, attributes) in definition.aliases.items():
            self._namespace(alias, target, attributes)
            self._publish(alias, attributes)
        self.order.append(("def", key))
        return key

    def _claim(self, key):
        """Give `key` its name in the script; on a clash one of the two is renamed module_name."""
        name = key[1]
        owner = self.owners.get(name)
        if owner is None or owner == key:
            self.owners[name] = key
            return
        if owner[0] in ("import", "namespace") or (key in self.roots and owner in self.roots):
            raise BundleError("Two things named '{}': {} and {}.{}".format(name, owner, *key))
        moved = owner if key in self.roots else key
        new = "{}_{}".format(moved[0].split(".")[-1], name)
        if new in self.owners:
            raise BundleError("Cannot rename {}.{}: '{}' is taken too".format(moved[0], name, new))
        self.owners[new] = moved
        self.owners[name] = owner if moved is key else key
        self.renames[moved] = new

    def _use(self, mod, name):
        if name in mod.definitions:
            self.meaning[(mod.name, name)] = self.add(mod.name, name)
            return
        target = mod.imports.get(name)
        if target is None:
            raise BundleError("{}: '{}' is used but neither defined nor imported".format(mod.name, name))
        if target[0] == "external":
            self.imports.add(target[1])
            owner = self.owners.get(name)
            if owner is not None and owner != ("import", target[1]):
                raise BundleError("'{}' is both {} and {}".format(name, target[1], owner))
            self.owners[name] = ("import", target[1])
        elif target[0] == "package":
            self.meaning[(mod.name, name)] = self.add(target[1], target[2])
        else:
            raise BundleError("{}: module '{}' used as a value".format(mod.name, name))

    def _namespace(self, alias, module, attributes):
        known = self.namespaces.get(alias)
        if known is not None and known[0] != module:
            raise BundleError("'{}' names both {} and {}".format(alias, known[0], module))
        members = known[1] if known else {}
        for attribute in sorted(attributes):
            if attribute not in members:
                members[attribute] = self.add(module, attribute)
        if known is None:
            owner = self.owners.get(alias)
            if owner is not None and owner != ("namespace", alias):
                raise BundleError("'{}' is both a module alias and {}".format(alias, owner))
            self.owners[alias] = ("namespace", alias)
            self.imports.add("from types import SimpleNamespace")
        self.namespaces[alias] = (module, members)

    def _publish(self, alias, attributes):
        """The namespace `alias` with `attributes`, written before the definition that uses it: a default value or
        a class body reads it as soon as the definition runs."""
        shown = self.published.setdefault(alias, set())
        missing = sorted(set(attributes) - shown)
        if missing or not shown:
            self.order.append(("namespace", alias, missing, not shown))
        shown.update(attributes)

    def name(self, key):
        return self.renames.get(key, key[1])

    def result(self):
        definitions = []
        for entry in self.order:
            if entry[0] == "def":
                key = entry[1]
                mod = self.module(key[0])
                definition = mod.definitions[key[1]]
                renames = {used: self.name(meant) for (module, used), meant in self.meaning.items()
                           if module == key[0] and self.name(meant) != used}
                for other, d in mod.definitions.items():
                    if d.node is definition.node and self.name((key[0], other)) != other:
                        renames[other] = self.name((key[0], other))
                definitions.append((key[0], mod.text(definition, renames)))
            else:
                _, alias, attributes, first = entry
                module, members = self.namespaces[alias]
                if first:
                    text = "#: GIOVA's {}, as far as this script uses it\n{} = SimpleNamespace({})".format(
                        module, alias, ", ".join("{}={}".format(k, self.name(members[k])) for k in attributes))
                else:
                    text = "\n".join("{}.{} = {}".format(alias, k, self.name(members[k])) for k in attributes)
                definitions.append((module, text))
        plain = sorted(i for i in self.imports if i.startswith("import "))
        grouped = {}
        for statement in self.imports:
            if statement.startswith("from "):
                source, _, names = statement[5:].partition(" import ")
                grouped.setdefault(source, set()).add(names)
        imports = plain + ["from {} import {}".format(s, ", ".join(sorted(n))) for s, n in sorted(grouped.items())]
        return Bundle(imports, definitions, set(self.owners))


def bundle(roots):
    """A Bundle of (module, name) roots and everything they need, e.g. [("eeg.pipeline.compute", "evaluate")].

    The roots keep their names, so the code calling them can use those."""
    bundler = Bundler()
    roots = list(dict.fromkeys(roots))
    bundler.roots = {bundler.find(module, name) for module, name in roots}
    for module, name in roots:
        bundler.add(module, name)
    return bundler.result()
