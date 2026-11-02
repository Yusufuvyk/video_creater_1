"""A report as a Word document (.docx), written with the standard library only.

A .docx is a zip of a few XML files (Office Open XML, ECMA-376). A report needs
only headings, paragraphs, tables, code and pictures, so those are written here
directly rather than adding a package (and its lxml) to the program for them.
Word and LibreOffice open the result and it stays editable.

No Qt.
"""
import re
from xml.sax.saxutils import escape
import zipfile

#: A4 with 2 cm margins, in twentieths of a point (twips); the text width in EMU (914400 per inch).
PAGE_W, PAGE_H, MARGIN = 11906, 16838, 1134
TEXT_WIDTH_EMU = (PAGE_W - 2 * MARGIN) * 635
#: EMU per screen pixel at 96 dpi
EMU_PER_PIXEL = 9525

NAMESPACES = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
              'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
              'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
              'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""

PACKAGE_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""

CORE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>{}</dc:title><dc:creator>GIOVA 3.1</dc:creator>
</cp:coreProperties>"""


def _style(style_id, name, size, color="1B2140", bold=True, before=240, after=80, outline=None, font=None):
    run = "<w:b/>" if bold else ""
    run += '<w:rFonts w:ascii="{0}" w:hAnsi="{0}" w:cs="{0}"/>'.format(font) if font else ""
    return ('<w:style w:type="paragraph" w:styleId="{}"><w:name w:val="{}"/><w:basedOn w:val="Normal"/>'
            '<w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="{}" w:after="{}"/>{}'
            '</w:pPr><w:rPr>{}<w:color w:val="{}"/><w:sz w:val="{}"/></w:rPr></w:style>').format(
        style_id, name, before, after, '<w:outlineLvl w:val="{}"/>'.format(outline) if outline is not None else "",
        run, color, size)


STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI" w:cs="Segoe UI"/>'
          '<w:sz w:val="20"/><w:color w:val="1F2328"/></w:rPr></w:rPrDefault>'
          '<w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault>'
          '</w:docDefaults>'
          '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
          + _style("Title", "Title", 36, before=0, after=160)
          + _style("Heading1", "heading 1", 28, outline=0)
          + _style("Heading2", "heading 2", 23, outline=1)
          + _style("Heading3", "heading 3", 21, before=160, outline=2)
          + _style("Code", "Code", 15, color="1F2328", bold=False, before=0, after=0, font="Consolas")
          + '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr>'
            '<w:tblBorders>' + "".join('<w:{} w:val="single" w:sz="4" w:space="0" w:color="DDE1E6"/>'.format(side)
                                       for side in ("top", "left", "bottom", "right", "insideH", "insideV")) +
          '</w:tblBorders><w:tblCellMar><w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/>'
          '</w:tblCellMar></w:tblPr></w:style></w:styles>')

#: XML 1.0 cannot hold most control characters; a value that carries one would break the file.
_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _text(value):
    return escape(_INVALID.sub("", str(value)))


def _run(text, bold=False, color=None):
    props = ("<w:b/>" if bold else "") + ('<w:color w:val="{}"/>'.format(color) if color else "")
    return '<w:r>{}<w:t xml:space="preserve">{}</w:t></w:r>'.format(
        "<w:rPr>{}</w:rPr>".format(props) if props else "", _text(text))


def _paragraph(text, style=None, color=None):
    props = '<w:pPr><w:pStyle w:val="{}"/></w:pPr>'.format(style) if style else ""
    return "<w:p>{}{}</w:p>".format(props, _run(text, color=color))


def _code(text):
    lines = str(text).split("\n")
    runs = '<w:r><w:br/></w:r>'.join(_run(line) for line in lines)
    return '<w:p><w:pPr><w:pStyle w:val="Code"/><w:shd w:val="clear" w:color="auto" w:fill="F6F7F9"/></w:pPr>{}</w:p>'.format(runs)


def _table(header, rows):
    def cell(value, head=False):
        shade = '<w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="F4F6F8"/></w:tcPr>' if head else ""
        return "<w:tc>{}<w:p><w:pPr><w:spacing w:after=\"0\"/></w:pPr>{}</w:p></w:tc>".format(
            shade, _run(value, bold=head))
    out = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr>']
    out.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>{}</w:tr>'.format("".join(cell(h, True) for h in header)))
    out += ["<w:tr>{}</w:tr>".format("".join(cell(c) for c in row)) for row in rows]
    out.append("</w:tbl><w:p/>")
    return "".join(out)


def _picture(number, width_px, height_px):
    cx = int(min(TEXT_WIDTH_EMU, width_px * EMU_PER_PIXEL))
    cy = int(cx * height_px / max(width_px, 1))
    return ('<w:p><w:pPr><w:spacing w:after="120"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            '<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{n}" name="Figure {n}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic><pic:nvPicPr><pic:cNvPr id="{n}" name="figure{n}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
            '<pic:blipFill><a:blip r:embed="rIdImage{n}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            '</wp:inline></w:drawing></w:r></w:p>').format(cx=cx, cy=cy, n=number)


def write_docx(path, title, parts, images):
    """`parts` as a Report keeps them; `images` [(png bytes, width, height in screen pixels)]."""
    body = [_paragraph(title, "Title")]
    for kind, content in parts:
        if kind in ("h2", "h3"):
            body.append(_paragraph(content, "Heading1" if kind == "h2" else "Heading2"))
        elif kind == "p":
            body.append(_paragraph(content))
        elif kind == "note":
            body.append(_paragraph(content, color="57606A"))
        elif kind == "pre":
            body.append(_code(content))
        elif kind == "table":
            body.append(_table(*content))
        elif kind == "image":
            index, _width = content
            _png, width, height = images[index]
            body.append(_picture(index + 1, width, height))
    body.append('<w:sectPr><w:pgSz w:w="{}" w:h="{}"/><w:pgMar w:top="{m}" w:right="{m}" w:bottom="{m}" '
                'w:left="{m}" w:header="567" w:footer="567" w:gutter="0"/></w:sectPr>'.format(PAGE_W, PAGE_H, m=MARGIN))
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {}><w:body>{}</w:body>'
                '</w:document>').format(NAMESPACES, "".join(body))
    relations = ['<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                 'relationships/styles" Target="styles.xml"/>']
    relations += ['<Relationship Id="rIdImage{0}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                  'relationships/image" Target="media/figure{0}.png"/>'.format(k + 1) for k in range(len(images))]
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", PACKAGE_RELS)
        archive.writestr("docProps/core.xml", CORE.format(_text(title)))
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", STYLES)
        archive.writestr("word/_rels/document.xml.rels",
                         '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://'
                         'schemas.openxmlformats.org/package/2006/relationships">{}</Relationships>'.format(
                             "".join(relations)))
        for k, (data, _w, _h) in enumerate(images):
            archive.writestr("word/media/figure{}.png".format(k + 1), data)
    return path
