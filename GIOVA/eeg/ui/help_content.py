"""The Help window's text: one (title, html) pair per topic. Pure data - no Qt.

Numbers quoted here come from the code (filters, quality limits, impedance
constants); tests/test_help.py checks the key ones so this text cannot drift
from what the program actually does.
"""
from .. import NAME, REPOSITORY, VERSION

ABOUT = """
<h2>About {name}</h2>
<p><b>{name} {version}</b>: an open-source biosignal lab for recording, monitoring and analysing EEG.</p>
<h3>For research and teaching only</h3>
<p>{name} is <b>not a medical device</b>. It is not made, tested or approved for diagnosing, monitoring or
treating any condition, and its results must not be used for such decisions. It comes with no warranty.</p>
<h3>Your data stays with you</h3>
<p>{name} sends nothing anywhere: no usage statistics, no crash reports, no recordings. Files are read and
written only where you choose. <b>Send feedback</b> (below) opens a page in your browser; what you write there
is all that is shared.</p>
<h3>Licence</h3>
<p>{name} is free software under the GNU General Public License, version 3 (GPL-3.0): you may use, study,
share and change it; versions you share stay under the same licence, with their source. The licences of the
libraries it carries are in THIRD_PARTY_NOTICES.md beside the program.</p>
<h3>Code, downloads and feedback</h3>
<p><a href="{repo}">{repo}</a>: the source, every release, and the place to report a problem or ask for a
feature (<a href="{repo}/issues">Issues</a>).</p>
""".format(name=NAME, version=VERSION, repo=REPOSITORY)

QUICK_START = """
<h2>Quick start</h2>
<h3>Modes</h3>
<p>GIOVA opens on <b>Home</b> the first time: choose <b>Live monitor</b> (stream, check electrodes,
record), <b>Workbench</b> (analyses built from blocks, over one recording or many), <b>Quick analysis</b>
(one recording with a ready profile) or <b>Games</b>. Under the modes, <b>Continue</b> lists the recordings,
pipelines (with their last accuracy) and reports used last, one click from reopening, and <b>Start a
task</b> opens a mode set up for a usual job. Afterwards it opens in the mode used last (switch that off on
Home). The icon rail down the left edge switches between them: <b>Home</b>, <b>Live monitor</b>,
<b>Workbench</b> (and <b>Games</b>) at the top, or Ctrl+1 to Ctrl+4 (Ctrl+0 for Home). Below a line the rail
opens the side panels: <b>Projects</b> in every mode, <b>Blocks</b> and <b>Library</b> on the Workbench; a
second click on the open one folds it away. <b>Help</b> and <b>Settings</b> sit at its foot. Every icon says
its name when the mouse rests on it. A mode is loaded the first time it is opened, so the ones you do
not use cost nothing.</p>
<h2>Demo day</h2>
<h3>Cyton+Daisy (OpenBCI)</h3>
<ol>
<li>Plug in the USB dongle and switch the board to <b>PC</b>.</li>
<li><b>Monitor</b> tab, source: <b>Live (Cyton+Daisy)</b>. Pick the dongle's port
    (<b>Refresh ports</b> if it is missing). Nothing connects until Start.</li>
<li>Open <b>Controls</b> and choose a channel <b>Profile</b>, e.g.
    <i>Alpha 4 · O1 O2 PO3 PO4</i>. The last profile opens by itself next time.</li>
<li>Put the <b>BIAS</b> electrode on the skin (earlobe or mastoid), then <b>Start</b>.</li>
<li>Run <b>Impedance</b> (top bar). Aim for green/yellow on the channels you use.</li>
<li>Check the quality table: every used channel should read <i>No issues detected</i>
    after a few seconds. Close the eyes: alpha (8–13 Hz) should rise on O1/O2.</li>
<li><b>Games</b> tab: choose the game and press play. The game reads the Monitor's
    stream; do not stop the Monitor while a game runs.</li>
</ol>
<h3>g.tec Unicorn Hybrid Black</h3>
<ol>
<li>Pair the headset in Windows Bluetooth settings first.</li>
<li>Source <b>Live (Unicorn Hybrid Black)</b>; type the serial (UN-XXXX.XX.XX) or leave it
    empty for the first paired headset; <b>Start</b>.</li>
<li>The Unicorn has no channel on/off, BIAS or impedance commands: unticked channels
    are left out in software only.</li>
</ol>
<p class="note">Live alpha demo: the default Controls → Window of <b>5 s</b> suits it. The spectra
are computed from that window, so closing the eyes shows within about 5 s (10 s: finer, but
slower; 2–4 s: faster, noisier).</p>
<p class="note">Low frame rate? First check Settings → Monitor → Trace width is Thin (1 px): wide
lines are the largest drawing cost. Then: use the 4-channel profile, close
<i>Live features</i> and the spectrum grid when not needed. Settings → Monitor → Display rate
offers 60 fps and 30 fps; either steps down by itself (60 → 30 → 20) when the computer cannot
keep up, and says so on the status line. What each number means: <b>Parameters explained</b>.</p>
<p class="note">Every live Start records the raw stream to <code>Data\\live_&lt;date&gt;_&lt;time&gt;.csv</code>.
The status bar shows the file name.</p>
"""

CHANNEL_PROFILES = """
<h2>Channel profiles (Cyton+Daisy)</h2>
<p>A profile says which electrode is on which pin and which pins are powered, so a
headset is described once instead of at every Start.</p>
<table>
<tr><th>Stored</th><td>16 pin names + for each pin <b>On</b> and <b>BIAS</b></td></tr>
<tr><th>Where</th><td><b>Controls → Channels → Profile</b></td></tr>
<tr><th>Applied</th><td>At once: names everywhere; On/BIAS sent to the board if live,
otherwise at the next Start</td></tr>
<tr><th>Remembered</th><td>The last profile picked opens with the program</td></tr>
</table>
<h3>Built-in profiles</h3>
<ul>
<li><b>OpenBCI default (16)</b>: OpenBCI's 16-channel layout, all pins on.</li>
<li><b>Alpha 4 · O1 O2 PO3 PO4</b>: only four pins on (and in BIAS):
  <table>
  <tr><th>Pin</th><td>5</td><td>6</td><td>7</td><td>15</td></tr>
  <tr><th>Electrode</th><td>PO3</td><td>PO4</td><td>O2</td><td>O1</td></tr>
  </table>
  The other 12 pins are off and named CH&lt;pin&gt;, so an old name (O1 on pin 7)
  never appears twice.</li>
</ul>
<h3>Your own profile</h3>
<p><b>…</b> next to Profile → <b>Edit profiles…</b>: works with no board connected and in any
mode, so a profile can be prepared in advance:</p>
<ol>
<li>Select a profile and press <b>New from this…</b> (built-ins are read only).</li>
<li>Type each pin's electrode (suggestions appear), tick <b>On</b> and <b>BIAS</b>.
Two pins with the same name are refused.</li>
<li><b>Save</b>, or <b>Save &amp; use</b> to apply it at once.</li>
</ol>
<p>Alternatively rename pins with <b>Channel names…</b>, tick On/BIAS in <b>Board channels</b>, then
<b>…</b> → <b>Save current as profile…</b>. Built-ins cannot be deleted.</p>
<p class="note">The names are saved beside each live recording
(<code>.channels.json</code>), so playback and the Workbench show O1/PO3… instead of the
default names. Games receive the same names, so a game asking for O1, O2, PO3, PO4
finds them.</p>
"""

MONITOR_MATH = """
<h2>How the Monitor computes</h2>
<h3>Display filters</h3>
<p>Causal (real-time) Butterworth filters, applied as samples arrive:
band-pass <b>3–45 Hz</b>, order 4, plus a <b>50 Hz</b> notch (Q = 30).
Causal filters add a small delay, so the Monitor's filtered samples are not identical
to the Workbench's zero-phase ones. Filters change only what is drawn; recordings and games
always get the raw signal.</p>

<h3>Quality table (every channel, raw data)</h3>
<p>Computed over the last <b>4 s of raw samples</b>, about once per second.</p>
<table>
<tr><th>Column</th><th>Definition</th></tr>
<tr><td>EEG µV</td><td><code>√ ∫ PSD(f) df</code> over 1–45 Hz (Welch). Band-limited on
purpose: raw RMS is mostly drift and mains.</td></tr>
<tr><td>Rail %</td><td>Share of samples within 5 % of the amplifier rail
(±187,500 µV on the Cyton at gain 24; ±750,000 µV on the Unicorn).</td></tr>
<tr><td>50 Hz %</td><td>Power 49–51 Hz ÷ power 1–60 Hz. With the filters' Mains at 60 Hz the column is
<b>60 Hz %</b>: power 59–61 Hz ÷ power 1–70 Hz.</td></tr>
</table>
<p>Verdicts, first match wins:</p>
<table>
<tr><th>Verdict</th><th>Default limit</th></tr>
<tr><td>Missing data</td><td>non-finite samples, or no data for 2 s (<i>Stale</i>)</td></tr>
<tr><td>Saturated</td><td>rail share &gt; 15 %</td></tr>
<tr><td>Flat</td><td>EEG µV &lt; 0.1</td></tr>
<tr><td>Check signal</td><td>EEG µV &gt; 100</td></tr>
<tr><td>Identical signal</td><td>two channels differ by &lt; 2 % (same electrical node:
floating or bridged electrodes)</td></tr>
<tr><td>Line noise</td><td>mains (50 or 60 Hz) share &gt; 20 %</td></tr>
</table>
<p>A warning must repeat <b>2</b> evaluations before it shows and <b>3</b> clean ones before it
clears, so single spikes do not flicker. Limits are editable in
<b>Settings → Monitor</b>. These are signal checks, not impedance. Each column and status,
with causes and what to do: <i>Quality table explained</i>.</p>

<h3>Spectra: which signal</h3>
<p>Both spectra are computed from the <b>last T seconds</b> of each channel, where
<b>T = Controls → Display → Window</b> (default 5 s): the same stretch the traces show.
Changing the window changes the spectra at once: the old spectrum is cleared and a new
one appears when T seconds have been collected. <i>Filtered</i> (default) uses the display
filters' output, <i>Raw</i> the unfiltered samples. The quality table does not follow the
window: it always uses the last 4 s.</p>

<h3>How the FFT is computed</h3>
<p>Every 0.5 s, in the background, for each channel (N = T × fs samples; at 125 Hz and
T = 5 s, N = 625):</p>
<ol>
<li><b>Remove the mean</b>: <code>x[n] − x̄</code>, so the electrode offset (DC) does not leak
into the low frequencies.</li>
<li><b>Hann window</b>: multiply by <code>w[n] = 0.5 − 0.5·cos(2πn / (N−1))</code>. Cutting a
signal to T seconds makes its edges jump; the window tapers them to zero, which keeps a peak
from spreading (<i>leaking</i>) into neighbouring frequencies (Harris, 1978).</li>
<li><b>Real FFT</b>: <code>X[k] = Σ w[n]·(x[n] − x̄)·e<sup>−i2πkn/N</sup></code>, for bins at
<code>f<sub>k</sub> = k·fs / N</code>, k = 0 … N/2. The bin spacing is
<b>Δf = fs / N = 1 / T</b>: 0.2 Hz for 5 s, 0.1 Hz for 10 s, 0.25 Hz for 4 s, 0.5 Hz for 2 s.</li>
<li><b>Scale to µV</b>: <code>A[k] = |X[k]| ÷ Σ w[n]</code> (divides out the window's
coherent gain), then <code>× 2</code> for every bin except DC and Nyquist (one-sided
spectrum). A 20 µV sine at 10 Hz therefore reads 20 µV.</li>
</ol>
<p class="note">The FFT is a single, unaveraged spectrum: sharp but noisy. Use it to see exact
peak frequencies (e.g. SSVEP at 10 Hz vs 10.5 Hz needs Δf ≤ 0.5 Hz, i.e. T ≥ 2 s).</p>

<h3>How the PSD (Welch) is computed</h3>
<ol>
<li>The same T seconds are cut into segments of <b>L seconds</b> (default <b>2 s</b>) that
overlap by a fraction <b>o</b> (default <b>50 %</b>):
<code>K = ⌊(T − L) / (L·(1 − o))⌋ + 1</code> segments. With the defaults: 4 for 5 s, 3 for 4 s, 9 for 10 s.</li>
<li>Each segment: remove its mean, Hann window, FFT, squared magnitude scaled to a
density in <b>µV²/Hz</b>.</li>
<li>The K periodograms are <b>averaged</b> (Welch, 1967). Averaging lowers the noise of the
estimate by about √K, so a longer window gives a <i>smoother</i> PSD. The bin spacing is set by
the segment, not the window: <b>Δf = 1 / L</b> (0.5 Hz for 2 s). If T is shorter than L, one
segment of length T is used.</li>
</ol>

<h3>The unit of the PSD graphs</h3>
<p><b>Settings → Graphs → PSD graphs in</b> chooses what the PSD views of the Monitor and the
Workbench's spectra draw. The Welch density is computed once; the other two are derived from it:</p>
<table>
<tr><th>Choice</th><th>Unit</th><th>Computed as</th></tr>
<tr><td>Power density (default)</td><td>µV²/Hz</td><td>the Welch density itself</td></tr>
<tr><td>Amplitude density</td><td>µV/√Hz</td><td><code>√PSD</code></td></tr>
<tr><td>Amplitude (voltage)</td><td>µV</td><td><code>√(2 · PSD · 1.5 · Δf)</code>: 1.5 · Δf is the
Hann window's noise bandwidth, so PSD · 1.5 · Δf is the power in one bin; a sine of amplitude A
carries A²/2 and therefore reads A, as in the FFT view</td></tr>
</table>
<p class="note">Display only. Band powers, features, SNR, time-frequency maps and every exported
value stay in µV²/Hz and µV². On the dB scale, power density and amplitude density give the same
numbers (10·log of a power = 20·log of its root). A spectrum cannot be in "µV/Hz": a density of
amplitude is per <i>root</i> hertz, because it is power, not amplitude, that adds up over
frequency.</p>

<h3>How the SNR is computed</h3>
<p>The spectrum's <b>SNR</b> mode (Monitor spectra, and the Workbench spectrum's Scale) asks: is the
power at this frequency higher than the power right next to it? It is the usual measure for
flicker (SSVEP) responses, where the stimulus puts a narrow peak on a broad background.</p>
<ol>
<li>Start from the <b>Welch PSD</b> above (µV²/Hz), bins <code>f<sub>k</sub></code> spaced
<b>Δf = 1 / L</b> (0.5 Hz with 2 s segments). On the Monitor it is the displayed, smoothed
PSD: a ratio of averaged powers is steadier than an average of noisy ratios.</li>
<li><b>Neighbours</b>: the bins <b>0.75–2.5 Hz</b> away on <i>both</i> sides. With Δf = 0.5 Hz
that is ±1.0, ±1.5, ±2.0, ±2.5 Hz: 8 bins. The bin right next to f (±0.5 Hz) is skipped on
purpose: the Hann window spreads a peak into it, and counting it would pull the SNR down.</li>
<li><b>Noise level</b>: the <b>median</b> of those neighbour powers. The median, not the mean,
so a second peak (a harmonic, or 50 Hz folding) among the neighbours does not inflate it.
Neighbours that fall outside the spectrum, or have zero power, are left out.</li>
<li><code>SNR(f) = PSD(f) ÷ median( PSD(f ± 0.75…2.5 Hz) )</code>, a plain ratio
(<b>×</b>, not dB).</li>
</ol>
<table>
<tr><th>SNR</th><th>Rough reading</th></tr>
<tr><td>≈ 1</td><td>No peak: f is as strong as its surroundings (most of the spectrum).</td></tr>
<tr><td>2–3</td><td>A visible peak; worth checking it repeats across trials.</td></tr>
<tr><td>&gt; 3–4</td><td>A clear narrow peak, e.g. an SSVEP at the flicker frequency.</td></tr>
</table>
<p class="note">SNR shows <i>narrow</i> peaks only. A broad bump such as eyes-closed alpha
(8–13 Hz) rises with its neighbours, so its SNR stays modest: use PSD or band power for alpha.
Neighbour-bin SNR follows Meigen &amp; Bach (1999) Doc. Ophthalmol. 98:207 and is standard in
the SSVEP literature. The impedance check's 4× test-tone limit is a separate SNR,
described under <i>Impedance check</i>.</p>

<h3>Changing these settings (also live)</h3>
<p><b>Settings → Monitor → Spectra (Monitor)</b>:</p>
<table>
<tr><th>Setting</th><th>Default</th><th>Effect</th></tr>
<tr><td>FFT/PSD window</td><td>Same as display window</td><td>A fixed number of seconds for the
spectra, independent of the traces. FFT Δf = 1 / window.</td></tr>
<tr><td>Welch segment L</td><td>2 s</td><td>PSD Δf = 1 / L. Shorter: more segments to average,
smoother but coarser; longer: finer but noisier.</td></tr>
<tr><td>Welch overlap o</td><td>50 %</td><td>More overlap gives more segments from the same data;
50 % is the usual choice with a Hann window (Welch, 1967).</td></tr>
</table>
<p>A line under the controls shows the result for the current sampling rate (FFT Δf, number
of segments, PSD Δf). Changes apply at once, also while streaming: the spectrum restarts on
the new frequency grid. The quality table is not affected (it keeps 4 s and 2 s segments),
nor are Live features or the Workbench.</p>

<h3>Choosing the window</h3>
<table>
<tr><th>Window T</th><th>FFT Δf</th><th>Welch segments</th><th>Behaviour</th></tr>
<tr><td>2 s</td><td>0.5 Hz</td><td>1</td><td>Reacts fast (eyes closed shows in ~2 s), noisy.</td></tr>
<tr><td>4 s</td><td>0.25 Hz</td><td>3</td><td>Good compromise for a live alpha demo.</td></tr>
<tr><td>5 s (default)</td><td>0.2 Hz</td><td>4</td><td>Balanced: a change shows within 5 s, calm enough to read.</td></tr>
<tr><td>10 s</td><td>0.1 Hz</td><td>9</td><td>Fine, calm spectra; a change takes up to 10 s to show fully.</td></tr>
<tr><td>30 s</td><td>0.033 Hz</td><td>29</td><td>Very fine and stable; for steady recordings, not for quick changes.</td></tr>
</table>
<p class="note">Rule: frequency resolution and reaction time trade off, Δf × T = 1. Displayed
spectra are also smoothed with a 2 s time constant and glide in 0.3 s; the numbers used
for features and exports are not smoothed.</p>

<h3>Band power and relative power</h3>
<ul>
<li><b>Band power</b>: the PSD integrated over the band, with exact (interpolated) edges:
<code>P = ∫<sub>f1</sub><sup>f2</sup> PSD(f) df</code>.</li>
<li>Bands: delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz.</li>
<li><b>Relative alpha</b> = <code>100 × P(8–13 Hz) ÷ P(1–45 Hz)</code>, same channel and
window, not relative to another channel or to eyes open.</li>
<li>Displayed curves and bars are smoothed (≈2 s time constant) for reading; the
numbers used for features and exports are not.</li>
</ul>
"""

IMPEDANCE_MATH = """
<h2>How the impedance check works</h2>
<p>For each channel that is <b>On</b>, one at a time (Cyton+Daisy only):</p>
<ol>
<li>The ADS1299 drives an AC lead-off current of <b>6 nA at 31.25 Hz</b> into the electrode.</li>
<li>After <b>1.5 s</b> settling, <b>4 s</b> are analysed (0.25 Hz bins, so 31.25 Hz falls on a bin).</li>
<li><code>Z = V<sub>31.25 Hz</sub> ÷ 6 nA − 2.2 kΩ</code> (the board's series resistor).</li>
<li>The tone must be at least <b>4×</b> the neighbouring bins; otherwise the result is
<i>No test signal</i> (or <i>Railed</i>), never a made-up number.</li>
</ol>
<table>
<tr><th>Colour</th><th>Meaning</th></tr>
<tr><td style="color:#1a7f37"><b>green</b></td><td>&lt; 750 kΩ</td></tr>
<tr><td style="color:#b08800"><b>yellow</b></td><td>&lt; 2500 kΩ</td></tr>
<tr><td style="color:#cf222e"><b>red</b></td><td>above</td></tr>
<tr><td><b>Open?</b></td><td>≥ 4 MΩ: electrode probably not touching</td></tr>
</table>
<ul>
<li>Limits are the OpenBCI GUI defaults for this same measurement; editable in Settings →
OpenBCI boards → Impedance colours. They belong to the OpenBCI boards only: they are not general EEG
limits, and are not used for the Unicorn or for recordings of other systems.</li>
<li>The current returns through <b>BIAS</b>: the value is electrode + head + BIAS
electrode, so BIAS must be on the skin.</li>
<li>Green means <i>contact is adequate</i>, not <i>signal is good</i>: still check the
mains (50 / 60 Hz) column and the trace.</li>
<li>Values are approximate (square-wave drive, 125 Hz sampling): use them to tell
touching / poor / open apart.</li>
<li>Afterwards each channel gets your own On/BIAS setting back.</li>
</ul>
"""

ANALYSIS_MATH = """
<h2>How quick analysis and features compute</h2>
<p>A quick analysis (Home → Quick analysis, Ctrl+3, or a profile in the Workbench's Library) is a pipeline on the Workbench made from a ready profile; a saved session (<code>.analysis.json</code>) opens there too, with its own trials. What follows is what those blocks compute.</p>
<h3>Filtering and trials</h3>
<ul>
<li>Offline filters are <b>zero-phase</b> (forward + backward): default band-pass
<b>1–45 Hz</b>, order 4, 50 Hz notch Q = 30. Applied to continuous runs before cutting
trials; missing samples keep their place.</li>
<li>Every accepted trial is cut to the same <b>common duration</b>. No stretching or padding.</li>
</ul>
<h3>Spectra</h3>
<ul>
<li>FFT: real, one-sided amplitude (µV), DC removed, normalised by the trial length.</li>
<li>PSD: Hann Welch density (µV²/Hz), 2 s windows, 50 % overlap: the same functions
as the Monitor. <b>Settings → Graphs</b> can draw it as µV/√Hz or µV instead; the computed
values and exports do not change.</li>
<li>Relative power divides by the direct integral over the total range (1–45 Hz by
default), not by the sum of the chosen bands.</li>
<li>PSD scale <b>SNR</b>: each bin over the median of its neighbours 0.75–2.5 Hz away on
both sides (1 = no peak); computed per trial PSD. Details under <i>Monitor: how it
computes → How the SNR is computed</i>.</li>
</ul>
<h3>Features</h3>
<table>
<tr><th>Feature</th><th>Definition</th></tr>
<tr><td>RMS</td><td><code>√ mean((x − x̄)²)</code>, µV</td></tr>
<tr><td>Variance / Hjorth activity</td><td>RMS², µV²</td></tr>
<tr><td>Hjorth mobility</td><td><code>√(var(x′) / var(x))</code>, s⁻¹ (derivative uses the sampling rate)</td></tr>
<tr><td>Hjorth complexity</td><td>mobility(x′) / mobility(x), no unit</td></tr>
<tr><td>Spectral entropy</td><td>Shannon entropy of the normalised PSD ÷ log₂(number of bins), 0–1</td></tr>
<tr><td>Zero crossings</td><td>sign changes per second after DC removal</td></tr>
<tr><td>Cliff's delta</td><td>P(B &gt; A) − P(B &lt; A), from −1 to +1; needs ≥ 3 trials per condition.
Describes overlap, not accuracy and not a significance test.</td></tr>
</table>
<h3>Advanced transforms</h3>
<ul>
<li>STFT frames are density (µV²/Hz); a wavelet map is power (µV², a sine of amplitude A reads
A²/2) and is not integrated as a PSD.</li>
<li>ERD needs a baseline: <code>100 × (P − R) / R</code>. With <b>Pooled power</b> (the default) the
power is averaged over trials before the division; <b>Mean of trial %</b> divides each trial first.</li>
<li>Time-frequency maps: <b>ERD / ERS %</b> draws a decrease (ERD) red and an increase (ERS) blue;
<b>Amplitude (µV)</b> is each trial's amplitude averaged over trials, on a linear scale, not smoothed.</li>
</ul>
<p class="note">A value that cannot be computed is shown as a dash with a reason, never as 0.</p>
"""

PARAMETERS = """
<h2>Parameters explained</h2>
<p>Every adjustable number: what it is, its default, and what changing it does.
One rule runs through all spectra: <b>frequency resolution Δf = 1 / window length</b>.
A longer window separates close peaks better but reacts more slowly.</p>

<h3>Filters (Controls → Filter settings; the Workbench's Filter block)</h3>
<table>
<tr><th>Parameter</th><th>Default</th><th>Meaning and effect</th></tr>
<tr><td>Low cut</td><td>Monitor 3 Hz · Workbench 1 Hz</td><td>High-pass edge: removes electrode drift
below it. Higher = steadier trace, but slow EEG (delta) is lost.</td></tr>
<tr><td>High cut</td><td>45 Hz</td><td>Low-pass edge: removes muscle and high-frequency noise above it.</td></tr>
<tr><td>Order</td><td>4</td><td>Butterworth order: steepness of the edges. Higher = sharper cut but more
ringing after sharp events. A band-pass of order 4 has 8 poles. The Workbench runs the filter
forwards and backwards (zero phase), which doubles the effective order and gives −6 dB
(not −3 dB) at the cut-off.</td></tr>
<tr><td>Notch</td><td>on, 50 Hz</td><td>Removes mains interference: 50 Hz in Europe, 60 Hz in the Americas.</td></tr>
<tr><td>Notch Q</td><td>30</td><td><b>Q = f₀ ÷ bandwidth</b> (the band where the notch removes more than
half the power, −3 dB). Q 30 at 50 Hz removes ≈1.7 Hz: <b>≈49.1–50.8 Hz</b>. Higher Q = narrower
notch, less EEG lost next to 50 Hz, but a drifting mains frequency may escape it.</td></tr>
<tr><td>+ harmonics</td><td>on</td><td>The Filter block's <i>Mains</i> also notches 100, 150 Hz… below Nyquist (its <i>Notch only at</i> takes exact frequencies instead, as a profile's list does)
(fs ÷ 2). At 125 Hz Nyquist is 62.5 Hz, so none exist and only 50 Hz is notched; at 250 Hz
(Unicorn) 100 Hz is notched too.</td></tr>
<tr><td>Reference</td><td>none (Workbench)</td><td><i>average</i>: subtract the mean of all channels;
<i>channels</i>: subtract the mean of the chosen ones. Changes every value; use the same
reference when comparing recordings.</td></tr>
</table>
<p class="note">The Monitor filters causally (as samples arrive, small delay);
the Workbench filters with zero phase (no delay, needs the whole recording). References:
Butterworth design and zero-phase filtering as in Oppenheim &amp; Schafer,
<i>Discrete-Time Signal Processing</i>; the notch is SciPy's <code>iirnotch</code>.</p>

<h3>Monitor spectra</h3>
<table>
<tr><th>Parameter</th><th>Value</th><th>Meaning</th></tr>
<tr><td>FFT window</td><td><b>= display Window</b> (5 s), or a fixed value in
Settings → Monitor → Spectra</td><td>The last T seconds of each channel:
<b>Δf = 1 / T</b> (5 s → 0.2 Hz, 10 s → 0.1 Hz, 4 s → 0.25 Hz). Hann-windowed, mean removed. See
<b>Monitor: how it computes</b> for the steps.</td></tr>
<tr><td>PSD (Welch)</td><td>same T, <b>2 s segments, 50 % overlap</b> (both adjustable in
Settings → Monitor → Spectra, live)</td><td>(T − 2) + 1 segments averaged
(4 for 5 s, 9 for 10 s, 3 for 4 s): <b>Δf = 0.5 Hz</b>, less noisy than one FFT (Welch, 1967).</td></tr>
<tr><td>Update</td><td>every 0.5 s</td><td>The newest T seconds are analysed twice a second.</td></tr>
<tr><td>Smoothing</td><td>2 s time constant</td><td>Displayed spectra and band bars are an
exponential average, so the reading is calm; curves then glide to it in 0.3 s. Features and
exports use unsmoothed values.</td></tr>
<tr><td>Raw / Filtered</td><td>Filtered</td><td>Raw shows interference the display filters hide.</td></tr>
<tr><td>Range, scale</td><td>1–60 Hz, Auto</td><td>Display only.</td></tr>
</table>

<h3>Quality table</h3>
<table>
<tr><th>Parameter</th><th>Value</th><th>Meaning</th></tr>
<tr><td>Window</td><td>last 4 s of raw data</td><td>Updated about once per second.</td></tr>
<tr><td>Flat / Check signal</td><td>0.1 µV / 100 µV</td><td>EEG-band amplitude limits
(±100 µV is the usual artefact limit for scalp EEG).</td></tr>
<tr><td>Line noise</td><td>20 %</td><td>Share of 49–51 Hz in 1–60 Hz power (59–61 Hz in 1–70 Hz
when the filters' Mains is 60 Hz).</td></tr>
<tr><td>Saturated</td><td>15 %</td><td>Share of samples near the amplifier rail.</td></tr>
<tr><td>Hysteresis</td><td>2 on / 3 off</td><td>Evaluations needed to show / clear a warning.</td></tr>
</table>

<h3>Live features</h3>
<table>
<tr><th>Parameter</th><th>Default</th><th>Meaning</th></tr>
<tr><td>Window s</td><td>2 s</td><td>Signal used for each value (PSD of one 2 s window: Δf = 0.5 Hz).</td></tr>
<tr><td>Every s</td><td>0.25 s</td><td>How often a new value is computed (4 per second).</td></tr>
<tr><td>History</td><td>60 s</td><td>Time shown on the feature plot; it scrolls with the signal.</td></tr>
</table>

<h3>Display</h3>
<table>
<tr><th>Parameter</th><th>Default</th><th>Meaning</th></tr>
<tr><td>Window</td><td>5 s</td><td>Seconds of signal on the traces <b>and</b> the stretch the
FFT/PSD spectra are computed from.</td></tr>
<tr><td>Auto / channel</td><td>on</td><td>Each trace gets its own scale (label shows ±µV); off = one common
scale for comparing amplitudes.</td></tr>
<tr><td>Traces</td><td>Scroll</td><td><b>Scroll</b>: the newest signal enters on the right and every
wave crosses the plot once per Window; with a short window (2 s) that is fast and tiring to follow.
<b>Sweep</b> (Settings → Monitor → Traces): the traces stand still and a cursor writes the new signal
over the old from left to right, as on a clinical EEG screen; a small gap ahead of the cursor
separates new from old. A wave stays where it was written until the cursor returns, one Window
later. Display only; the motion plots under the traces keep scrolling. A longer Window (10 s) also
slows the picture.</td></tr>
<tr><td>Large waves</td><td>always in full</td><td>A wave larger than the row scale (a blink on
Fp1, a movement) is drawn in full, over the neighbouring rows, as clinical EEG viewers do; a railed
channel lies along the top or bottom edge of the plot. Display only: recorded and analysed samples
are never clipped.</td></tr>
<tr><td>Time rulings</td><td>always on</td><td>A faint line every <b>second</b>, finer ones every
<b>0.2 s</b> when the Window is 6 s or shorter, and each row's zero line: the ruling of EEG paper.
A line belongs to its second: it moves with a scrolling signal and stands still in a sweep.</td></tr>
<tr><td>Event markers</td><td>file playback</td><td>The recording's events (BrainVision markers,
annotations, a marker sidecar) are drawn as vertical lines across the traces with their code, e.g.
<b>P 1</b> for <code>Primary/P  1</code>. A live stream shows none.</td></tr>
<tr><td>Display rate</td><td>30 fps (or 60 fps)</td><td>How often the screen is redrawn
(Settings → Monitor). 60 fps is smoother; measured on the lab PC with 16 channels it reaches
55–58 fps. On a slower computer it steps down by itself, 60 → 30 → 20 fps, when the last 20
frames arrive 30 % late on average. Samples, features and recordings do not depend on it.</td></tr>
<tr><td>Trace width</td><td>Thin, 1 px</td><td>Line width of traces and spectra (Settings → Monitor).
Bold 2 px reads better on a projector but Qt draws wide lines about 5× more slowly: with 16
channels 1 px keeps 30 fps, 2 px can drop to under 20.</td></tr>
</table>

<h3>Trials (the Epochs block)</h3>
<table>
<tr><th>Parameter</th><th>Default</th><th>Meaning</th></tr>
<tr><td>Common duration (To − From)</td><td>30 s</td><td>Every trial is cut to this length. FFT of a 30 s
trial: Δf ≈ 0.033 Hz.</td></tr>
<tr><td>Welch window / overlap</td><td>2 s / 50 %</td><td>As on the Monitor: Δf = 0.5 Hz.</td></tr>
<tr><td>Review above</td><td>6 MAD</td><td>A trial is flagged when its peak-to-peak amplitude exceeds
median + 6 × MAD (median absolute deviation) of all trials. Flags never exclude a trial.</td></tr>
</table>

<h3>Settings you can change (Settings, top right)</h3>
<table>
<tr><th>Setting</th><th>Default</th><th>Meaning and effect</th></tr>
<tr><td>Graphs → PSD graphs in</td><td>Power density, µV²/Hz</td><td>The unit of the PSD graphs
of the Monitor and of the Workbench's spectra (each spectrum's Scale is chosen beside it): <b>µV²/Hz</b>, <b>µV/√Hz</b> (its square root) or <b>µV</b> (voltage
amplitude). Applied at once; display only. Formulas: <b>Monitor: how it computes → The unit of
the PSD graphs</b>.</td></tr>
<tr><td>Files → Downsample files to</td><td>off</td><td>Files sampled faster than this are
reduced when they are opened in Monitor playback (the Workbench reads a file at its own
rate: its <b>Downsample</b> block reduces it there). The rate is divided by a whole
number n and stays at or above the value: 500 Hz with 250 Hz → 250 Hz (n = 2); 512 Hz → 256 Hz.
Before every n-th sample is kept, a zero-phase low-pass at the new Nyquist (new rate ÷ 2) removes
what the lower rate cannot represent. Open the recording again to apply.</td></tr>
<tr><td>Interface → Hover tips</td><td>on</td><td>The explanations shown when the mouse rests on a control;
Help (F1) has them all either way.</td></tr>
<tr><td>Interface → Theme</td><td>dark</td><td>The window's colours, from the next start; reports and saved
figures are on white either way.</td></tr>
<tr><td>Interface → Show the Games tab</td><td>on</td><td>Games played on the Monitor's live stream.</td></tr>
<tr><td>Interface → Dragging on the Workbench's sheet</td><td>moves the sheet</td><td>Or selects blocks; Space or
the middle button and a drag then moves the sheet. Two fingers or the wheel move it either way.</td></tr>
<tr><td>Recording → Folder, File name</td><td>Data, live</td><td>Where live recordings are written (the
active project) and how their names begin; date and time follow.</td></tr>
</table>
<p class="note">Downsampling: analysis is faster and files of different systems become comparable,
but everything above the new Nyquist is gone. Keep the new rate above twice the highest frequency
you filter or notch: 125 Hz or more for a 50 Hz notch. The file on disk is never changed, events
keep their times, and the load line says <i>downsampled from … Hz</i>. Live streams are not
reduced.</p>
<p class="note">References: Welch, P. (1967) IEEE Trans. Audio Electroacoust. 15:70 (PSD);
Hjorth, B. (1970) Electroencephalogr. Clin. Neurophysiol. 29:306 (activity, mobility,
complexity); Harris, F. J. (1978) Proc. IEEE 66:51 (Hann window, leakage); Cliff, N. (1993) Psychol. Bull. 114:494 (Cliff's delta); Pfurtscheller, G. &amp;
Lopes da Silva, F. (1999) Clin. Neurophysiol. 110:1842 (ERD/ERS); Texas Instruments ADS1299
datasheet and OpenBCI documentation (lead-off impedance).</p>
"""

RECORDING_FILES = """
<h2>Recording files</h2>
<p>Live recordings go to the <b>Data</b> folder next to <code>GIOVA.exe</code>
(or the folder chosen in <b>Settings → Recording</b>).</p>
<table>
<tr><th>File</th><th>Contents</th></tr>
<tr><td><code>live_&lt;date&gt;_&lt;time&gt;.csv</code></td><td>Raw board stream, BrainFlow tab-separated
layout (unfiltered µV). Opens in playback and on the Workbench.</td></tr>
<tr><td><code>….board.json</code></td><td>Which board recorded it (layout, sampling rate).</td></tr>
<tr><td><code>….hardware.tsv</code></td><td>Which channels were on / in BIAS, with unix time, at Start,
after every Apply and during impedance checks.</td></tr>
<tr><td><code>….channels.json</code></td><td>Electrode names (the channel profile's names for live
recordings; your renames for files).</td></tr>
<tr><td><code>…_events.csv</code></td><td>Event markers, when present.</td></tr>
</table>
<h3>Files from other systems</h3>
<p>The Workbench and Monitor playback open these files directly (<b>Browse…</b>, or drop them in the
Data folder):</p>
<table>
<tr><th>Format</th><th>Open this file</th><th>What is read</th></tr>
<tr><td>BrainVision (Brain Products Recorder, e.g. LiveAmp, actiCHamp)</td><td><code>.vhdr</code></td>
<td>The header names two more files, <code>.eeg</code> (samples) and <code>.vmrk</code> (markers):
keep all three together, with their original names. EEG channels in µV at the file's own
sampling rate; every marker becomes an event, labelled as in the file (e.g.
<code>Primary/P  1</code>, <code>Stimulus/S  1</code>).</td></tr>
<tr><td>EDF, BDF, GDF</td><td><code>.edf .bdf .gdf</code></td><td>EEG channels; annotations or
the stimulus channel as events.</td></tr>
<tr><td>EEGLAB, MNE</td><td><code>.set .fif</code></td><td>As above; an epochs file
(<code>-epo.fif</code>) is laid out trial after trial.</td></tr>
<tr><td>MATLAB</td><td><code>.mat</code></td><td>Cho2017 and BNCI 2014-001 are read as they are.
Any other: <b>How this file is read…</b> in the Recording block (or the button its error offers) shows
every variable and asks for the signal, its rate and, if a variable holds them, the events. Runs kept as
<code>data{1}.X</code>, <code>data{2}.X</code> … are read together as <code>data{*}.X</code>; events kept in a
struct array (<code>EEG.event(i).latency</code>) are offered as <code>EEG.event.latency</code>. Trials already cut
(a 3-D array): choose which axis is trials, channels and samples, the variable with one label per trial, and
when each trial starts; the list says which From and To in Epochs give them back as stored. A list at the
bottom says what is still missing. The answer is kept beside the file, and with <b>Read every file laid
out like this one</b> for all files with the same variables (in <code>Data/layouts</code>). Files saved
with <code>-v7.3</code> (HDF5) read the same as older ones, EEGLAB .set files too. MATLAB <code>string</code>,
table and other objects are named as not read, with the MATLAB call that turns them into what can be.</td></tr>
</table>
<p class="note">Read through MNE-Python. Only EEG channels are loaded: an accelerometer
(<code>x_dir</code>, <code>y_dir</code>, <code>z_dir</code> on the LiveAmp) and other non-EEG
channels are left out, and the impedances written in a <code>.vhdr</code> are not shown. These
files carry no amplifier rail, so <i>Rail %</i> is not available for them. Markers are instants:
in the Epochs block set <b>From</b> / <b>To</b> around each event (or <b>Each event's
length</b>). A file sampled faster than you need can be reduced on opening:
<b>Settings → Files → Downsample files to</b>, or a Downsample block on the Workbench.</p>

<p>A game writes its own recording in its own <code>recordings</code> folder, through the
Monitor's board. If that folder's path contains non-English letters (e.g. <i>Masaüstü</i>,
<i>LAST_VERSİON</i>), GIOVA records to a temporary folder and moves the file into
place when the recording stops.</p>
"""

PROJECTS = """
<h2>Projects and side panes</h2>
<p>The <b>Projects</b> pane on the left lists folders you added as projects, each with the
recordings inside it. It is the same on every page, so the project and the recording stay the
same from Monitor to Workbench.</p>
<h3>What a project is</h3>
<p>A project is a <b>folder on disk</b>. Nothing is copied, and there is no database: the program
remembers which folders were added and reads them again each time, so files moved or added in the
file manager show up after <b>Refresh</b> (the circling arrows). Sub-folders are shown as branches (a folder per participant or
per day works without any setting); folders with no recording in them are left out, and so are
sidecar files. A BrainVision recording is listed once, as its <code>.vhdr</code>.</p>
<table>
<tr><th>To</th><th>Do</th></tr>
<tr><td>Add a project</td><td><b>Add project…</b> and choose its folder</td></tr>
<tr><td>Open a recording</td><td>Double-click it: the Workbench runs the pipeline in view on it, the Monitor plays it (whichever page is in view). Right-click: <b>Quick analysis…</b> (a ready profile, in a new Workbench tab) or <b>Play in Monitor</b>.</td></tr>
<tr><td>Rename it, name its channels or events</td><td>Right-click → <b>Rename recording…</b>, <b>Channel names…</b>, <b>Event names…</b> (and <b>MAT layout / dataset preset…</b> for a .mat): written beside the recording, its sidecars renamed with it.</td></tr>
<tr><td>Open a saved analysis</td><td>Double-click <i>saved analysis</i> under a recording (<code>&lt;recording&gt;.analysis.json</code>): a new Workbench tab with its own trials, filters and settings, not saved yet; the file itself is only read.</td></tr>
<tr><td>Record into a project</td><td>Select it and press <b>Make active</b> (also in its
right-click menu). The active project is marked ● and named at the bottom right of the window.</td></tr>
<tr><td>Describe a project</td><td>Right-click → <b>Project info…</b>: name, participant, notes. All
optional; kept in <code>eegstudio_project.json</code> in the folder, and no file is written when
everything is left empty.</td></tr>
<tr><td>Take a project off the list</td><td>Right-click → <b>Remove from this list</b>. The folder
and its files are not touched.</td></tr>
</table>
<p class="note">The <b>active project</b> is where every new live recording is written. It is the
same setting as <b>Settings → Recording → Folder</b>: changing either changes both. A recording
that is already being written stays in the folder it was started in. Check the project name at the
bottom right before pressing Start.</p>
<h3>Folding the side panes</h3>
<p>Everything is shown the first time. The side panel beside the rail (Projects, Blocks, Library) folds
with a second click on its icon in the rail. A Monitor pane you do not need folds to a narrow strip at the
edge of the window: press the small arrow (<b>‹</b> or <b>›</b>) in its heading, and click the strip to
bring it back. What was folded is remembered, for each mode.</p>
<table>
<tr><th>Pane</th><th>Where</th></tr>
<tr><td>Projects</td><td>the side panel, every mode (folder icon on the rail)</td></tr>
<tr><td>All channels · raw signal quality</td><td>Monitor, left</td></tr>
<tr><td>Spectrum</td><td>Monitor, right, when Signals and Spectrum are side by side (as tabs
there is nothing to fold: choose the other tab)</td></tr>
</table>
<p class="note">Folding changes only what is drawn. A folded quality table still checks every
channel, and a folded spectrum is still computed.</p>
<h3>Only what applies is shown</h3>
<p>The program serves live boards and recorded files of several systems. To keep each of them
readable, a control that cannot apply to what you are doing is not drawn at all:</p>
<table>
<tr><th>Shown</th><th>When</th></tr>
<tr><td>File list and <b>Browse…</b> (Monitor, top bar)</td><td>source is <i>File playback</i></td></tr>
<tr><td>Port or serial box, <b>Refresh ports</b></td><td>a live board is the source (Refresh ports:
OpenBCI only)</td></tr>
<tr><td><b>Board channels</b> in Controls (On, BIAS, impedance)</td><td>live Cyton or Cyton+Daisy;
the Unicorn shows its own channel list</td></tr>
<tr><td>Spectrum options (grid or overlay, filtered or raw, log, frequencies, y axis)</td><td>press
<b>Options</b> above the spectrum; the button names what is set</td></tr>
<tr><td>Impedance colours and board diagnostics</td><td><b>Settings → OpenBCI boards</b>, a page of
their own</td></tr>
<tr><td><b>Games</b> tab</td><td><b>Settings → Interface → Show the Games tab</b> (on by default)</td></tr>
<tr><td>Light window instead of dark</td><td><b>Settings → Interface → Theme</b>, from the next start;
reports and saved figures are on white either way</td></tr>
</table>
<p class="note">Nothing is removed: choosing the other source, or pressing Options, brings each
control back where it was.</p>
"""

TROUBLESHOOTING = """
<h2>Troubleshooting</h2>
<table>
<tr><th>Symptom</th><th>Cause and fix</th></tr>
<tr><td><b>Could not start</b> / connection failed</td>
<td>Wrong or busy COM port. Close the OpenBCI GUI and other programs using the dongle,
press <b>Refresh ports</b>, check the board switch is on <b>PC</b>, replug the dongle.</td></tr>
<tr><td><code>BrainFlowError … unable to add streamer</code></td>
<td>BrainFlow cannot write to a path with non-English letters. This version works around
it for game recordings; if it still appears, move the program/game to a folder with
plain letters (e.g. <code>C:\\EEG</code>).</td></tr>
<tr><td>Channel <b>Saturated</b> / railed</td>
<td>Electrode not touching, or BIAS off the skin. Check contact and BIAS, run Impedance.
Switch unused pins off (a profile does this).</td></tr>
<tr><td>Channel <b>Flat</b></td>
<td>Channel is off, or the lead is disconnected.</td></tr>
<tr><td><b>Line noise</b> high</td>
<td>Poor contact or long unshielded leads near mains cables. Improve contact first;
the notch hides it on screen but it is still in the recording.</td></tr>
<tr><td><b>Identical signal</b></td>
<td>Two electrodes on one electrical node: both floating off the head, or bridged
by gel/sweat.</td></tr>
<tr><td><b>Stale</b> / Missing data</td>
<td>No samples for 2 s: dongle too far, battery low, USB power saving. Stop and Start.</td></tr>
<tr><td>Impedance shows <b>No test signal</b></td>
<td>Channel off, or not live. Only On channels on the Cyton+Daisy are measured.</td></tr>
<tr><td>Wrong electrode names</td>
<td>Pick the right <b>Profile</b>, or fix them in <b>Channel names…</b> and save a profile.</td></tr>
<tr><td>Unicorn does not connect</td>
<td>Pair it in Windows Bluetooth first; check the serial number; keep it charged.</td></tr>
<tr><td>A blink or large wave has a <b>flat top</b> on the traces</td>
<td>Waves cross into the neighbouring rows, but not out of the plot: on the top or bottom row a
wave more than twice the row scale is cut at the plot's edge. Raise the scale (Controls →
Display). The data itself is not clipped.</td></tr>
<tr><td>A <code>.vhdr</code> file does not open</td>
<td>Its <code>.eeg</code> and <code>.vmrk</code> files must be in the same folder under the names
written inside the <code>.vhdr</code>. Renaming one of the three by hand breaks the link.</td></tr>
<tr><td>Program fails to open after copying</td>
<td>Copy the whole <code>GIOVA</code> folder (with <code>_internal</code>), not only the exe.
<code>GIOVA.exe --selftest</code> checks the bundle.</td></tr>
</table>
<p class="note">The mouse wheel never changes a value box, list or slider: it scrolls the
panel. Type the value or use the arrows. The wheel still zooms plots.</p>
"""

QUALITY_TABLE = """
<h2>Quality table, item by item</h2>
<p>The table on the Monitor gives every channel one row. All of it is computed from the
<b>last 4 s of raw samples</b> (before the display filters and the notch), about once a
second. These are <b>signal checks, not impedance</b>: for impedance use Controls → Board
channels → Impedance check. Limits are editable in <b>Settings → Monitor</b>.</p>

<h3>Columns</h3>
<p><b>Channel</b>: the electrode name from the channel profile. Click a row to open that
channel's spectrum.</p>
<p><b>Status</b>: the verdict, one word. What each one means is below.</p>
<p><b>EEG µV</b>: how large the signal is inside the EEG band, 1–45 Hz:
<code>√ ∫ PSD(f) df</code> over 1–45 Hz, after the linear drift is removed. Awake scalp EEG is
typically <b>5–50 µV</b>; eyes closed raises it at the back of the head (alpha). Near 0 means
nothing is being measured; above 100 µV is usually movement, blinking, jaw or a loose
electrode. Band-limited on purpose: the raw RMS is mostly electrode drift and mains, which the
display filters remove anyway.</p>
<p><b>Rail %</b>: the share of the 4 s whose samples sit within 5 % of the amplifier's limit
(the <i>rail</i>): ±187,500 µV on the Cyton at gain 24, ±750,000 µV on the Unicorn. A good
channel reads <b>0 %</b>. A railed sample carries no EEG at all: the input is so far off that
the amplifier is clipped. Typical cause: the electrode is not on the skin, or the reference
(SRB2 / ear clip) is not connected.</p>
<p><b>50 Hz %</b>: how much of the 1–60 Hz power is mains hum, 49–51 Hz. Where the mains is
60 Hz (the Americas, Korea, Japan), choose 60 Hz as the filters' Mains: the column becomes
<b>60 Hz %</b>, the share of 59–61 Hz in 1–70 Hz. A good contact
reads a few %. High values mean the electrode picks up the room's wiring: poor contact, a
dry electrode, a cable lying near a power lead or a charger. The notch filter removes the mains
from the <i>traces</i>, so a channel can look clean on screen and still read high here; this
column is measured on raw data for that reason. It shows — when the sampling rate is too low
to see the mains.</p>

<h3>Statuses (first match wins, in this order)</h3>
<p><b>Collecting data</b>: less than 4 s since Start, or since the channels changed. Nothing
is judged yet.</p>
<p><b>Missing data</b> / <b>Stale</b>: the channel has non-numeric samples, or no new data
arrived for 2 s (board disconnected, Bluetooth dropped, cable out). Check the connection
first.</p>
<p><b>Saturated</b>: rail share <b>&gt; 15 %</b>. The channel is clipped for a good part of the
time and has no usable EEG. Seat the electrode on skin (move hair aside, add gel or saline),
and check the reference clip. A channel with nothing plugged in also saturates.</p>
<p><b>Flat</b>: EEG µV <b>&lt; 0.1</b>. The channel does not move: switched off on the board
(Board channels → On), not wired, or shorted. An OFF channel reads exactly 0.</p>
<p><b>Check signal</b>: EEG µV <b>&gt; 100</b>. Too large for scalp EEG. Short bursts are normal
(a blink on frontal channels, a head turn, chewing); if it stays, the electrode is loose or
moving, or muscle tension is high: ask the person to relax the jaw and forehead, then press
the electrode gently.</p>
<p><b>Identical signal</b>: this channel and another one carry the same waveform and offset
(they differ by less than 2 %). Two electrodes centimetres apart never do that on a head.
Either both are not touching the skin and float together, or gel / sweat bridges them. The
detail names the partner channel.</p>
<p><b>Line noise</b>: mains (50 or 60 Hz) share <b>&gt; 20 %</b>. The EEG is there but buried in mains hum.
Improve the contact of that electrode, and move cables away from power leads and
chargers; a laptop on its charger often makes it worse.</p>
<p><b>No issues detected</b>: none of the checks above fired. It means the signal is plausible,
not that it is perfect: a channel can pass and still be mostly muscle or eye movement. Look
at the trace and the spectrum too.</p>

<h3>Why a status changes slowly</h3>
<p>A warning must repeat in <b>2</b> evaluations in a row before it shows, and a channel must
be clean <b>3</b> times before the warning clears, so a single spike does not make the table
flicker. <i>Missing data</i> and <i>Saturated</i> show at once. The very first clean window
after start shows <i>No issues detected</i> immediately.</p>
"""


WORKBENCH = """
<h2>Workbench</h2>
<p>The Workbench builds an analysis from <b>blocks</b>. Each block does one step; wires carry its result to the
next. The same pipeline runs on any recording and can be saved, shared and opened again. What each block does
and cannot do: <b>Block library</b>, in this window's list.</p>
<h3>Building</h3>
<table>
<tr><th>To</th><th>Do</th></tr>
<tr><td>Start</td><td><b>Library</b> tab, left: <b>Mine</b> (your saved pipelines, with their last accuracy),
<b>Templates</b>, <b>Literature</b>. Double-click opens one in a new tab. <b>+</b> beside the tabs: an empty
sheet</td></tr>
<tr><td>Add a block</td><td>Point at it in <b>Blocks</b> to see it as it will look; drag it onto the sheet, or
double-click it: it is added after the selected block and wired to it</td></tr>
<tr><td>Wire</td><td>Drag from a port on the right of a block to one on the left of the next. Colours: blue
recording, green trials, amber spectrum, violet features, grey model, red scores</td></tr>
<tr><td>Settings</td><td>Select the block: its settings are on the right, rare ones behind <b>Options</b>. A
setting that means nothing with the method or design chosen (an STFT window under Morlet) is hidden; its value
is kept for when it applies again</td></tr>
<tr><td>Help on a block</td><td>Right-click it, <b>Help</b></td></tr>
<tr><td>Remove</td><td>Select a block or wire, Delete</td></tr>
<tr><td>Move around</td><td>The sheet has no edge. Drag on empty sheet, scroll with two fingers or the wheel, or
hold Space (or the middle button) and drag; a block or a wire dragged to the sheet's edge moves it along. Shift + drag
selects several. In Settings → Interface a plain drag can select instead (Space or the middle button then moves the
sheet)</td></tr>
<tr><td>Zoom</td><td>Pinch, Ctrl + wheel or + / − zoom toward the mouse, from 5 % to 400 %. The zoom bar in the
sheet's lower left corner shows the level; its menu offers 50–200 %, the whole pipeline and the selected blocks.
Keys: <b>0</b> 100 %, <b>1</b> (or F) the whole pipeline, <b>2</b> the selected blocks. A tab opens where it was
left</td></tr>
</table>
<h3>Running</h3>
<table>
<tr><th>To</th><th>Do</th></tr>
<tr><td>Run</td><td><b>Run</b> (F5): every block that has its inputs. Right-click a block, <b>Run up to here</b></td></tr>
<tr><td>Follow it</td><td>The strip along the bottom: two mirrored waves move in it while the run goes on, and
it names the block being computed, how many are done, the time spent and the time left</td></tr>
<tr><td>Stop</td><td>While it runs, <b>Run</b> turns into <b>Stop</b> in its place (F5 too): the blocks already
computed keep their results; the next Run goes on from there</td></tr>
<tr><td>Results and Log</td><td>Click the strip: under the sheet, <b>Results</b> (Compare) and the <b>Log</b> of
every run, its time, its blocks and its errors. It never opens by itself; an error is counted on the strip</td></tr>
<tr><td>Read a block's state</td><td>Green dot: computed. Amber: a setting changed, computed again at the next
Run. Red: an error, said under the block</td></tr>
</table>
<p>Blocks whose settings did not change are not computed again, so trying another filter is quick.</p>
<h3>Files</h3>
<p><b>Save</b> writes a <code>.pipeline.json</code>: blocks, settings and their places, not the data. Open tabs
come back at the next start.</p>
"""

WORKBENCH_RESULTS = """
<h2>Workbench: results and tabs</h2>
<h3>Looking at a result</h3>
<table>
<tr><th>Choice</th><th>Shows</th></tr>
<tr><td>A channel</td><td>That channel</td></tr>
<tr><td><b>All channels</b></td><td>Up to 64 side by side, one scale, in head order (Fp, AF, F, FC, C, CP, P, PO, O;
left to right)</td></tr>
<tr><td><b>Mean of channels...</b></td><td>A region (Fz, Cz, Pz) averaged</td></tr>
<tr><td><b>Head maps over time</b></td><td>(Time-frequency) one band's head map per stretch of time, and a slider.
Type any band (<i>6-20</i>) and Enter; <b>Add to block</b> keeps it in the block's Bands</td></tr>
<tr><td><b>Each trial</b></td><td>(Trials) the trials one above the other; in Select trials, click one to leave it out</td></tr>
</table>
<table>
<tr><th>Button</th><th>Does</th></tr>
<tr><td><b>Larger</b></td><td>The result in its own window, with the same choices; they come back when it closes</td></tr>
<tr><td><b>Zoom</b> / <b>Reset</b></td><td>Graphs do not zoom with the wheel (it scrolls the column): Zoom to zoom
and drag, Reset to fit</td></tr>
<tr><td><b>Save image</b></td><td>A sharp PNG</td></tr>
<tr><td><b>Save every trial...</b></td><td>(Time-frequency) one PNG per trial and per condition's average</td></tr>
<tr><td><b>Play...</b></td><td>(Recording) the signal scrolling like a video, with its spectrum and spectrogram</td></tr>
</table>
<h3>Tabs</h3>
<table>
<tr><th>To</th><th>Do</th></tr>
<tr><td>Open another pipeline</td><td><b>+</b> beside the tabs, <b>New</b>, <b>Open</b>, or the Library</td></tr>
<tr><td>Run every tab</td><td><b>Run all tabs</b> (beside Run): two compute at once, the others wait (<i>queued</i>)</td></tr>
<tr><td>Use a second screen</td><td>Drag a tab off the bar; close the window to put it back</td></tr>
<tr><td>Compare the open tabs</td><td><b>Compare</b>, then <b>+</b> beside Pipelines, <b>The other open tabs</b></td></tr>
<tr><td>Try a recording</td><td>Double-click a recording in Projects: it goes into the Recording block and runs</td></tr>
</table>
<p>A tab says how it stands: <i>queued</i>, its progress, <i>accuracy</i>, <i>error</i>, <i>stopped</i>;
* means unsaved changes.</p>
"""

WORKBENCH_CLASSIFY = """
<h2>Workbench: classification</h2>
<p>A model is a chain of steps under <b>Machine learning</b>, a classifier last, wired into the <b>model</b>
input of <b>Cross-validate</b>; the trials or features go into its <b>data</b> input.</p>
<table>
<tr><th>Typical chain</th><th>Blocks</th></tr>
<tr><td>CSP + LDA</td><td>Epochs into Cross-validate's data; CSP, LDA into its model</td></tr>
<tr><td>Band power + LDA</td><td>Epochs, Welch PSD, Band power into data; Standardize, LDA into model</td></tr>
<tr><td>Riemannian</td><td>Epochs into data; Covariance, Tangent space, Logistic regression (or Covariance, MDM) into model</td></tr>
</table>
<h3>What Cross-validate gives</h3>
<table>
<tr><th>Result</th><th>Meaning</th></tr>
<tr><td>Accuracy ± SD</td><td>Mean of the folds' accuracies; ± is their standard deviation across folds</td></tr>
<tr><td>Balanced accuracy</td><td>Mean of each condition's hit rate: honest with unequal trial counts</td></tr>
<tr><td>Cohen's &kappa;</td><td>Agreement beyond chance, from the confusion matrix: 0 is chance, 1 every trial
right; comparable across different numbers of conditions (Cohen, 1960; Schlögl et al., 2007)</td></tr>
<tr><td>F1 (macro)</td><td>Each condition's F1 (the harmonic mean of its precision and hit rate), averaged
without weighting; a condition never predicted counts 0</td></tr>
<tr><td>ROC AUC, ROC curve</td><td>Two conditions: how well the scores rank the trials</td></tr>
<tr><td>Confusion matrix</td><td>Which condition was taken for which</td></tr>
<tr><td>Compared with chance</td><td>Where the accuracy stands against what a random classifier reaches with this
many trials: <i>clearly above chance</i> (past the alpha 0.01 limit), <i>above chance</i> (past 0.05), or
<i>within what chance can reach</i></td></tr>
<tr><td>Permutation p</td><td>Only when run (Options, Permutation test)</td></tr>
</table>
<h3>Choices</h3>
<table>
<tr><th>Setting</th><th>Use</th></tr>
<tr><td>Split</td><td>k-fold (default); repeated k-fold for a steadier mean; chronological to train on early trials and test on late ones</td></tr>
<tr><td>Tune</td><td>On a classifier: its settings searched inside each training fold</td></tr>
<tr><td>Permutation test</td><td>99 for a look, 999 for a report; each repeats the whole cross-validation</td></tr>
<tr><td>Seed</td><td>Same seed, same folds, same result</td></tr>
</table>
<p>How each of these is computed: <b>Workbench: how it computes</b>.</p>
"""

WORKBENCH_SHARE = """
<h2>Workbench: compare, report, Python</h2>
<table>
<tr><th>Button</th><th>What it does</th></tr>
<tr><td><b>Compare</b></td><td>Opens under the sheet, in Results, as three steps: <b>1 Pipelines</b>, <b>2 Recordings</b>,
<b>3 Compare</b> (<b>+</b> adds, <b>−</b> removes; Compare turns into Stop while it runs). Runs pipelines over many
recordings with the same folds: each recording's accuracy
under each pipeline, the means, and a Wilcoxon test between pipelines. Double-click a recording for its full
result. <b>Across recordings</b> tests each recording with a model trained on all the others (a new person only when each file is a different person);
reduce raw trials first, it holds them all at once</td></tr>
<tr><td><b>Report</b></td><td>After a run: the pipeline, the methods paragraph, <b>Data handling</b> (how
each file was read, the trials used, every trial left out with its reason), every block's result with its
figure, the scores, software versions, references. PDF, Word, web page or PNG figures, always on white</td></tr>
<tr><td><b>Library, Literature</b></td><td>Published methods (Barachant 2012 MDRM, Lotte &amp; Guan 2011 TRCSP,
Ang 2012 FBCSP) set up for Cho2017, each with its paper, DOI and how it differs from the paper. Its accuracy
here is not the paper's</td></tr>
<tr><td><b>Python</b></td><td>The pipeline as Python that runs without GIOVA and gives the same numbers.
A zip: <code>run.py</code> (one line per block), <code>eeg_studio_code.py</code>, <code>requirements.txt</code>,
a README. Unzip, <code>pip install -r requirements.txt</code>, <code>python run.py</code></td></tr>
</table>
<h3>Other ways in</h3>
<table>
<tr><th>From</th><th>Opens</th></tr>
<tr><td><b>Quick analysis</b> (Home, Ctrl+3, a recording's right-click)</td><td>A ready profile's pipeline on a recording, in a new tab</td></tr>
<tr><td>A saved <code>.analysis.json</code></td><td>Its pipeline, with its own trials in the Epochs block's table</td></tr>
</table>
"""

WORKBENCH_MATH = """
<h2>Workbench: how it computes</h2>
<h3>Cross-validation</h3>
<p>Every step before the classifier (CSP, selection, scaling, augmentation) is fitted again in each fold on
that fold's training trials only; no test trial shapes the features. <b>Tune</b> searches a classifier's
settings by a cross-validation inside each training fold (nested), fitting the whole chain again. Its
<b>Options</b> hold the values searched (C values, Depth values…), the <i>Inner folds</i> (3) and <i>Tune by</i>:
accuracy, or balanced accuracy when the conditions have unequal trials. With two tuned steps (TRCSP and an SVM,
say) every combination is tried, with the first tuned step's inner folds and score. Without Tune, the values
in Options (an SVM's gamma, a forest's depth and leaf size) are used as given.
Accuracy is the mean of the folds' accuracies; ± is their standard deviation, not a confidence interval.
Preprocessing before the trials (Filter, ICA) is not part of the folds.</p>
<h3>Compared with chance</h3>
<p>A random classifier's accuracy spreads around chance, the more so the fewer the trials, so 50% is not the
line to beat with few trials (Müller-Putz et al., 2008; Combrisson &amp; Jerbi, 2015). GIOVA gives the upper
limit of that spread for the number of tested trials, at alpha 0.05 and 0.01, and says where the accuracy
stands in three steps rather than a yes or no: <i>clearly above chance</i> past the 0.01 limit, <i>above
chance</i> past the 0.05 one, <i>within what chance can reach</i> below it. Two conditions: the exact binomial
limit (Combrisson &amp; Jerbi, 2015); more: a simulation. With 100 trials the 0.05 limit is 60%: 60 correct is
within what chance can reach, 61 above it. With unequal trials (90 and 10) always guessing the larger scores
90%, so balanced accuracy must also beat a random classifier simulated for the same trials.</p>
<p>The limit takes the tested trials as independent. Cross-validated ones are not: the folds share their training
trials, and on noise a cross-validated accuracy passes the limit more often than alpha says (Noirhomme et al.,
2014). So the limit is approximate, a first look. A permutation test does not assume independence: run one
(<i>Permutation test</i> under the result, or in Cross-validate's Options) before calling a result above
chance.</p>
<h3>Permutation test</h3>
<p>The whole cross-validation is run again with the labels shuffled, B times; p = (1 + the number that score
as well or better) / (B + 1) (Ojala &amp; Garriga, 2010). The smallest p is 1 / (B + 1): 0.01 for 99. When it is
run, being above chance also needs p ≤ 0.05. Trials are shuffled as independent units. Under a result the limit
alone judged, <i>Permutation test</i> takes the number of shuffles (200, p down to 0.005; pick another or type any)
and <i>Run</i> runs it; the number stays in Cross-validate's Options.</p>
<h3>Time left</h3>
<p>Guessed from the pace of the current block's own steps (its folds, its permutations) once it has done a
few; quick blocks before it say nothing of how long a cross-validation takes. <i>in this block</i>: more blocks
follow.</p>
<h3>Time-frequency</h3>
<p>ERD / ERS per electrode: the band's power (its frequencies averaged), the trials' power averaged, then
100 × (power − baseline) / baseline. The <b>head map</b> of the active window and the <b>statistics</b> are window
means. By default (<i>Head map and statistics from: stft</i>) they come from a separate STFT of each electrode (the
block's STFT window, at most 50 % overlap, not smoothed), whatever the method of the maps: one transform across
the scalp, and fast on 64 electrodes. With <i>maps</i> they come from the maps' own transform, not smoothed, so
everything shown comes from one transform (slower with wavelets on many electrodes). The head map's power is then
in the maps' units: µV² for wavelets, µV²/Hz for an STFT. <b>Head maps over time</b> are drawn from the maps
themselves, on the analysed channels. Every head map places the electrodes at the standard 10-05 positions of
their names (MNE's template), not at positions measured on the person's head. Amplitude: a sine of amplitude A reads A (half of what scipy's STFT
gives). <b>wavelet</b>: complex Morlet wavelets. Their width is given in one of two ways: <i>cycles per Hz</i>
(each wavelet spans that many cycles per Hz of its frequency, 2 to 12 cycles, about 40 frequencies), or
<i>one wavelet</i> cmorB-C as PyWavelets names it, the same for every frequency (128 frequencies, computed from
its formula). The two are one transform: cmorB-C is a Morlet wavelet of 2πC√(B/2) cycles, so cmor1.5-1.0, the
lab scalogram's, spans 5.44 cycles at every frequency. Pipelines saved with the methods <b>morlet</b> and
<b>cwt</b> open as these two and give the same numbers.</p>
<h3>Filters and wavelets</h3>
<p>IIR filters are designed by SciPy, FIR as MNE-Python does by default. <b>ideal</b> sets every FFT bin outside
the band to zero: sharp, but it rings in time. <b>Wavelet denoise</b>: 0 µV means the universal threshold of
Donoho &amp; Johnstone, per trial and channel. <b>DWT bands</b>: each band's share of the energy adds up to 100%.</p>
<h3>Compare windows</h3>
<p>Each window's Welch power from segments of the same length; log power compared trial by trial (Wilcoxon
signed-rank or paired t-test); p values adjusted across channels and bands by Holm's method.</p>
"""

TOPICS = [
    ("Quick start", QUICK_START),
    ("Projects and side panes", PROJECTS),
    ("Channel profiles", CHANNEL_PROFILES),
    ("Quality table explained", QUALITY_TABLE),
    ("Monitor: how it computes", MONITOR_MATH),
    ("Parameters explained", PARAMETERS),
    ("Impedance check", IMPEDANCE_MATH),
    ("Workbench", WORKBENCH),
    ("Workbench: results and tabs", WORKBENCH_RESULTS),
    ("Workbench: classification", WORKBENCH_CLASSIFY),
    ("Workbench: compare, report, Python", WORKBENCH_SHARE),
    ("Workbench: how it computes", WORKBENCH_MATH),
    ("Analysis & features", ANALYSIS_MATH),
    ("Recording files", RECORDING_FILES),
    ("Troubleshooting", TROUBLESHOOTING),
    ("About GIOVA", ABOUT),
]

#: How the topics are grouped in the Help window's list; the block library is added between Workbench and Reference.
SECTIONS = [
    ("Getting started", ["Quick start", "Projects and side panes", "About GIOVA"]),
    ("Monitor", ["Channel profiles", "Quality table explained", "Impedance check", "Monitor: how it computes"]),
    ("Workbench guide", ["Workbench", "Workbench: results and tabs", "Workbench: classification",
                         "Workbench: compare, report, Python", "Workbench: how it computes"]),
    ("Reference", ["Parameters explained", "Analysis & features", "Recording files", "Troubleshooting"]),
]

#: The topic opened from each main tab.
TAB_TOPICS = {"Home": "Quick start", "Monitor": "Quick start", "Workbench": "Workbench", "Analysis": "Analysis & features",  # older mode name
              "Games": "Quick start"}
