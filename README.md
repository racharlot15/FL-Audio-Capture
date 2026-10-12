# FL Studio stereo capture test (Windows)

This local prototype streams two input channels to a stereo WAV and analyzes saved recordings. It reports left/right RMS and peak levels in dBFS, channel balance, stereo correlation and width, mono fold-down level, crest factor, six frequency-band energy levels, silence and clipping durations, and level changes over time. Capture and analysis use bounded blocks; processing stays on this computer, and no audio is uploaded. It does not control FL Studio or interpret musical content like a human listener.

## Capture route and limits

With FL Studio set to **Focusrite USB ASIO**, it sends audio directly through the Focusrite ASIO driver. Windows WASAPI loopback records the mix on a Windows playback endpoint; it is not a dependable way to capture a DAW stream that bypasses the Windows audio engine, and it may include other apps ([Microsoft WASAPI loopback documentation](https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording)). Windows process-specific loopback also depends on the app using the Windows audio engine. This prototype therefore captures a normal Windows recording input. For an FL-only test, route only FL Studio's playback into a stereo Focusrite hardware Loopback input/mix, then select that exposed input below. Availability and routing controls vary by Scarlett generation/model and driver.

On Focusrite interfaces that support Loopback, configure its Loopback mix in Focusrite Control / Focusrite Control 2 to include the FL Studio playback channels ([Focusrite Loopback setup](https://support.focusrite.com/hc/en-gb/articles/13229216604562-Scarlett-4th-Gen-using-Loopback), [supported models](https://support.focusrite.com/hc/en-gb/articles/27140338179858-Which-Focusrite-interfaces-feature-Loopback)). For the strongest isolation, route FL Studio to a dedicated playback pair and include only that pair in the Loopback mix, if the interface and FL Studio output configuration support it. Make sure the chosen input is exposed to non-ASIO Windows applications (Focusrite Notifier may offer **Expose/Hide Windows Channels**; see [Focusrite channel exposure steps](https://support.focusrite.com/hc/en-gb/articles/360010711620-Expose-Hide-Windows-Channels-for-non-ASIO-Windows-apps)). Some models do not provide hardware Loopback. If no Focusrite stereo loopback input appears, the options are to use a Windows playback endpoint only if FL Studio is actually routed through it, or use an available spare physical output-to-line-input loopback with suitable cables and safe levels. Do not connect a line output to a microphone input with phantom power enabled. Avoid opening the same Focusrite ASIO driver from a second app while FL Studio is using it; many ASIO drivers are exclusive.

The capture tool cannot identify which app produced samples by itself. FL-only capture depends on the Focusrite routing being configured as described. Playback monitoring is not changed by this program.

## Install and run

Use Python 3.10 or newer. Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python capture.py list
```

### Desktop interface

After installing the requirements, double-click `run-ui.bat` or launch the graphical interface from this folder with:

```powershell
python ui.py
```

The **Capture** tab lists stereo-capable input devices, lets you choose a duration up to 120 seconds and WAV location, and saves the recording plus its measurement JSON. Live left/right RMS meters and peak markers update during recording. **Check input** listens to the selected device for two seconds without saving a file and reports channel levels and buffer overflows. After a successful recording, **Analyze latest capture** starts analysis of its WAV automatically. You can stop a capture early; it saves the audio received so far and marks the report as interrupted. Analysis can also be stopped; the partial report is saved with an interruption flag and the chart shows the analyzed portion. The **Analyze** tab reads a stereo PCM WAV and writes a detailed analysis JSON with a left/right RMS timeline chart. You can request windows up to 30 seconds; for very long recordings, the tool widens them automatically when needed to keep the report at or below 10,000 timeline windows. The report shows the effective window size. The **Compare** tab accepts two complete analysis reports, applies an optional listening preference, and writes the comparison JSON with a frequency-band change chart. Stereo-wide levels combine left/right channel power before conversion to dB. Long capture, analysis, and comparison operations run in the background so the window remains responsive. The interface checks that output files do not already exist before starting, to protect prior recordings and reports, and keeps the window open while an operation is active so its result can be displayed.

The interface uses Python's built-in Tkinter library. If your Python installation was installed without Tcl/Tk support, install a Python build that includes Tkinter.

Start a test tone or known stereo material in FL Studio, then choose the listed Focusrite Loopback stereo input index:

```powershell
python capture.py capture --device 3 --seconds 10 --output fl-studio-test.wav
```

Replace `3` with the device index shown on your computer. The command saves a stereo WAV plus a JSON report beside it. RMS silence is defined as below -60 dBFS; clipping is flagged at magnitude 0.999 or greater. `left_right_rms_delta_db` is left RMS minus right RMS. `waveform_difference_rms_dbfs` measures the RMS of L-R and will be low for similar channels, including dual-mono content.

## Analyze a recording locally

Analyze a captured WAV in quarter-second windows:

```powershell
python capture.py analyze fl-playing.wav
```

This writes `fl-playing.analysis.json` beside the WAV. The report includes whole-file channel levels and a `timeline` list with RMS, peak, silence, and clipping flags for each window. Use `--window-seconds 1` for a less detailed one-second timeline, or `--output my-report.json` to choose the report path. Only uncompressed stereo PCM WAV files are supported. The analysis describes measured audio levels and stereo relationships; it does not transcribe speech or identify instruments, notes, or musical meaning.

The mix details are measurements for comparison, not pass/fail mix judgments. Frequency-band percentages show how analyzed energy from 20 Hz to the lower of 20 kHz or Nyquist is distributed among sub-bass (20–60 Hz), bass (60–250 Hz), low mids (250–500 Hz), mids (500 Hz–2 kHz), high mids (2–6 kHz), and highs (6–20 kHz). `stereo_image.side_to_mid_db` compares side-channel energy with mid-channel energy; more negative values mean a narrower image. `mono_fold_down_rms_dbfs` estimates level after averaging left and right. Crest factor is peak level minus RMS level. These values depend on the excerpt and arrangement; they do not by themselves say whether a mix is balanced or sounds good.

## Compare a mix change and get a local suggestion

Analyze the baseline and changed WAVs first, then compare their reports. Pass the listening preference that describes the change:

```powershell
python capture.py compare bass-original-10s.analysis.json bass-minus3.analysis.json --preference tighter
```

The command prints measured changes and a suggestion, then writes a comparison JSON beside the trial report. The suggestion combines the measured changes with your stated preference. It is an explainable local rule in this prototype, not a trained model; the preference is what lets it distinguish a desired creative choice from a raw meter change.

## Automated checks

```powershell
python -m unittest -v
```

The automated checks validate capture metrics, WAV analysis, and before/after suggestions using synthetic samples. They do not prove that the Focusrite route is available or that FL Studio is routed into it; that requires the device-list and live capture steps on the target PC.
