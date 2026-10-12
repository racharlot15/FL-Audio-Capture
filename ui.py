"""Desktop interface for the local FL Studio stereo capture and analysis tools."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd

from capture import analyze_wav, capture, compare_reports, measure_stereo


BG = "#10141c"
PANEL = "#181f2b"
TEXT = "#e8edf5"
MUTED = "#9aa8ba"
ACCENT = "#77e0b5"
WARN = "#ffb86b"


class CaptureApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FL Audio Capture")
        self.geometry("1060x760")
        self.minsize(850, 650)
        self.configure(bg=BG)
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.devices: list[tuple[int, str]] = []
        self.capture_output_auto = True
        self._updating_capture_path = False
        self._style()
        self._build()
        self.refresh_devices()
        self.after(100, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._request_close)

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 22))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI Semibold", 11))
        style.configure("TButton", padding=(12, 8), font=("Segoe UI Semibold", 9))
        style.configure("Accent.TButton", background=ACCENT, foreground="#0d1715")
        style.map("Accent.TButton", background=[("disabled", "#536b65"), ("active", "#a0f0d0")])
        style.configure("TEntry", fieldbackground="#202938", foreground=TEXT, insertcolor=TEXT, padding=7)
        style.configure("TCombobox", fieldbackground="#202938", foreground=TEXT, padding=6)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(18, 10), font=("Segoe UI Semibold", 9))
        style.map("TNotebook.Tab", background=[("selected", "#263446")], foreground=[("selected", TEXT)])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#263040", bordercolor="#263040")

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=(26, 22, 26, 14))
        outer.pack(fill="both", expand=True)
        head = ttk.Frame(outer)
        head.pack(fill="x", pady=(0, 18))
        ttk.Label(head, text="FL Audio Capture", style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, text="Capture a routed stereo signal, inspect its levels, and compare mix changes. Audio stays on this computer.", style="Subtitle.TLabel").pack(anchor="w", pady=(4, 0))

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.capture_tab = ttk.Frame(self.tabs, padding=22)
        self.analyze_tab = ttk.Frame(self.tabs, padding=22)
        self.compare_tab = ttk.Frame(self.tabs, padding=22)
        self.tabs.add(self.capture_tab, text="  Capture  ")
        self.tabs.add(self.analyze_tab, text="  Analyze  ")
        self.tabs.add(self.compare_tab, text="  Compare  ")
        self._build_capture()
        self._build_analyze()
        self._build_compare()

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(12, 0))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=130)
        self.progress.pack(side="right")
        self.stop_button = ttk.Button(footer, text="Stop", command=self.stop_operation, state="disabled")
        self.stop_button.pack(side="right", padx=(0, 10))

    def _section(self, parent: ttk.Frame, title: str, description: str) -> None:
        ttk.Label(parent, text=title, style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text=description, style="Muted.TLabel", wraplength=770).pack(anchor="w", pady=(4, 16))

    def _row(self, parent: ttk.Frame, label: str, var: tk.StringVar, browse=None, width=70) -> ttk.Entry:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text=label, width=19).pack(side="left", anchor="w")
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side="left", fill="x", expand=True)
        if browse:
            ttk.Button(row, text="Browse…", command=browse).pack(side="left", padx=(8, 0))
        return entry

    def _build_capture(self) -> None:
        tab = self.capture_tab
        self._section(tab, "Record a stereo take", "Choose the Focusrite Loopback input (or another stereo input that receives FL Studio). Recording starts only when you press Capture.")
        route_card = ttk.Frame(tab, style="Panel.TFrame", padding=(14, 11))
        route_card.pack(fill="x", pady=(0, 10))
        ttk.Label(route_card, text="Route check", style="Panel.TLabel", font=("Segoe UI Semibold", 10)).pack(anchor="w")
        ttk.Label(route_card,
                  text="Route FL Studio through Focusrite Loopback and expose that stereo input to Windows. The capture can include other apps if they share the loopback mix; this tool cannot identify the source. Avoid opening the same exclusive ASIO driver in another app, and never connect a line output to a mic input with phantom power enabled.",
                  style="PanelMuted.TLabel", wraplength=760, justify="left").pack(anchor="w", pady=(4, 0))
        meter_header = ttk.Frame(tab)
        meter_header.pack(fill="x", pady=(4, 5))
        ttk.Label(meter_header, text="Live input levels", style="Section.TLabel").pack(side="left")
        ttk.Label(meter_header, text="RMS bar · peak marker", style="Muted.TLabel").pack(side="right")
        self.level_canvas = tk.Canvas(tab, height=84, bg=PANEL, highlightthickness=0)
        self.level_canvas.pack(fill="x", pady=(0, 8))
        self.live_levels = None
        self.level_canvas.bind("<Configure>", lambda _event: self._draw_live_levels())
        self._draw_live_levels()
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="Input device", width=19).pack(side="left", anchor="w")
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(row, textvariable=self.device_var, state="readonly")
        self.device_box.pack(side="left", fill="x", expand=True)
        self.check_button = ttk.Button(row, text="Check input", command=self.check_input)
        self.check_button.pack(side="left", padx=(8, 0))
        self.refresh_button = ttk.Button(row, text="Refresh", command=self.refresh_devices)
        self.refresh_button.pack(side="left", padx=(8, 0))
        self.seconds_var = tk.StringVar(value="10")
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="Duration (seconds)", width=19).pack(side="left", anchor="w")
        ttk.Entry(row, textvariable=self.seconds_var, width=14).pack(side="left")
        ttk.Label(row, text="  Choose a duration from 0.1 to 120 seconds", style="Muted.TLabel").pack(side="left")
        self.capture_output = tk.StringVar(value=str(self._new_default_capture_path()))
        self.capture_output.trace_add("write", self._capture_path_edited)
        self._row(tab, "Save WAV to", self.capture_output, self._browse_capture_output)
        ttk.Label(tab, text="The WAV and its measurement JSON are saved beside the selected path. Existing files are protected from accidental overwrite.", style="Muted.TLabel", wraplength=780).pack(anchor="w", pady=(8, 18))
        self.capture_button = ttk.Button(tab, text="Capture stereo audio", style="Accent.TButton", command=self.start_capture)
        self.capture_button.pack(anchor="w")
        self.analyze_capture_button = ttk.Button(tab, text="Analyze latest capture", command=self.analyze_latest_capture, state="disabled")
        self.analyze_capture_button.pack(anchor="w", pady=(8, 0))
        self.capture_result = self._result_box(tab, height=11)

    def _build_analyze(self) -> None:
        tab = self.analyze_tab
        self._section(tab, "Analyze a saved recording", "Read a stereo PCM WAV and create a local JSON report with levels, stereo image, frequency bands, and a timeline.")
        self.analyze_input = tk.StringVar()
        self._row(tab, "Stereo WAV", self.analyze_input, self._browse_wav)
        self.window_var = tk.StringVar(value="0.25")
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="Timeline window", width=19).pack(side="left", anchor="w")
        ttk.Entry(row, textvariable=self.window_var, width=14).pack(side="left")
        ttk.Label(row, text=" seconds (maximum 30)", style="Muted.TLabel").pack(side="left")
        self.analysis_output = tk.StringVar()
        self._row(tab, "Report JSON", self.analysis_output, self._browse_analysis_output)
        self.analyze_button = ttk.Button(tab, text="Analyze recording", style="Accent.TButton", command=self.start_analyze)
        self.analyze_button.pack(anchor="w", pady=(14, 10))
        chart_header = ttk.Frame(tab)
        chart_header.pack(fill="x", pady=(4, 5))
        ttk.Label(chart_header, text="RMS over time", style="Section.TLabel").pack(side="left")
        ttk.Label(chart_header, text="Left", foreground="#77e0b5").pack(side="right", padx=(12, 0))
        ttk.Label(chart_header, text="Right", foreground="#8ab4ff").pack(side="right")
        self.timeline_canvas = tk.Canvas(tab, height=150, bg=PANEL, highlightthickness=0)
        self.timeline_canvas.pack(fill="x", expand=False)
        self.timeline_canvas.bind("<Configure>", lambda _event: self._draw_timeline())
        self.timeline_data = None
        self.timeline_duration = 0.0
        self._draw_timeline()
        self.analyze_result = self._result_box(tab, height=15)

    def _build_compare(self) -> None:
        tab = self.compare_tab
        self._section(tab, "Compare two analysis reports", "Compare a baseline with a changed mix. The suggestion combines measurements with your stated preference; it does not replace listening.")
        self.before_var = tk.StringVar()
        self.trial_var = tk.StringVar()
        self._row(tab, "Baseline report", self.before_var, self._browse_before)
        self._row(tab, "Trial report", self.trial_var, self._browse_trial)
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="Listening preference", width=19).pack(side="left", anchor="w")
        self.preference_var = tk.StringVar(value="No preference")
        ttk.Combobox(row, textvariable=self.preference_var, state="readonly", values=("No preference", "Tighter", "Fuller"), width=22).pack(side="left")
        self.comparison_output = tk.StringVar()
        self._row(tab, "Comparison JSON", self.comparison_output, self._browse_comparison_output)
        self.compare_button = ttk.Button(tab, text="Compare reports", style="Accent.TButton", command=self.start_compare)
        self.compare_button.pack(anchor="w", pady=(14, 10))
        chart_header = ttk.Frame(tab)
        chart_header.pack(fill="x", pady=(4, 5))
        ttk.Label(chart_header, text="Frequency-band shifts", style="Section.TLabel").pack(side="left")
        ttk.Label(chart_header, text="More energy", foreground="#77e0b5").pack(side="right", padx=(12, 0))
        ttk.Label(chart_header, text="Less energy", foreground="#ffb86b").pack(side="right")
        self.band_canvas = tk.Canvas(tab, height=170, bg=PANEL, highlightthickness=0)
        self.band_canvas.pack(fill="x", expand=False)
        self.band_data = None
        self.band_canvas.bind("<Configure>", lambda _event: self._draw_band_chart())
        self._draw_band_chart()
        self.compare_result = self._result_box(tab, height=15)

    def _result_box(self, parent: ttk.Frame, height: int) -> tk.Text:
        wrap = ttk.Frame(parent, style="Panel.TFrame", padding=1)
        wrap.pack(fill="both", expand=True, pady=(18, 0))
        scrollbar = ttk.Scrollbar(wrap, orient="vertical")
        box = tk.Text(wrap, height=height, wrap="word", bg=PANEL, fg=TEXT, insertbackground=TEXT,
                      relief="flat", padx=14, pady=12, font=("Consolas", 9), state="disabled",
                      yscrollcommand=scrollbar.set)
        scrollbar.configure(command=box.yview)
        scrollbar.pack(side="right", fill="y")
        box.pack(side="left", fill="both", expand=True)
        return box

    def _write(self, box: tk.Text, value: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", value)
        box.configure(state="disabled")

    @staticmethod
    def _write_new_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            json.dump(value, output, indent=2, allow_nan=False)

    @staticmethod
    def _new_default_capture_path() -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        candidate = Path.cwd() / f"fl-capture-{stamp}.wav"
        suffix = 1
        while candidate.exists() or candidate.with_suffix(".json").exists():
            candidate = Path.cwd() / f"fl-capture-{stamp}-{suffix}.wav"
            suffix += 1
        return candidate

    def _capture_path_edited(self, *_args) -> None:
        if not self._updating_capture_path:
            self.capture_output_auto = False

    def _advance_default_capture_path(self) -> None:
        self._updating_capture_path = True
        self.capture_output.set(str(self._new_default_capture_path()))
        self._updating_capture_path = False
        self.capture_output_auto = True

    def _browse_capture_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Save captured WAV", defaultextension=".wav", filetypes=(("WAV audio", "*.wav"),))
        if path:
            self.capture_output_auto = False
            self.capture_output.set(path)

    def _browse_wav(self) -> None:
        path = filedialog.askopenfilename(title="Choose stereo WAV", filetypes=(("WAV audio", "*.wav"), ("All files", "*.*")))
        if path:
            self.analyze_input.set(path)
            self.analysis_output.set(str(Path(path).with_suffix(".analysis.json")))

    def _browse_analysis_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Save analysis report", defaultextension=".json", filetypes=(("JSON report", "*.json"),))
        if path:
            self.analysis_output.set(path)

    def _browse_before(self) -> None:
        path = filedialog.askopenfilename(title="Choose baseline analysis", filetypes=(("JSON report", "*.json"),))
        if path:
            self.before_var.set(path)

    def _browse_trial(self) -> None:
        path = filedialog.askopenfilename(title="Choose trial analysis", filetypes=(("JSON report", "*.json"),))
        if path:
            self.trial_var.set(path)
            self.comparison_output.set(str(Path(path).with_name(f"{Path(path).stem}.comparison.json")))

    def _browse_comparison_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Save comparison", defaultextension=".json", filetypes=(("JSON report", "*.json"),))
        if path:
            self.comparison_output.set(path)

    def refresh_devices(self) -> None:
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            self.devices = []
            labels = []
            for index, device in enumerate(devices):
                if device["max_input_channels"] >= 2:
                    api = hostapis[device["hostapi"]]["name"]
                    label = f"[{index}] {device['name']} — {api} — {device['max_input_channels']} inputs"
                    self.devices.append((index, label))
                    labels.append(label)
            self.device_box["values"] = labels
            if labels:
                if self.device_var.get() not in labels:
                    preferred = next((label for label in labels if "focusrite" in label.lower() or "loopback" in label.lower()), labels[0])
                    self.device_var.set(preferred)
                self.status.set(f"Found {len(labels)} stereo-capable input device(s)")
            else:
                self.device_var.set("")
                self.status.set("No stereo-capable input device found")
        except Exception as exc:
            self.status.set("Could not list audio devices")
            messagebox.showerror("Audio device error", str(exc))

    @staticmethod
    def _ensure_new(paths: list[Path]) -> None:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise ValueError("These output files already exist; choose new filenames to protect the existing files:\n" + "\n".join(existing))

    @staticmethod
    def _finite_positive(value: str, name: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number.") from exc
        if not 0 < number < float("inf"):
            raise ValueError(f"{name} must be greater than zero.")
        return number

    def _begin(self, label: str, target, box: tk.Text, cancellable: bool = False) -> None:
        if self.busy:
            return
        self.busy = True
        self.status.set(label)
        self.active_operation = label
        cancel_event = threading.Event() if cancellable else None
        self.cancel_event = cancel_event
        if cancellable:
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=100, value=0)
            stop_label = "Stop capture" if "Recording" in label else "Stop analysis"
            self.stop_button.configure(text=stop_label, state="normal")
        else:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        for button in (self.capture_button, self.analyze_button, self.compare_button, self.refresh_button,
                       self.check_button, self.analyze_capture_button):
            button.configure(state="disabled")
        self._write(box, f"{label}…\nThe interface will stay responsive while this runs.")

        def work() -> None:
            try:
                if cancellable:
                    last_progress = [-1.0]
                    def report_progress(fraction: float) -> None:
                        if not cancel_event.is_set() and (fraction >= 1.0 or fraction - last_progress[0] >= 0.01):
                            last_progress[0] = fraction
                            self.events.put(("progress", box, fraction, None))
                    value = target(cancel_event, report_progress)
                else:
                    value = target()
                self.events.put(("done", box, value, None))
            except Exception as exc:
                self.events.put(("done", box, None, exc))

        threading.Thread(target=work, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, box, value, error = self.events.get_nowait()
                if kind == "progress":
                    self.status.set(f"{self.active_operation}: {value * 100:.0f}%")
                    self.progress.configure(value=value * 100)
                    continue
                if kind == "levels":
                    self.live_levels = value
                    self._draw_live_levels()
                    continue
                self.busy = False
                self.progress.stop()
                self.progress.configure(mode="indeterminate", value=0)
                self.stop_button.configure(state="disabled")
                self.stop_button.configure(text="Stop")
                self.cancel_event = None
                for button in (self.capture_button, self.analyze_button, self.compare_button, self.refresh_button,
                               self.check_button):
                    button.configure(state="normal")
                if error:
                    self.status.set("Operation failed")
                    self._write(box, f"Could not complete operation.\n\n{error}")
                    if box is self.capture_result:
                        self.pending_capture_path = None
                else:
                    self.status.set("Operation complete")
                    if box is self.capture_result and getattr(self, "pending_capture_path", None):
                        if self.pending_capture_path.is_file():
                            self.last_capture_path = self.pending_capture_path
                            self.analyze_capture_button.configure(state="normal")
                            if self.capture_output_auto:
                                self._advance_default_capture_path()
                        self.pending_capture_path = None
                    if isinstance(value, dict):
                        self._write(box, value.get("message", ""))
                        if "timeline" in value:
                            self.timeline_data = value["timeline"]
                            self.timeline_duration = value["duration"]
                            self._draw_timeline()
                        if "bands" in value:
                            self.band_data = value["bands"]
                            self._draw_band_chart()
                    else:
                        self._write(box, value)
                latest_capture = getattr(self, "last_capture_path", None)
                self.analyze_capture_button.configure(
                    state="normal" if latest_capture is not None and latest_capture.is_file() else "disabled"
                )
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def stop_operation(self) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.stop_button.configure(state="disabled")
            self.status.set("Stopping after the current audio block…")

    def analyze_latest_capture(self) -> None:
        path = getattr(self, "last_capture_path", None)
        if path is not None and path.is_file():
            self.analyze_input.set(str(path))
            self.analysis_output.set(str(path.with_suffix(".analysis.json")))
            self.tabs.select(self.analyze_tab)
            self.start_analyze()

    def _draw_live_levels(self) -> None:
        canvas = getattr(self, "level_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        width, height = max(300, canvas.winfo_width()), max(84, canvas.winfo_height())
        left, right = 48, 86
        plot_width = width - left - right
        for level in (-60, -40, -20, 0):
            x = left + (level + 60) / 60 * plot_width
            canvas.create_line(x, 8, x, height - 18, fill="#303b4a", width=1)
            canvas.create_text(x, height - 8, text=str(level), fill=MUTED,
                               font=("Segoe UI", 8), anchor="center")
        if not self.live_levels:
            canvas.create_text(left + plot_width / 2, height / 2,
                               text="Run Check input or start a capture to see levels.",
                               fill=MUTED, font=("Segoe UI", 9))
            return

        for index, (name, color) in enumerate((("L", "#77e0b5"), ("R", "#8ab4ff"))):
            item = self.live_levels[index]
            center_y = 25 + index * 27
            y0, y1 = center_y - 6, center_y + 6
            canvas.create_text(14, center_y, text=name, fill=TEXT,
                               font=("Segoe UI Semibold", 9), anchor="w")
            canvas.create_rectangle(left, y0, left + plot_width, y1,
                                    fill="#253040", outline="")
            rms = item["rms_dbfs"]
            peak = item["peak_dbfs"]
            if rms is not None:
                rms_x = left + (max(-60.0, min(0.0, rms)) + 60) / 60 * plot_width
                canvas.create_rectangle(left, y0, rms_x, y1, fill=color, outline="")
            if peak is not None:
                peak_x = left + (max(-60.0, min(0.0, peak)) + 60) / 60 * plot_width
                peak_color = "#ff626f" if peak >= -0.01 else WARN
                canvas.create_line(peak_x, y0 - 2, peak_x, y1 + 2, fill=peak_color, width=2)
            canvas.create_text(width - 8, center_y,
                               text=f"{self._db(rms)} / {self._db(peak)} dBFS",
                               fill=TEXT, font=("Segoe UI", 8), anchor="e")

    def check_input(self) -> None:
        selected = self.device_var.get()
        device = next((index for index, label in self.devices if label == selected), None)
        if device is None:
            messagebox.showerror("Choose an input", "Choose a stereo input device first.")
            return

        def run() -> str:
            info = sd.query_devices(device, "input")
            samplerate = int(round(info["default_samplerate"]))
            chunks = []
            overflow_count = 0
            frames = samplerate * 2
            block = max(1, samplerate // 10)
            with sd.InputStream(samplerate=samplerate, channels=2, dtype="float32", device=device) as stream:
                collected = 0
                while collected < frames:
                    chunk, overflowed = stream.read(min(block, frames - collected))
                    overflow_count += int(overflowed)
                    if chunk.shape[0]:
                        chunks.append(chunk.copy())
                        collected += chunk.shape[0]
            metrics = measure_stereo(np.concatenate(chunks, axis=0))
            left, right = metrics["channels"]
            self.events.put(("levels", self.capture_result, [
                {"rms_dbfs": channel["rms_dbfs"], "peak_dbfs": channel["peak_dbfs"]}
                for channel in metrics["channels"]
            ], None))
            has_signal = not metrics["any_silence"]
            return (f"2-second input check complete\n\nDevice: {info['name']}\n"
                    f"Left RMS: {self._db(left['rms_dbfs'])} dBFS   Peak: {self._db(left['peak_dbfs'])} dBFS\n"
                    f"Right RMS: {self._db(right['rms_dbfs'])} dBFS   Peak: {self._db(right['peak_dbfs'])} dBFS\n"
                    f"Signal above the silence threshold: {'Yes' if has_signal else 'No'}\n"
                    f"Input buffer overflow events: {overflow_count}\n"
                    "This check cannot identify the app producing the samples or confirm FL-only routing.")

        self._begin("Checking selected input", run, self.capture_result)

    def _request_close(self) -> None:
        if self.busy:
            messagebox.showwarning("Operation in progress", "Wait for the current operation to finish before closing, so the result can be saved and shown.")
            return
        self.destroy()

    def start_capture(self) -> None:
        try:
            seconds = self._finite_positive(self.seconds_var.get(), "Duration")
            if seconds > 120:
                raise ValueError("Duration must be 120 seconds or less to keep capture memory use bounded.")
            selected = self.device_var.get()
            device = next((index for index, label in self.devices if label == selected), None)
            if device is None:
                raise ValueError("Choose a stereo input device. Use Refresh if the device list is stale.")
            output = Path(self.capture_output.get()).expanduser()
            if output.suffix.lower() != ".wav":
                raise ValueError("Capture output must use the .wav extension.")
            report = output.with_suffix(".json")
            self._ensure_new([output, report])
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check capture settings", str(exc))
            return

        self.pending_capture_path = output
        self.analyze_capture_button.configure(state="disabled")

        def run(cancel_event, progress_callback) -> str:
            def report_levels(levels) -> None:
                self.events.put(("levels", self.capture_result, levels, None))
            self.events.put(("levels", self.capture_result, None, None))
            result = capture(device, seconds, output, cancel_event, progress_callback,
                             overwrite=False, level_callback=report_levels)
            state = "Stopped early" if result["interrupted"] else "Capture saved"
            input_error = f"\nAudio input error: {result['input_error']}" if result.get("input_error") else ""
            return (f"{state}{input_error}\n\nWAV: {output.resolve()}\nMeasurements: {report.resolve()}\n"
                    f"Duration: {result['duration_seconds']:.2f} s at {result['sample_rate_hz']} Hz\n"
                    f"Left RMS: {self._db(result['channels'][0]['rms_dbfs'])} dBFS   Peak: {self._db(result['channels'][0]['peak_dbfs'])} dBFS\n"
                    f"Right RMS: {self._db(result['channels'][1]['rms_dbfs'])} dBFS   Peak: {self._db(result['channels'][1]['peak_dbfs'])} dBFS\n"
                    f"Left minus right RMS: {self._signed(result['left_right_rms_delta_db'])} dB\n"
                    f"Clipping detected: {'Yes' if result['any_clipping'] else 'No'}\n"
                    f"Input buffer overflows: {result['input_overflow_count']}\n"
                    f"Non-finite input samples replaced with silence: {result['non_finite_samples_replaced']}")
        self._begin("Recording stereo input", run, self.capture_result, cancellable=True)

    def start_analyze(self) -> None:
        try:
            source = Path(self.analyze_input.get()).expanduser()
            if not source.is_file():
                raise ValueError("Choose an existing WAV file.")
            window = self._finite_positive(self.window_var.get(), "Timeline window")
            if window > 30:
                raise ValueError("Timeline windows must be 30 seconds or shorter to keep memory use bounded.")
            output_text = self.analysis_output.get().strip()
            output = Path(output_text).expanduser() if output_text else source.with_suffix(".analysis.json")
            if output.suffix.lower() != ".json":
                raise ValueError("Analysis report output must use the .json extension.")
            if output.resolve() == source.resolve():
                raise ValueError("The report path cannot replace the input WAV.")
            self._ensure_new([output])
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check analysis settings", str(exc))
            return

        self.timeline_data = None
        self.timeline_duration = 0.0
        self._draw_timeline()

        def run(cancel_event, progress_callback) -> dict:
            report = analyze_wav(source, window, cancel_event, progress_callback)
            self._write_new_json(output, report)
            return {
                "message": self._format_analysis(report, output),
                "timeline": report["timeline"],
                "duration": report["duration_seconds"],
            }
        self._begin("Analyzing WAV", run, self.analyze_result, cancellable=True)

    def start_compare(self) -> None:
        try:
            before_path, trial_path = Path(self.before_var.get()).expanduser(), Path(self.trial_var.get()).expanduser()
            if not before_path.is_file() or not trial_path.is_file():
                raise ValueError("Choose existing baseline and trial report files.")
            if before_path.resolve() == trial_path.resolve():
                raise ValueError("Choose two different reports to compare.")
            output_text = self.comparison_output.get().strip()
            output = Path(output_text).expanduser() if output_text else trial_path.with_name(f"{trial_path.stem}.comparison.json")
            if output.suffix.lower() != ".json":
                raise ValueError("Comparison report output must use the .json extension.")
            if output.resolve() in (before_path.resolve(), trial_path.resolve()):
                raise ValueError("The comparison output cannot replace either input report.")
            self._ensure_new([output])
            preference = self.preference_var.get().lower()
            if preference == "no preference":
                preference = None
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check comparison settings", str(exc))
            return

        def run() -> dict:
            try:
                before = json.loads(before_path.read_text(encoding="utf-8"))
                trial = json.loads(trial_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read analysis report: {exc}") from exc
            for name, item in (("Baseline", before), ("Trial", trial)):
                if not isinstance(item, dict) or not isinstance(item.get("channels"), list) or not isinstance(item.get("frequency_bands"), list):
                    raise ValueError(f"{name} file is not an FL Audio Capture analysis report.")
                if item.get("interrupted", False):
                    raise ValueError(f"{name} analysis was stopped early. Use a completed report for a meaningful comparison.")
            result = compare_reports(before, trial, preference)
            notes = []
            before_duration, trial_duration = before.get("duration_seconds"), trial.get("duration_seconds")
            if isinstance(before_duration, (int, float)) and isinstance(trial_duration, (int, float)):
                longest = max(before_duration, trial_duration)
                if longest > 0 and abs(before_duration - trial_duration) / longest > 0.1:
                    notes.append("Clip lengths differ by more than 10%; confirm these are comparable sections.")
            if before.get("sample_rate_hz") and trial.get("sample_rate_hz") and before["sample_rate_hz"] != trial["sample_rate_hz"]:
                notes.append("Sample rates differ; frequency-band measurements may not be directly comparable.")
            result["comparison_notes"] = notes
            self._write_new_json(output, result)
            return {"message": self._format_comparison(result, output, notes),
                    "bands": result["frequency_band_changes"]}
        self._begin("Comparing reports", run, self.compare_result)

    @staticmethod
    def _db(value) -> str:
        return f"{value:.2f}" if value is not None else "−∞"

    @staticmethod
    def _signed(value) -> str:
        return f"{value:+.2f}" if value is not None else "unavailable"

    @staticmethod
    def _measurement(value) -> str:
        return f"{value:.2f}" if value is not None else "unavailable"

    def _draw_timeline(self) -> None:
        canvas = getattr(self, "timeline_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        width, height = max(280, canvas.winfo_width()), max(100, canvas.winfo_height())
        left, right, top, bottom = 48, 14, 14, 24
        plot_width, plot_height = width - left - right, height - top - bottom
        if not self.timeline_data:
            canvas.create_text(width / 2, height / 2, text="Analyze a recording to draw its level timeline.",
                               fill=MUTED, font=("Segoe UI", 9))
            return

        for db_level in (0, -20, -40, -60):
            y = top + (0 - db_level) / 60 * plot_height
            canvas.create_line(left, y, width - right, y, fill="#303b4a", width=1)
            canvas.create_text(left - 8, y, text=str(db_level), fill=MUTED,
                               font=("Segoe UI", 8), anchor="e")
        canvas.create_text(left, height - 7, text="0 s", fill=MUTED,
                           font=("Segoe UI", 8), anchor="w")
        canvas.create_text(width - right, height - 7, text=f"{self.timeline_duration:.1f} s",
                           fill=MUTED, font=("Segoe UI", 8), anchor="e")

        def points(channel: str) -> list[float]:
            rows = self.timeline_data
            stride = max(1, (len(rows) + 1499) // 1500)
            indexes = list(range(0, len(rows), stride))
            if indexes[-1] != len(rows) - 1:
                indexes.append(len(rows) - 1)
            coords = []
            for index in indexes:
                row = rows[index]
                time = row.get("start_seconds", 0.0)
                level = row.get(f"{channel}_rms_dbfs")
                if level is None:
                    continue
                level = max(-60.0, min(0.0, float(level)))
                x = left + (time / self.timeline_duration * plot_width if self.timeline_duration else 0)
                y = top + (0 - level) / 60 * plot_height
                coords.extend((x, y))
            return coords

        for channel, color in (("left", "#77e0b5"), ("right", "#8ab4ff")):
            coords = points(channel)
            if len(coords) >= 4:
                canvas.create_line(*coords, fill=color, width=2, smooth=False)
            elif len(coords) == 2:
                x, y = coords
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)

    def _draw_band_chart(self) -> None:
        canvas = getattr(self, "band_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        width, height = max(300, canvas.winfo_width()), max(120, canvas.winfo_height())
        left, right, top, bottom = 50, 20, 16, 38
        plot_width, plot_height = width - left - right, height - top - bottom
        if not self.band_data:
            canvas.create_text(width / 2, height / 2, text="Compare two reports to see frequency-band shifts.",
                               fill=MUTED, font=("Segoe UI", 9))
            return

        changes = [item["change_db"] for item in self.band_data if item["change_db"] is not None]
        max_abs = max(2.0, max((abs(value) for value in changes), default=2.0))
        scale = max(2.0, (int(max_abs / 2) + 1) * 2)
        baseline = top + plot_height / 2
        for tick in (-scale, 0, scale):
            y = top + (scale - tick) / (2 * scale) * plot_height
            canvas.create_line(left, y, width - right, y,
                               fill="#536173" if tick == 0 else "#303b4a",
                               width=2 if tick == 0 else 1)
            canvas.create_text(left - 8, y, text=f"{tick:+g}", fill=MUTED,
                               font=("Segoe UI", 8), anchor="e")
        labels = {
            "sub_bass_20_60_hz": "Sub\n20–60",
            "bass_60_250_hz": "Bass\n60–250",
            "low_mid_250_500_hz": "Low mids\n250–500",
            "mid_500_2000_hz": "Mids\n500–2k",
            "high_mid_2000_6000_hz": "High mids\n2–6k",
            "high_6000_20000_hz": "High\n6–20k",
        }
        slot = plot_width / max(1, len(self.band_data))
        bar_width = min(54, slot * 0.48)
        for index, item in enumerate(self.band_data):
            center_x = left + (index + 0.5) * slot
            change = item["change_db"]
            if change is not None:
                bounded = max(-scale, min(scale, float(change)))
                y_value = top + (scale - bounded) / (2 * scale) * plot_height
                color = "#77e0b5" if change >= 0 else "#ffb86b"
                canvas.create_rectangle(center_x - bar_width / 2, min(baseline, y_value),
                                        center_x + bar_width / 2, max(baseline, y_value),
                                        fill=color, outline="")
                label_y = y_value - 11 if change >= 0 else y_value + 11
                canvas.create_text(center_x, label_y, text=f"{change:+.1f} dB",
                                   fill=TEXT, font=("Segoe UI Semibold", 8))
            band_name = item["band"]
            canvas.create_text(center_x, height - 22, text=labels.get(band_name, band_name),
                               fill=MUTED, font=("Segoe UI", 8), justify="center")

    def _format_analysis(self, report: dict, output: Path) -> str:
        title = "ANALYSIS STOPPED EARLY" if report.get("interrupted") else "ANALYSIS COMPLETE"
        duration_label = f"Duration: {report['duration_seconds']:.2f} s"
        if report.get("interrupted"):
            duration_label += f" of {report['requested_duration_seconds']:.2f} s requested"
        lines = [title, "", f"Source: {report['wav_file']}",
                 f"Report: {output.resolve()}", f"{duration_label}   Sample rate: {report['sample_rate_hz']} Hz",
                 f"Window: {report['window_seconds']:.3f} s   Timeline windows: {len(report['timeline'])}", "",
                 "CHANNEL LEVELS"]
        requested_window = report.get("requested_window_seconds")
        if requested_window is not None and report["window_seconds"] > requested_window:
            lines[5:5] = [
                f"Timeline window widened from {requested_window:.3f} s to keep the report within 10,000 windows.",
                "",
            ]
        for channel in report["channels"]:
            lines.append(f"{channel['channel'].title():5}  RMS {self._db(channel['rms_dbfs'])} dBFS   peak {self._db(channel['peak_dbfs'])} dBFS   silence {channel['silence_duration_seconds']:.2f} s   clipped samples {channel['clipped_samples']}")
            lines.append(f"      Crest {self._measurement(channel['crest_factor_db'])} dB   active-window range {self._measurement(channel['active_window_level_range_db'])} dB")
        image = report["stereo_image"]
        lines.extend(["", "STEREO", f"Left minus right RMS: {self._signed(report['left_right_rms_delta_db'])} dB",
                      f"Correlation: {self._measurement(report['stereo_correlation'])}",
                      f"Side / mid: {self._measurement(image['side_to_mid_db'])} dB   Mono fold-down: {self._db(image['mono_fold_down_rms_dbfs'])} dBFS",
                      f"Clipping detected: {'Yes' if report['any_clipping'] else 'No'}   Silence detected: {'Yes' if report['any_silence'] else 'No'}", "", "FREQUENCY BANDS"])
        for band in report["frequency_bands"]:
            lines.append(f"{band['band']}: L {self._db(band['left_rms_dbfs'])} dBFS ({band['left_energy_percent']:.1f}%)  |  R {self._db(band['right_rms_dbfs'])} dBFS ({band['right_energy_percent']:.1f}%)")
        return "\n".join(lines)

    def _format_comparison(self, result: dict, output: Path, notes: list[str] | None = None) -> str:
        lines = ["COMPARISON COMPLETE", "", f"Baseline: {result['before_file']}", f"Trial: {result['trial_file']}",
                 f"Saved report: {output.resolve()}", "", result["recommendation"], "",
                 f"Average RMS change: {self._signed(result['average_rms_change_db'])} dB",
                 f"Peak change: {self._signed(result['peak_change_db'])} dB"]
        if notes:
            lines.extend(["", "COMPARISON NOTES", *notes])
        lines.extend(["", "FREQUENCY BAND CHANGES"])
        for band in result["frequency_band_changes"]:
            lines.append(f"{band['band']}: {self._signed(band['change_db'])} dB")
        lines.append(f"\nClipping: baseline {'yes' if result['before_clipping'] else 'no'}; trial {'yes' if result['trial_clipping'] else 'no'}")
        return "\n".join(lines)


if __name__ == "__main__":
    CaptureApp().mainloop()
