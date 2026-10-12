"""Local stereo capture and level meter for a routed FL Studio signal."""

from __future__ import annotations

import argparse
import json
import math
import sys
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd


SILENCE_DBFS = -60.0
CLIP_THRESHOLD = 0.999
FREQUENCY_BANDS = (
    ("sub_bass_20_60_hz", 20.0, 60.0),
    ("bass_60_250_hz", 60.0, 250.0),
    ("low_mid_250_500_hz", 250.0, 500.0),
    ("mid_500_2000_hz", 500.0, 2000.0),
    ("high_mid_2000_6000_hz", 2000.0, 6000.0),
    ("high_6000_20000_hz", 6000.0, 20000.0),
)
MAX_TIMELINE_WINDOWS = 10_000


def dbfs(amplitude: float) -> float | None:
    # JSON has no standard representation for -infinity; null means digital zero.
    return 20.0 * math.log10(amplitude) if amplitude > 0 else None


def frequency_band_energy(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel mean-square energy for bands and all analyzed frequencies."""
    frame_count = audio.shape[0]
    window = np.hanning(frame_count) if frame_count > 1 else np.ones(frame_count)
    normalization = float(np.sum(np.square(window)))
    spectrum = np.fft.rfft(audio * window[:, None], axis=0)
    power = np.square(np.abs(spectrum))
    weights = np.full(power.shape[0], 2.0)
    weights[0] = 1.0
    if frame_count % 2 == 0:
        weights[-1] = 1.0
    power *= weights[:, None]
    # NumPy's FFT is unscaled, so Parseval's sum must also be divided by N.
    power /= normalization * frame_count
    frequencies = np.fft.rfftfreq(frame_count, 1.0 / sample_rate)
    upper_limit = min(20000.0, sample_rate / 2.0)
    band_powers = []
    for _, low, high in FREQUENCY_BANDS:
        mask = (frequencies >= low) & (frequencies < min(high, upper_limit))
        band_powers.append(np.sum(power[mask], axis=0) if np.any(mask)
                           else np.zeros(audio.shape[1], dtype=np.float64))
    return np.stack(band_powers), np.sum(np.stack(band_powers), axis=0)


def stereo_image_metrics(audio: np.ndarray) -> dict:
    left, right = audio[:, 0], audio[:, 1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    mid_rms = float(np.sqrt(np.mean(np.square(mid))))
    side_rms = float(np.sqrt(np.mean(np.square(side))))
    mid_dbfs, side_dbfs = dbfs(mid_rms), dbfs(side_rms)
    return {
        "mid_rms_dbfs": mid_dbfs,
        "side_rms_dbfs": side_dbfs,
        "side_to_mid_db": (side_dbfs - mid_dbfs
                           if side_dbfs is not None and mid_dbfs is not None else None),
        "mono_fold_down_rms_dbfs": mid_dbfs,
    }


def _band_report(band_power: np.ndarray, total_power: np.ndarray) -> list[dict]:
    result = []
    for index, (name, _, _) in enumerate(FREQUENCY_BANDS):
        powers = band_power[index]
        result.append({
            "band": name,
            "left_rms_dbfs": dbfs(float(np.sqrt(powers[0]))),
            "right_rms_dbfs": dbfs(float(np.sqrt(powers[1]))),
            "left_energy_percent": (float(100 * powers[0] / total_power[0])
                                     if total_power[0] > 0 else 0.0),
            "right_energy_percent": (float(100 * powers[1] / total_power[1])
                                     if total_power[1] > 0 else 0.0),
        })
    return result


def measure_stereo(audio: np.ndarray) -> dict:
    """Return channel metrics for a normalized float32 array shaped (frames, 2)."""
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] == 0:
        raise ValueError("audio must contain at least one stereo frame")

    rms = np.sqrt(np.mean(np.square(samples), axis=0))
    peak = np.max(np.abs(samples), axis=0)
    diff = samples[:, 0] - samples[:, 1]
    diff_rms = float(np.sqrt(np.mean(np.square(diff))))
    left_dbfs, right_dbfs = dbfs(float(rms[0])), dbfs(float(rms[1]))
    level_delta = (left_dbfs - right_dbfs
                   if left_dbfs is not None and right_dbfs is not None else None)

    channels = []
    for index, label in enumerate(("left", "right")):
        channels.append({
            "channel": label,
            "rms": float(rms[index]),
            "rms_dbfs": dbfs(float(rms[index])),
            "peak": float(peak[index]),
            "peak_dbfs": dbfs(float(peak[index])),
            "silent": dbfs(float(rms[index])) is None or dbfs(float(rms[index])) < SILENCE_DBFS,
            "clipping": bool(peak[index] >= CLIP_THRESHOLD),
            "clipped_samples": int(np.count_nonzero(np.abs(samples[:, index]) >= CLIP_THRESHOLD)),
        })

    return {
        "frames": int(samples.shape[0]),
        "left_right_rms_delta_db": level_delta,
        "waveform_difference_rms": diff_rms,
        "waveform_difference_rms_dbfs": dbfs(diff_rms),
        "channels": channels,
        "silence_threshold_dbfs": SILENCE_DBFS,
        "clipping_threshold": CLIP_THRESHOLD,
        "any_silence": any(channel["silent"] for channel in channels),
        "any_clipping": any(channel["clipping"] for channel in channels),
    }


def list_inputs() -> None:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    print("Input devices (look for Focusrite Loopback or loopback channels):")
    for index, device in enumerate(devices):
        if device["max_input_channels"]:
            api = hostapis[device["hostapi"]]["name"]
            print(f"[{index}] {device['name']} | {api} | inputs={device['max_input_channels']} | "
                  f"default rate={device['default_samplerate']:.0f} Hz")


def _decode_pcm(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    """Decode uncompressed integer PCM from a WAV chunk into normalized float64."""
    if sample_width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        values = (values - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8)
        if data.size % 3:
            raise ValueError("WAV data ends with an incomplete 24-bit sample")
        triplets = data.reshape(-1, 3).astype(np.int32)
        integers = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        values = integers.astype(np.float64) / 8388608.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width * 8} bits")
    if values.size % channels:
        raise ValueError("WAV data ends with an incomplete audio frame")
    return values.reshape(-1, channels)


def analyze_wav(path: Path, window_seconds: float = 0.25, cancel_event=None,
                progress_callback=None) -> dict:
    """Measure a stereo PCM WAV overall and in short timeline windows."""
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window-seconds must be a finite number greater than zero")
    try:
        wav = wave.open(str(path), "rb")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Could not read WAV file: {exc}") from exc

    with wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        sample_width = wav.getsampwidth()
        if wav.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM WAV files are supported")
        if channels != 2:
            raise ValueError(f"Expected stereo WAV (2 channels), found {channels}")
        if sample_rate <= 0 or frame_count <= 0:
            raise ValueError("WAV file contains no audio frames")

        requested_frames_per_window = max(1, int(round(window_seconds * sample_rate)))
        # A very long source or very small requested window must not create an
        # unbounded in-memory timeline/report. Keep at most 10,000 entries.
        frames_per_window = max(
            requested_frames_per_window,
            math.ceil(frame_count / MAX_TIMELINE_WINDOWS),
        )
        sum_squares = np.zeros(2, dtype=np.float64)
        peaks = np.zeros(2, dtype=np.float64)
        clip_counts = np.zeros(2, dtype=np.int64)
        silence_frames = np.zeros(2, dtype=np.int64)
        band_energy = np.zeros((len(FREQUENCY_BANDS), 2), dtype=np.float64)
        total_band_energy = np.zeros(2, dtype=np.float64)
        sum_left = sum_right = sum_left_sq = sum_right_sq = sum_lr = 0.0
        timeline = []
        processed = 0

        while processed < frame_count:
            if cancel_event is not None and cancel_event.is_set():
                break
            count = min(frames_per_window, frame_count - processed)
            audio = _decode_pcm(wav.readframes(count), sample_width, channels)
            if audio.shape[0] != count:
                raise ValueError("WAV file ended before its declared frame count")
            n = audio.shape[0]
            rms = np.sqrt(np.mean(np.square(audio), axis=0))
            peak = np.max(np.abs(audio), axis=0)
            db_values = [dbfs(float(value)) for value in rms]
            peak_db_values = [dbfs(float(value)) for value in peak]
            silent = [value is None or value < SILENCE_DBFS for value in db_values]
            clipped = [bool(value >= CLIP_THRESHOLD) for value in peak]
            window_band_power, window_total_band_power = frequency_band_energy(audio, sample_rate)
            band_energy += window_band_power * n
            total_band_energy += window_total_band_power * n
            image = stereo_image_metrics(audio)

            sum_squares += np.sum(np.square(audio), axis=0)
            peaks = np.maximum(peaks, peak)
            clip_counts += np.count_nonzero(np.abs(audio) >= CLIP_THRESHOLD, axis=0)
            for channel in range(2):
                if silent[channel]:
                    silence_frames[channel] += n
            left, right = audio[:, 0], audio[:, 1]
            sum_left += float(np.sum(left))
            sum_right += float(np.sum(right))
            sum_left_sq += float(np.sum(np.square(left)))
            sum_right_sq += float(np.sum(np.square(right)))
            sum_lr += float(np.sum(left * right))

            window_delta = (db_values[0] - db_values[1]
                            if db_values[0] is not None and db_values[1] is not None else None)
            start = processed / sample_rate
            timeline.append({
                "start_seconds": start,
                "end_seconds": (processed + n) / sample_rate,
                "left_rms_dbfs": db_values[0],
                "right_rms_dbfs": db_values[1],
                "left_peak_dbfs": peak_db_values[0],
                "right_peak_dbfs": peak_db_values[1],
                "left_right_rms_delta_db": window_delta,
                "left_silent": silent[0],
                "right_silent": silent[1],
                "left_clipping": clipped[0],
                "right_clipping": clipped[1],
                "stereo_image": image,
                "frequency_bands": _band_report(window_band_power, window_total_band_power),
            })
            processed += n
            if progress_callback is not None:
                progress_callback(min(1.0, processed / frame_count))

    if processed == 0:
        raise ValueError("Analysis stopped before any audio frames were processed.")

    total_rms = np.sqrt(sum_squares / processed)
    channel_reports = []
    active_levels = [[], []]
    for i, name in enumerate(("left", "right")):
        level = dbfs(float(total_rms[i]))
        for item in timeline:
            value = item[f"{name}_rms_dbfs"]
            if value is not None and value >= SILENCE_DBFS:
                active_levels[i].append(value)
        channel_reports.append({
            "channel": name,
            "rms": float(total_rms[i]),
            "rms_dbfs": level,
            "peak": float(peaks[i]),
            "peak_dbfs": dbfs(float(peaks[i])),
            "silent": level is None or level < SILENCE_DBFS,
            "silence_duration_seconds": float(silence_frames[i] / sample_rate),
            "clipping": bool(clip_counts[i] > 0),
            "clipped_samples": int(clip_counts[i]),
            "active_window_level_range_db": (
                float(max(active_levels[i]) - min(active_levels[i])) if active_levels[i] else None
            ),
            "crest_factor_db": (dbfs(float(peaks[i])) - level
                                if level is not None and peaks[i] > 0 else None),
        })

    covariance = sum_lr - (sum_left * sum_right / processed)
    variance_left = sum_left_sq - (sum_left * sum_left / processed)
    variance_right = sum_right_sq - (sum_right * sum_right / processed)
    denominator = math.sqrt(max(0.0, variance_left) * max(0.0, variance_right))
    correlation = max(-1.0, min(1.0, covariance / denominator)) if denominator > 0 else None
    left_dbfs, right_dbfs = dbfs(float(total_rms[0])), dbfs(float(total_rms[1]))
    mid_power = max(0.0, (sum_left_sq + sum_right_sq + 2.0 * sum_lr) / (4.0 * processed))
    side_power = max(0.0, (sum_left_sq + sum_right_sq - 2.0 * sum_lr) / (4.0 * processed))
    mid_dbfs, side_dbfs = dbfs(math.sqrt(mid_power)), dbfs(math.sqrt(side_power))
    stereo_image = {
        "mid_rms_dbfs": mid_dbfs,
        "side_rms_dbfs": side_dbfs,
        "side_to_mid_db": (side_dbfs - mid_dbfs
                           if side_dbfs is not None and mid_dbfs is not None else None),
        "mono_fold_down_rms_dbfs": mid_dbfs,
    }

    return {
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wav_file": str(path.resolve()),
        "sample_rate_hz": sample_rate,
        "duration_seconds": processed / sample_rate,
        "requested_duration_seconds": frame_count / sample_rate,
        "frames": processed,
        "interrupted": processed < frame_count,
        "requested_window_seconds": requested_frames_per_window / sample_rate,
        "window_seconds": frames_per_window / sample_rate,
        "silence_threshold_dbfs": SILENCE_DBFS,
        "clipping_threshold": CLIP_THRESHOLD,
        "channels": channel_reports,
        "left_right_rms_delta_db": (
            left_dbfs - right_dbfs if left_dbfs is not None and right_dbfs is not None else None
        ),
        "stereo_correlation": correlation,
        "stereo_image": stereo_image,
        "frequency_bands": _band_report(band_energy / processed,
                                         total_band_energy / processed),
        "any_silence": any(channel["silent"] for channel in channel_reports),
        "any_clipping": any(channel["clipping"] for channel in channel_reports),
        "timeline": timeline,
    }


def analyze_command(input_path: Path, window_seconds: float, output: Path | None) -> dict:
    report_path = output or input_path.with_suffix(".analysis.json")
    if report_path.resolve() == input_path.resolve():
        raise ValueError("The analysis report cannot replace the input WAV")
    if report_path.suffix.lower() != ".json":
        raise ValueError("The analysis report output must use the .json extension")
    report = analyze_wav(input_path, window_seconds)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for channel in report["channels"]:
        rms = "-inf" if channel["rms_dbfs"] is None else f"{channel['rms_dbfs']:.2f}"
        peak = "-inf" if channel["peak_dbfs"] is None else f"{channel['peak_dbfs']:.2f}"
        print(f"{channel['channel'].title()}: RMS {rms} dBFS, peak {peak} dBFS, "
              f"silent {channel['silence_duration_seconds']:.2f}s, "
              f"clipped samples {channel['clipped_samples']}")
    print(f"Stereo correlation: {report['stereo_correlation']}")
    print(f"Timeline windows: {len(report['timeline'])} at {report['window_seconds']:.3f}s each")
    print(f"Saved analysis: {report_path.resolve()}")
    return report


def _mean_channel_db(report: dict, key: str) -> float | None:
    """Return the RMS across channels by averaging power, not logarithmic dB values."""
    values = [channel[key] for channel in report.get("channels", [])
              if channel.get(key) is not None]
    if not values:
        return None
    mean_power = sum(10.0 ** (value / 10.0) for value in values) / len(values)
    return 10.0 * math.log10(mean_power) if mean_power > 0 else None


def _mean_level_db(values: list[float | None]) -> float | None:
    """Combine available dBFS RMS values into one mean-power dBFS value."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    mean_power = sum(10.0 ** (value / 10.0) for value in present) / len(present)
    return 10.0 * math.log10(mean_power) if mean_power > 0 else None


def compare_reports(before: dict, trial: dict, preference: str | None = None) -> dict:
    """Compare two analyses and turn objective changes plus a listening preference into advice."""
    before_bands = {item["band"]: item for item in before.get("frequency_bands", [])}
    trial_bands = {item["band"]: item for item in trial.get("frequency_bands", [])}
    bands = []
    for name, _, _ in FREQUENCY_BANDS:
        a, b = before_bands.get(name), trial_bands.get(name)
        if not a or not b:
            continue
        # Band report entries use left/right values rather than a channels list.
        before_db = _mean_level_db([a.get("left_rms_dbfs"), a.get("right_rms_dbfs")])
        trial_db = _mean_level_db([b.get("left_rms_dbfs"), b.get("right_rms_dbfs")])
        bands.append({
            "band": name,
            "before_dbfs": before_db,
            "trial_dbfs": trial_db,
            "change_db": trial_db - before_db if before_db is not None and trial_db is not None else None,
        })

    before_peak = max((c.get("peak_dbfs") for c in before.get("channels", [])
                       if c.get("peak_dbfs") is not None), default=None)
    trial_peak = max((c.get("peak_dbfs") for c in trial.get("channels", [])
                      if c.get("peak_dbfs") is not None), default=None)
    before_rms = _mean_channel_db(before, "rms_dbfs")
    trial_rms = _mean_channel_db(trial, "rms_dbfs")
    band_by_name = {band["band"]: band for band in bands}
    sub_change = band_by_name.get("sub_bass_20_60_hz", {}).get("change_db")
    rms_change = trial_rms - before_rms if before_rms is not None and trial_rms is not None else None
    peak_change = trial_peak - before_peak if before_peak is not None and trial_peak is not None else None

    if trial.get("any_clipping") and not before.get("any_clipping"):
        advice = "The trial introduced clipping. Reduce the level before keeping this change."
    elif preference == "tighter" and sub_change is not None and sub_change < -0.5:
        advice = (f"Keep the trial setting: you preferred the tighter feel, and the 20–60 Hz band "
                  f"fell by {abs(sub_change):.1f} dB without clipping. A/B it once more at matched loudness.")
    elif preference == "fuller" and sub_change is not None and sub_change > 0.5:
        advice = (f"Keep the trial setting: you preferred the fuller feel, and the 20–60 Hz band "
                  f"rose by {sub_change:.1f} dB without clipping. A/B it once more at matched loudness.")
    elif sub_change is not None and abs(sub_change) >= 0.5:
        direction = "lowered" if sub_change < 0 else "raised"
        advice = (f"The trial {direction} 20–60 Hz energy by {abs(sub_change):.1f} dB. "
                  "Keep it only if that matches the sound you want; compare at matched loudness.")
    else:
        advice = "The trial made little change below 60 Hz. Use your listening preference to choose."

    return {
        "before_file": before.get("wav_file"),
        "trial_file": trial.get("wav_file"),
        "preference": preference,
        "average_rms_change_db": rms_change,
        "peak_change_db": peak_change,
        "frequency_band_changes": bands,
        "before_clipping": bool(before.get("any_clipping")),
        "trial_clipping": bool(trial.get("any_clipping")),
        "recommendation": advice,
        "basis": "Local level and frequency measurements plus the stated listening preference.",
    }


def compare_command(before_path: Path, trial_path: Path, preference: str | None,
                    output: Path | None) -> dict:
    if before_path.resolve() == trial_path.resolve():
        raise ValueError("Choose two different analysis reports")
    try:
        before = json.loads(before_path.read_text(encoding="utf-8"))
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read analysis report: {exc}") from exc
    for label, report in (("Baseline", before), ("Trial", trial)):
        if (not isinstance(report, dict) or not isinstance(report.get("channels"), list)
                or not isinstance(report.get("frequency_bands"), list)):
            raise ValueError(f"{label} file is not an FL Audio Capture analysis report")
        if report.get("interrupted", False):
            raise ValueError(f"{label} analysis was stopped early; use a completed report")
    comparison = compare_reports(before, trial, preference)
    report_path = output or trial_path.with_name(f"{trial_path.stem}.comparison.json")
    if report_path.resolve() in (before_path.resolve(), trial_path.resolve()):
        raise ValueError("The comparison report cannot replace either input report")
    if report_path.suffix.lower() != ".json":
        raise ValueError("The comparison report output must use the .json extension")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    print(comparison["recommendation"])
    print(f"Average RMS change: {comparison['average_rms_change_db']:+.2f} dB" if comparison["average_rms_change_db"] is not None else "Average RMS change: unavailable")
    print(f"Peak change: {comparison['peak_change_db']:+.2f} dB" if comparison["peak_change_db"] is not None else "Peak change: unavailable")
    for band in comparison["frequency_band_changes"]:
        if band["change_db"] is not None:
            print(f"{band['band']}: {band['change_db']:+.2f} dB")
    print(f"Saved comparison: {report_path.resolve()}")
    return comparison


def capture(device: int, seconds: float, output: Path, cancel_event=None,
            progress_callback=None, overwrite: bool = True, level_callback=None) -> dict:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Capture duration must be a finite number greater than zero")
    if output.suffix.lower() != ".wav":
        raise ValueError("Capture output must use the .wav extension")
    info = sd.query_devices(device, "input")
    if info["max_input_channels"] < 2:
        raise ValueError(f"Selected device has fewer than two input channels: {info['name']}")
    samplerate = int(round(info["default_samplerate"]))
    frames = max(1, int(round(seconds * samplerate)))
    # Stream chunks to disk and accumulate metrics so long takes stay memory-bounded.
    captured_frames = 0
    overflow_count = 0
    read_error = None
    block_frames = max(1, samplerate // 10)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".json")
    if not overwrite and (output.exists() or report_path.exists()):
        raise FileExistsError(f"Capture output or measurement report already exists: {output} / {report_path}")
    sum_squares = np.zeros(2, dtype=np.float64)
    peaks = np.zeros(2, dtype=np.float64)
    clip_counts = np.zeros(2, dtype=np.int64)
    difference_sum_squares = 0.0
    non_finite_samples = 0
    open_mode = "wb" if overwrite else "xb"
    with sd.InputStream(samplerate=samplerate, channels=2, dtype="float32", device=device) as stream:
        with output.open(open_mode) as audio_file:
            with wave.open(audio_file, "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(samplerate)
                while captured_frames < frames:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    count = min(block_frames, frames - captured_frames)
                    try:
                        chunk, overflowed = stream.read(count)
                    except Exception as exc:
                        read_error = exc
                        break
                    if overflowed:
                        overflow_count += 1
                    if chunk.shape[0] == 0:
                        continue
                    chunk = np.asarray(chunk, dtype=np.float32)
                    non_finite_samples += int(np.size(chunk) - np.count_nonzero(np.isfinite(chunk)))
                    audio = np.nan_to_num(chunk.astype(np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
                    sum_squares += np.sum(np.square(audio), axis=0)
                    peaks = np.maximum(peaks, np.max(np.abs(audio), axis=0))
                    clip_counts += np.count_nonzero(np.abs(audio) >= CLIP_THRESHOLD, axis=0)
                    difference = audio[:, 0] - audio[:, 1]
                    difference_sum_squares += float(np.sum(np.square(difference)))
                    pcm16 = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
                    wav.writeframesraw(pcm16.tobytes())
                    captured_frames += chunk.shape[0]
                    if level_callback is not None:
                        chunk_rms = np.sqrt(np.mean(np.square(audio), axis=0))
                        chunk_peak = np.max(np.abs(audio), axis=0)
                        level_callback([
                            {"rms_dbfs": dbfs(float(chunk_rms[index])),
                             "peak_dbfs": dbfs(float(chunk_peak[index]))}
                            for index in range(2)
                        ])
                    if progress_callback is not None:
                        progress_callback(min(1.0, captured_frames / frames))
    if captured_frames == 0:
        output.unlink(missing_ok=True)
        if read_error is not None:
            raise ValueError(f"Audio input failed before any frames were captured: {read_error}") from read_error
        raise ValueError("Capture stopped before any audio frames were received.")
    rms = np.sqrt(sum_squares / captured_frames)
    channel_reports = []
    for index, label in enumerate(("left", "right")):
        level, peak = dbfs(float(rms[index])), dbfs(float(peaks[index]))
        channel_reports.append({
            "channel": label,
            "rms": float(rms[index]),
            "rms_dbfs": level,
            "peak": float(peaks[index]),
            "peak_dbfs": peak,
            "silent": level is None or level < SILENCE_DBFS,
            "clipping": bool(peaks[index] >= CLIP_THRESHOLD),
            "clipped_samples": int(clip_counts[index]),
        })
    delta = (dbfs(float(rms[0])) - dbfs(float(rms[1]))
             if dbfs(float(rms[0])) is not None and dbfs(float(rms[1])) is not None else None)
    difference_rms = math.sqrt(difference_sum_squares / captured_frames)
    metrics = {
        "frames": captured_frames,
        "left_right_rms_delta_db": delta,
        "waveform_difference_rms": difference_rms,
        "waveform_difference_rms_dbfs": dbfs(difference_rms),
        "channels": channel_reports,
        "silence_threshold_dbfs": SILENCE_DBFS,
        "clipping_threshold": CLIP_THRESHOLD,
        "any_silence": any(channel["silent"] for channel in channel_reports),
        "any_clipping": any(channel["clipping"] for channel in channel_reports),
    }

    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_index": device,
        "device_name": info["name"],
        "host_api": sd.query_hostapis()[info["hostapi"]]["name"],
        "sample_rate_hz": samplerate,
        "duration_seconds": captured_frames / samplerate,
        "requested_duration_seconds": seconds,
        "interrupted": captured_frames < frames or read_error is not None,
        "input_error": str(read_error) if read_error is not None else None,
        "input_overflowed": overflow_count > 0,
        "input_overflow_count": overflow_count,
        "non_finite_samples_replaced": non_finite_samples,
        "wav_file": str(output.resolve()),
        **metrics,
    }
    report_mode = "w" if overwrite else "x"
    with report_path.open(report_mode, encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved stereo WAV: {output.resolve()}")
    print(f"Saved measurements: {report_path.resolve()}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list recording input devices")
    record = subparsers.add_parser("capture", help="capture and measure a stereo input")
    record.add_argument("--device", type=int, required=True, help="input device index from 'list'")
    record.add_argument("--seconds", type=float, default=10.0)
    record.add_argument("--output", type=Path, default=Path("fl-studio-test.wav"))
    analyze = subparsers.add_parser("analyze", help="analyze a saved stereo PCM WAV locally")
    analyze.add_argument("input", type=Path, help="stereo WAV file created by capture")
    analyze.add_argument("--window-seconds", type=float, default=0.25,
                         help="timeline window size (default: 0.25 seconds)")
    analyze.add_argument("--output", type=Path, help="JSON report path (default: <input>.analysis.json)")
    compare = subparsers.add_parser("compare", help="compare two analysis reports and get a local recommendation")
    compare.add_argument("before", type=Path, help="baseline .analysis.json report")
    compare.add_argument("trial", type=Path, help="changed-mix .analysis.json report")
    compare.add_argument("--preference", choices=("tighter", "fuller"),
                         help="listening preference for the trial")
    compare.add_argument("--output", type=Path, help="comparison report path")
    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            list_inputs()
        elif args.command == "capture":
            if not math.isfinite(args.seconds) or args.seconds <= 0:
                raise ValueError("seconds must be a finite number greater than zero")
            capture(args.device, args.seconds, args.output)
        else:
            if args.command == "analyze":
                analyze_command(args.input, args.window_seconds, args.output)
            else:
                compare_command(args.before, args.trial, args.preference, args.output)
    except (sd.PortAudioError, wave.Error, ValueError, OSError) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
