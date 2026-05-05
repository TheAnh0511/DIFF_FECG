from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample


def butter_bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 3,
) -> np.ndarray:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def notch_filter(
    signal: np.ndarray,
    freq: float,
    fs: int,
    q: float = 50.0,
) -> np.ndarray:
    b, a = iirnotch(w0=freq, Q=q, fs=fs)
    return filtfilt(b, a, signal, axis=-1)


def zscore_per_channel(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    return (signal - mean) / (std + eps)


def resample_signal(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return signal
    new_len = int(round(signal.shape[-1] * target_fs / orig_fs))
    return resample(signal, new_len, axis=-1)


def preprocess_ecg(
    signal: np.ndarray,
    orig_fs: int,
    target_fs: int,
    bandpass_low: float = 7.5,
    bandpass_high: float = 75.0,
    bandpass_order: int = 3,
    notch_freqs: List[float] | None = None,
    notch_q: float = 50.0,
    normalize: bool = True,
) -> np.ndarray:
    if signal.ndim == 1:
        signal = signal[None, :]

    out = butter_bandpass_filter(
        signal,
        lowcut=bandpass_low,
        highcut=bandpass_high,
        fs=orig_fs,
        order=bandpass_order,
    )

    if notch_freqs is not None:
        for f in notch_freqs:
            out = notch_filter(out, freq=f, fs=orig_fs, q=notch_q)

    if normalize:
        out = zscore_per_channel(out)

    out = resample_signal(out, orig_fs=orig_fs, target_fs=target_fs)
    return out.astype(np.float32)


def segment_signal(
    signal: np.ndarray,
    fs: int,
    seg_sec: float = 5.0,
    overlap: float = 0.5,
) -> np.ndarray:
    if signal.ndim == 1:
        signal = signal[None, :]

    seg_len = int(round(seg_sec * fs))
    hop = int(round(seg_len * (1.0 - overlap)))
    hop = max(hop, 1)

    total_len = signal.shape[-1]
    if total_len < seg_len:
        return np.empty((0, signal.shape[0], seg_len), dtype=np.float32)

    segments = []
    for start in range(0, total_len - seg_len + 1, hop):
        end = start + seg_len
        segments.append(signal[:, start:end])

    return np.stack(segments, axis=0).astype(np.float32)


def segment_pair(
    aecg: np.ndarray,
    fecg: np.ndarray,
    fs: int,
    seg_sec: float = 5.0,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if fecg.ndim == 1:
        fecg = fecg[None, :]

    aecg_segs = segment_signal(aecg, fs=fs, seg_sec=seg_sec, overlap=overlap)
    fecg_segs = segment_signal(fecg, fs=fs, seg_sec=seg_sec, overlap=overlap)
    return aecg_segs, fecg_segs
