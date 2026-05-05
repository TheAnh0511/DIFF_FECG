from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks, spectrogram
# from sklearn.preprocessing import scale
from scipy import stats


# =========================
# Waveform metrics - DIFF-like
# =========================
def _safe_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    # xử lý NaN/Inf nếu có
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(x)
    std = np.std(x)

    if std < eps:
        return x - mean

    return (x - mean) / (std + eps)


def mae_np(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def rmse_np(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def prd_np(original_signal: np.ndarray, reconstructed_signal: np.ndarray, eps: float = 1e-8) -> float:
    """
    DIFF-style PRD:
    sqrt(sum((original - reconstructed)^2) / sum(original^2)) * 100
    """
    original_signal = np.asarray(original_signal)
    reconstructed_signal = np.asarray(reconstructed_signal)

    numerator = np.sum((original_signal - reconstructed_signal) ** 2)
    denominator = np.sum(original_signal ** 2) + eps

    return float(np.sqrt(numerator / denominator) * 100.0)


def pcc_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0

    pcc, _ = stats.pearsonr(x, y)
    return float(abs(pcc))


def spc_np(x: np.ndarray, y: np.ndarray, fs: int = 200, nperseg: int = 256) -> float:
    """
    DIFF code uses spectrogram correlation.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    nperseg = min(nperseg, len(x), len(y))
    if nperseg < 8:
        return 0.0

    _, _, sxx1 = spectrogram(x, fs=fs, nperseg=nperseg)
    _, _, sxx2 = spectrogram(y, fs=fs, nperseg=nperseg)

    a = sxx1.ravel()
    b = sxx2.ravel()

    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def evaluate_signal_diff_style(
    ground: np.ndarray,
    pred: np.ndarray,
    fs: int = 200,
) -> Dict[str, float]:
    """
    Match DIFF util.py style:
      ground = scale(ground)
      pred = scale(pred)
      pcc = abs(pearsonr)
      prd = PRD * 100
      spc = spectrogram correlation
      mae, mse on scaled signals
    """
    ground_s = _safe_scale(ground)
    pred_s = _safe_scale(pred)

    return {
        "mae": mae_np(ground_s, pred_s),
        "rmse": rmse_np(ground_s, pred_s),
        "prd": prd_np(ground_s, pred_s),
        "pcc": pcc_np(ground_s, pred_s),
        "spc": spc_np(ground_s, pred_s, fs=fs),
        "mse": float(np.mean((ground_s - pred_s) ** 2)),
    }


@torch.no_grad()
def batch_reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: int = 200,
) -> Dict[str, float]:
    """
    pred, target: [B, 1, L]
    DIFF-like waveform metrics.
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    maes, rmses, prds, pccs, spcs, mses = [], [], [], [], [], []

    for i in range(pred_np.shape[0]):
        p = pred_np[i, 0]
        t = target_np[i, 0]

        m = evaluate_signal_diff_style(t, p, fs=fs)

        maes.append(m["mae"])
        rmses.append(m["rmse"])
        prds.append(m["prd"])
        pccs.append(m["pcc"])
        spcs.append(m["spc"])
        mses.append(m["mse"])

    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "prd": float(np.mean(prds)),
        "pcc": float(np.mean(pccs)),
        "spc": float(np.mean(spcs)),
        "mse": float(np.mean(mses)),
    }


# =========================
# PT-style R-peak detection
# =========================
def _bandpass_pt_style(signal: np.ndarray, fs: int) -> np.ndarray:
    low = 5.0
    high = 15.0
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def detect_rpeaks_pt(signal: np.ndarray, fs: int = 200) -> np.ndarray:
    """
    Practical PT-style detector.
    """
    x = np.asarray(signal).astype(np.float64).reshape(-1)

    x_f = _bandpass_pt_style(x, fs)
    dx = np.diff(x_f, prepend=x_f[0])
    sq = dx ** 2

    win_size = max(1, int(round(0.150 * fs)))
    kernel = np.ones(win_size, dtype=np.float64) / win_size
    mwi = np.convolve(sq, kernel, mode="same")

    refractory = max(1, int(round(0.20 * fs)))
    height = np.mean(mwi) + 0.5 * np.std(mwi)

    peaks, _ = find_peaks(mwi, distance=refractory, height=height)
    return peaks.astype(np.int64)


# =========================
# R-peak matching - DIFF-like counts
# =========================
def evaluate_rpeak_counts_diff_style(
    r_ref_list,
    r_ans_list,
    fs: int = 200,
    thr_ms: float = 50.0,
    ignore_sec: float = 0.0,
    length: int | None = None,
) -> Tuple[int, int, int]:
    """
    DIFF util.py-like evaluation:
    - loop reference peaks
    - if >=1 predicted peaks within tolerance: TP += 1
    - extra predictions around same ref counted as FP
    - predictions not detected by any ref counted as FP
    Returns total TP, FP, FN.
    """
    tol = int(thr_ms * fs / 1000.0)

    all_tp = 0
    all_fn = 0
    all_fp = 0

    for r_ref, r_ans in zip(r_ref_list, r_ans_list):
        r_ref = np.asarray(r_ref, dtype=np.int64)
        r_ans = np.asarray(r_ans, dtype=np.int64)

        if ignore_sec > 0 and length is not None:
            on = int(ignore_sec * fs)
            off = int(length - ignore_sec * fs)
            r_ref = r_ref[(r_ref > on) & (r_ref < off)]
            r_ans = r_ans[(r_ans > on) & (r_ans < off)]

        fn = 0
        fp = 0
        tp = 0
        detect_loc = 0

        for ref_peak in r_ref:
            loc = np.where(np.abs(r_ans - ref_peak) <= tol)[0]
            detect_loc += len(loc)

            if len(loc) >= 1:
                tp += 1
                fp += len(loc) - 1
            else:
                fn += 1

        fp += len(r_ans) - detect_loc

        all_tp += tp
        all_fn += fn
        all_fp += fp

    return int(all_tp), int(all_fp), int(all_fn)


def compute_rpeak_metrics_from_counts(tp: int, fp: int, fn: int, percent: bool = True) -> Dict[str, float]:
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * sen * ppv / (sen + ppv) if (sen + ppv) > 0 else 0.0

    scale_factor = 100.0 if percent else 1.0

    return {
        "sen": float(sen * scale_factor),
        "ppv": float(ppv * scale_factor),
        "f1": float(f1 * scale_factor),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


@torch.no_grad()
def batch_rpeak_counts(
    pred: torch.Tensor,
    fqrs_list: List[torch.Tensor],
    fs: int = 200,
    tolerance_ms: float = 50.0,
) -> Tuple[int, int, int]:
    """
    Detect peaks for each predicted FECG segment, then compute total TP/FP/FN.
    """
    pred_np = pred.detach().cpu().numpy()

    r_ref_list = []
    r_ans_list = []

    for i in range(pred_np.shape[0]):
        pred_signal = pred_np[i, 0]
        ref_peaks = fqrs_list[i].detach().cpu().numpy() if hasattr(fqrs_list[i], "detach") else np.asarray(fqrs_list[i])

        pred_peaks = detect_rpeaks_pt(pred_signal, fs=fs)

        r_ref_list.append(ref_peaks)
        r_ans_list.append(pred_peaks)

    return evaluate_rpeak_counts_diff_style(
        r_ref_list=r_ref_list,
        r_ans_list=r_ans_list,
        fs=fs,
        thr_ms=tolerance_ms,
    )


@torch.no_grad()
def batch_rpeak_metrics(
    pred: torch.Tensor,
    fqrs_list: List[torch.Tensor],
    fs: int = 200,
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """
    Backward-compatible function:
    returns metrics computed from total TP/FP/FN, not macro-average.
    """
    tp, fp, fn = batch_rpeak_counts(
        pred=pred,
        fqrs_list=fqrs_list,
        fs=fs,
        tolerance_ms=tolerance_ms,
    )
    return compute_rpeak_metrics_from_counts(tp, fp, fn, percent=True)