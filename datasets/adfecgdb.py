from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyedflib
import wfdb

from config import cfg
from preprocessing.preprocess import preprocess_ecg, segment_pair


def _get_edf_path(record_id: str) -> Path:
    path = Path(cfg.paths.adfecgdb_root) / f"{record_id}.edf"
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")
    return path


def _get_qrs_path(record_id: str) -> Path:
    path = Path(cfg.paths.adfecgdb_root) / f"{record_id}.edf.qrs"
    if not path.exists():
        raise FileNotFoundError(f"QRS file not found: {path}")
    return path


def _read_edf_signals(edf_path: Path) -> Tuple[np.ndarray, List[str], List[int]]:
    reader = pyedflib.EdfReader(str(edf_path))
    n = reader.signals_in_file
    labels = reader.getSignalLabels()
    fs_list = [int(reader.getSampleFrequency(i)) for i in range(n)]
    signals = np.vstack([reader.readSignal(i) for i in range(n)]).astype(np.float32)
    reader.close()
    return signals, labels, fs_list


def _read_qrs_file(record_id: str) -> np.ndarray:
    """
    Read PhysioNet/WFDB annotation file: r01.edf.qrs
    using wfdb.rdann.

    For local files:
      record name should be path without the final '.qrs'
      extension = 'qrs'
    """
    record_base = str(Path(cfg.paths.adfecgdb_root) / f"{record_id}.edf")
    ann = wfdb.rdann(record_base, extension="qrs")
    return np.asarray(ann.sample, dtype=np.int64)


def _split_adfecgdb_channels(signals: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Real labels observed:
    ['Direct_1', 'Abdomen_1', 'Abdomen_2', 'Abdomen_3', 'Abdomen_4']

    So:
      channel 0 = direct fetal ECG
      channel 1..4 = abdominal ECG
    """
    if signals.shape[0] < 5:
        raise ValueError(f"Expected at least 5 channels in ADFECGDB EDF, got {signals.shape[0]}")

    fecg = signals[0, :]        # Direct_1
    aecg_all = signals[1:5, :]  # Abdomen_1..4
    return aecg_all, fecg


def load_adfecgdb_record(record_id: str) -> Dict:
    edf_path = _get_edf_path(record_id)
    qrs_path = _get_qrs_path(record_id)

    signals, labels, fs_list = _read_edf_signals(edf_path)
    aecg_all, fecg = _split_adfecgdb_channels(signals, labels)
    fqrs = _read_qrs_file(record_id)

    unique_fs = sorted(set(fs_list))
    if len(unique_fs) != 1:
        raise ValueError(f"Channels in {edf_path} do not share same sampling rate: {unique_fs}")
    fs = unique_fs[0]

    if cfg.data.use_single_channel:
        ch = cfg.data.single_channel_index
        aecg = aecg_all[ch:ch + 1, :]
    else:
        aecg = aecg_all

    return {
        "dataset": "ADFECGDB",
        "record_id": record_id,
        "edf_path": str(edf_path),
        "qrs_path": str(qrs_path),
        "labels": labels,
        "fs": fs,
        "aecg_raw": aecg,
        "fecg_raw": fecg.astype(np.float32),
        "fqrs_raw": fqrs,
        "num_samples_raw": aecg.shape[-1],
    }


def preprocess_adfecgdb_record(record: Dict) -> Dict:
    aecg_pre = preprocess_ecg(
        signal=record["aecg_raw"],
        orig_fs=record["fs"],
        target_fs=cfg.data.target_fs,
        bandpass_low=cfg.data.bandpass_low,
        bandpass_high=cfg.data.bandpass_high,
        bandpass_order=cfg.data.bandpass_order,
        notch_freqs=cfg.data.notch_freqs,
        notch_q=cfg.data.notch_q,
        normalize=True,
    )

    fecg_pre = preprocess_ecg(
        signal=record["fecg_raw"],
        orig_fs=record["fs"],
        target_fs=cfg.data.target_fs,
        bandpass_low=cfg.data.bandpass_low,
        bandpass_high=cfg.data.bandpass_high,
        bandpass_order=cfg.data.bandpass_order,
        notch_freqs=cfg.data.notch_freqs,
        notch_q=cfg.data.notch_q,
        normalize=True,
    )[0]

    fqrs_pre = np.round(record["fqrs_raw"] * cfg.data.target_fs / record["fs"]).astype(np.int64)

    return {
        **record,
        "aecg_pre": aecg_pre,
        "fecg_pre": fecg_pre,
        "fqrs_pre": fqrs_pre,
        "fs_pre": cfg.data.target_fs,
        "num_samples_pre": aecg_pre.shape[-1],
    }


def segment_adfecgdb_record(record: Dict) -> Dict:
    aecg_segs, fecg_segs = segment_pair(
        aecg=record["aecg_pre"],
        fecg=record["fecg_pre"],
        fs=record["fs_pre"],
        seg_sec=cfg.data.seg_sec,
        overlap=cfg.data.overlap,
    )

    return {
        **record,
        "aecg_segments": aecg_segs,
        "fecg_segments": fecg_segs,
        "num_segments": aecg_segs.shape[0],
    }


def load_all_adfecgdb_records() -> List[Dict]:
    records = []
    for rid in cfg.data.adfecgdb_records:
        rec = load_adfecgdb_record(rid)
        rec = preprocess_adfecgdb_record(rec)
        rec = segment_adfecgdb_record(rec)
        records.append(rec)
    return records