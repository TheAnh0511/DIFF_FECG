from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.adfecgdb import load_all_adfecgdb_records


class ADFECGDBSegmentDataset(Dataset):
    """
    Dataset of segmented ADFECGDB samples.

    Supports:
    - all records
    - or a selected subset of records for LOOCV
    """
    def __init__(self, selected_record_ids: List[str] | None = None) -> None:
        super().__init__()

        all_records = load_all_adfecgdb_records()

        if selected_record_ids is not None:
            selected_record_ids = set(selected_record_ids)
            self.records = [r for r in all_records if r["record_id"] in selected_record_ids]
        else:
            self.records = all_records

        self.samples: List[Dict] = []

        for rec in self.records:
            aecg_segments = rec["aecg_segments"]   # [N, C, L]
            fecg_segments = rec["fecg_segments"]   # [N, 1, L]
            fqrs = rec["fqrs_pre"]                 # [M]
            fs = rec["fs_pre"]

            seg_len = aecg_segments.shape[-1]
            hop = seg_len // 2   # because overlap = 0.5

            for i in range(aecg_segments.shape[0]):
                start = i * hop
                end = start + seg_len

                seg_qrs = fqrs[(fqrs >= start) & (fqrs < end)] - start

                self.samples.append({
                    "dataset": rec["dataset"],
                    "record_id": rec["record_id"],
                    "segment_idx": i,
                    "aecg": aecg_segments[i].astype(np.float32),
                    "fecg": fecg_segments[i].astype(np.float32),
                    "fqrs": seg_qrs.astype(np.int64),
                    "fs": fs,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            "aecg": torch.from_numpy(sample["aecg"]),   # [1, L]
            "fecg": torch.from_numpy(sample["fecg"]),   # [1, L]
            "record_id": sample["record_id"],
            "segment_idx": sample["segment_idx"],
            "fqrs": torch.from_numpy(sample["fqrs"]),
            "fs": sample["fs"],
        }


def collate_adfecgdb(batch: List[Dict]) -> Dict:
    aecg = torch.stack([b["aecg"] for b in batch], dim=0)   # [B, 1, L]
    fecg = torch.stack([b["fecg"] for b in batch], dim=0)   # [B, 1, L]
    record_ids = [b["record_id"] for b in batch]
    segment_idx = torch.tensor([b["segment_idx"] for b in batch], dtype=torch.long)
    fqrs = [b["fqrs"] for b in batch]
    fs = batch[0]["fs"] if batch else None

    return {
        "aecg": aecg,
        "fecg": fecg,
        "record_id": record_ids,
        "segment_idx": segment_idx,
        "fqrs": fqrs,
        "fs": fs,
    }


def build_loocv_datasets(all_record_ids: List[str], test_record_id: str):
    train_ids = [r for r in all_record_ids if r != test_record_id]
    test_ids = [test_record_id]

    train_set = ADFECGDBSegmentDataset(selected_record_ids=train_ids)
    test_set = ADFECGDBSegmentDataset(selected_record_ids=test_ids)

    return train_set, test_set