from __future__ import annotations

from typing import List, Dict

from datasets.adfecgdb import load_all_adfecgdb_records


def build_adfecgdb_segment_dataset() -> List[Dict]:
    records = load_all_adfecgdb_records()
    samples = []

    for rec in records:
        aecg_segments = rec["aecg_segments"]
        fecg_segments = rec["fecg_segments"]

        for i in range(aecg_segments.shape[0]):
            samples.append({
                "dataset": rec["dataset"],
                "record_id": rec["record_id"],
                "segment_idx": i,
                "aecg": aecg_segments[i],
                "fecg": fecg_segments[i],
            })

    return samples


def summarize_segment_dataset(samples: List[Dict]) -> None:
    print(f"Total segments: {len(samples)}")
    if not samples:
        return
    print(f"AECG shape example: {samples[0]['aecg'].shape}")
    print(f"FECG shape example: {samples[0]['fecg'].shape}")
    print(f"Example record     : {samples[0]['record_id']}")
