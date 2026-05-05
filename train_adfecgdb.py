from __future__ import annotations

from pathlib import Path
import copy
import statistics

import torch
from torch.utils.data import DataLoader

from config import cfg
from datasets.torch_dataset import build_loocv_datasets, collate_adfecgdb
from models.separator import DualHeadSeparator
from models.losses import separator_hybrid_loss
from utils.seed import set_seed
from utils.metrics import batch_reconstruction_metrics, batch_rpeak_metrics


def run_epoch(model, loader, optimizer, device, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0

    total_mae = 0.0
    total_rmse = 0.0
    total_prd = 0.0
    total_pcc = 0.0
    total_spc = 0.0

    total_sen = 0.0
    total_ppv = 0.0
    total_f1 = 0.0

    for batch in loader:
        aecg = batch["aecg"].to(device)
        fecg = batch["fecg"].to(device)
        fqrs = batch["fqrs"]
        fs = batch["fs"]

        with torch.set_grad_enabled(train):
            pred_mecg, pred_fecg = model(aecg)

            loss, _ = separator_hybrid_loss(
                x_aecg=aecg,
                pred_mecg=pred_mecg,
                pred_fecg=pred_fecg,
                target_fecg=fecg,
                lambda_fecg_l1=cfg.loss.lambda_fecg_l1,
                lambda_fecg_corr=cfg.loss.lambda_fecg_corr,
                lambda_mix=cfg.loss.lambda_mix,
                lambda_mecg_energy=cfg.loss.lambda_mecg_energy,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                if cfg.optim.grad_clip is not None and cfg.optim.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                optimizer.step()

        recon = batch_reconstruction_metrics(pred_fecg, fecg, fs=fs)
        rpk = batch_rpeak_metrics(pred_fecg, fqrs, fs=fs, tolerance_ms=50.0)

        total_loss += float(loss.item())
        total_mae += recon["mae"]
        total_rmse += recon["rmse"]
        total_prd += recon["prd"]
        total_pcc += recon["pcc"]
        total_spc += recon["spc"]

        total_sen += rpk["sen"]
        total_ppv += rpk["ppv"]
        total_f1 += rpk["f1"]

        n_batches += 1

    if n_batches == 0:
        return {
            "loss": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "prd": 0.0,
            "pcc": 0.0,
            "spc": 0.0,
            "sen": 0.0,
            "ppv": 0.0,
            "f1": 0.0,
        }

    return {
        "loss": total_loss / n_batches,
        "mae": total_mae / n_batches,
        "rmse": total_rmse / n_batches,
        "prd": total_prd / n_batches,
        "pcc": total_pcc / n_batches,
        "spc": total_spc / n_batches,
        "sen": total_sen / n_batches,
        "ppv": total_ppv / n_batches,
        "f1": total_f1 / n_batches,
    }


def train_one_fold(channel_index: int, test_record_id: str):
    cfg.data.use_single_channel = True
    cfg.data.single_channel_index = channel_index
    channel_name = cfg.data.adfecgdb_abd_channel_names[channel_index]

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_record_ids = cfg.data.adfecgdb_records
    train_set, test_set = build_loocv_datasets(all_record_ids, test_record_id)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.optim.sep_batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_adfecgdb,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.optim.sep_batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_adfecgdb,
    )

    model = DualHeadSeparator(
        in_ch=cfg.model.in_ch,
        base_ch=cfg.model.base_ch,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optim.sep_lr,
        weight_decay=cfg.optim.weight_decay,
    )

    best_test_loss = float("inf")
    best_metrics = None

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print("-" * 80)
    print(f"Channel={channel_name} | LOOCV test record={test_record_id} | device={device}")
    print(f"Train segments={len(train_set)} | Test segments={len(test_set)}")

    for epoch in range(1, cfg.optim.sep_epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, train=True)
        test_stats = run_epoch(model, test_loader, optimizer, device, train=False)

        print(
            f"[{channel_name}] [test={test_record_id}] [Epoch {epoch:03d}] "
            f"train_loss={train_stats['loss']:.5f} | "
            f"test_loss={test_stats['loss']:.5f}, "
            f"test_MAE={test_stats['mae']:.4f}, test_PRD={test_stats['prd']:.4f}, "
            f"test_PCC={test_stats['pcc']:.4f}, test_SPC={test_stats['spc']:.4f}, "
            f"test_Sen={test_stats['sen']:.2f}, test_PPV={test_stats['ppv']:.2f}, test_F1={test_stats['f1']:.2f}"
        )

        if test_stats["loss"] < best_test_loss:
            best_test_loss = test_stats["loss"]
            best_metrics = copy.deepcopy(test_stats)

            torch.save(
                {
                    "channel_index": channel_index,
                    "channel_name": channel_name,
                    "test_record_id": test_record_id,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": best_test_loss,
                    "test_metrics": best_metrics,
                },
                ckpt_dir / f"best_separator_{channel_name}_test_{test_record_id}.pt",
            )

    return {
        "channel_name": channel_name,
        "test_record_id": test_record_id,
        **best_metrics,
    }


def summarize_channel_results(channel_name: str, results):
    keys = ["mae", "prd", "pcc", "spc", "sen", "ppv", "f1"]
    summary = {"channel_name": channel_name}
    for k in keys:
        vals = [r[k] for r in results]
        summary[k + "_mean"] = statistics.mean(vals)
        summary[k + "_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return summary


def main():
    all_channel_summaries = []

    for ch in range(4):
        cfg.data.single_channel_index = ch
        channel_name = cfg.data.adfecgdb_abd_channel_names[ch]

        fold_results = []
        for test_record_id in cfg.data.adfecgdb_records:
            r = train_one_fold(channel_index=ch, test_record_id=test_record_id)
            fold_results.append(r)

        summary = summarize_channel_results(channel_name, fold_results)
        all_channel_summaries.append(summary)

    print("\n" + "=" * 100)
    print("FINAL LOOCV RESULTS PER SINGLE CHANNEL")
    print("=" * 100)
    for s in all_channel_summaries:
        print(
            f"{s['channel_name']}: "
            f"MAE={s['mae_mean']:.4f}±{s['mae_std']:.4f}, "
            f"PRD={s['prd_mean']:.4f}±{s['prd_std']:.4f}, "
            f"PCC={s['pcc_mean']:.4f}±{s['pcc_std']:.4f}, "
            f"SPC={s['spc_mean']:.4f}±{s['spc_std']:.4f}, "
            f"Sen={s['sen_mean']:.2f}±{s['sen_std']:.2f}, "
            f"PPV={s['ppv_mean']:.2f}±{s['ppv_std']:.2f}, "
            f"F1={s['f1_mean']:.2f}±{s['f1_std']:.2f}"
        )


if __name__ == "__main__":
    main()