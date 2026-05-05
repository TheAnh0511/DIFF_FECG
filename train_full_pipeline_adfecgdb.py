from __future__ import annotations

from pathlib import Path
import copy
import statistics

import torch
from torch.utils.data import DataLoader

from config import cfg
from datasets.torch_dataset import build_loocv_datasets, collate_adfecgdb
from models.separator import DualHeadSeparator
from models.diffusion_refiner import ConditionalDenoiser1D, GaussianDiffusion1D
from models.losses import separator_hybrid_loss, diffusion_hybrid_loss
from utils.seed import set_seed
from utils.metrics import (
    batch_reconstruction_metrics,
    batch_rpeak_counts,
    compute_rpeak_metrics_from_counts,
)


# =========================================================
# Debug switches
# =========================================================
DEBUG_ONE_CHANNEL_ONLY = False
DEBUG_CHANNEL_INDEX = 0

DEBUG_ONE_RECORD_ONLY = False
DEBUG_RECORD_ID = "r01"


def build_diffusion_model(device: torch.device):
    denoiser = ConditionalDenoiser1D(
        in_ch=cfg.model.diff_in_ch,
        cond_ch=cfg.model.diff_cond_ch,
        base_ch=cfg.model.diff_base_ch,
        time_dim=cfg.model.time_embed_dim,
    ).to(device)

    diffusion = GaussianDiffusion1D(
        denoiser=denoiser,
        num_steps=cfg.diffusion.num_steps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
    ).to(device)

    return diffusion


def build_joint_optimizer(separator, diffusion):
    return torch.optim.Adam(
        [
            {"params": separator.parameters(), "lr": cfg.optim.sep_lr},
            {"params": diffusion.parameters(), "lr": cfg.optim.diff_lr},
        ],
        weight_decay=cfg.optim.weight_decay,
    )


def run_epoch_full(separator, diffusion, loader, optimizer, device, train: bool = True):
    if train:
        separator.train()
        diffusion.train()
    else:
        separator.eval()
        diffusion.eval()

    total_loss = 0.0
    n_batches = 0

    total_mae = 0.0
    total_rmse = 0.0
    total_prd = 0.0
    total_pcc = 0.0
    total_spc = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for batch in loader:
        aecg = batch["aecg"].to(device)   # [B, 1, L]
        fecg = batch["fecg"].to(device)   # [B, 1, L]
        fqrs = batch["fqrs"]
        fs = batch["fs"]

        batch_size = aecg.shape[0]

        with torch.set_grad_enabled(train):
            # -------------------------
            # separator stage
            # -------------------------
            pred_mecg, pred_fecg = separator(aecg)

            sep_loss, _ = separator_hybrid_loss(
                x_aecg=aecg,
                pred_mecg=pred_mecg,
                pred_fecg=pred_fecg,
                target_fecg=fecg,
                lambda_fecg_l1=cfg.loss.lambda_fecg_l1,
                lambda_fecg_corr=cfg.loss.lambda_fecg_corr,
                lambda_mix=cfg.loss.lambda_mix,
                lambda_sep_decorr=getattr(cfg.loss, "lambda_sep_decorr", 0.1),
            )

            # -------------------------
            # diffusion stage
            # -------------------------
            residual = aecg - pred_mecg
            cond = torch.cat([pred_fecg, residual], dim=1)   # [B, 2, L]

            t = torch.randint(
                low=0,
                high=cfg.diffusion.num_steps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(fecg)
            x_t = diffusion.q_sample(fecg, t, noise)

            eps_hat = diffusion.predict_eps(x_t, cond, t)
            x0_hat = diffusion.predict_x0_from_eps(x_t, eps_hat, t)

            diff_loss, _ = diffusion_hybrid_loss(
                noise=noise,
                eps_hat=eps_hat,
                x0_hat=x0_hat,
                target_x0=fecg,
                fqrs_list=fqrs,
                lambda_noise=cfg.loss.lambda_diff_noise,
                lambda_x0=cfg.loss.lambda_diff_x0,
                lambda_corr=cfg.loss.lambda_diff_corr,
                lambda_qrs=cfg.loss.lambda_diff_qrs,
            )

            alpha_sep = getattr(cfg.loss, "alpha_total_sep", 1.0)
            alpha_diff = getattr(cfg.loss, "alpha_total_diff", 1.0)

            total = alpha_sep * sep_loss + alpha_diff * diff_loss

            if train:
                optimizer.zero_grad()
                total.backward()
                if cfg.optim.grad_clip is not None and cfg.optim.grad_clip > 0:
                    all_params = list(separator.parameters()) + list(diffusion.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, cfg.optim.grad_clip)
                optimizer.step()

        # metrics on refined x0_hat
        recon = batch_reconstruction_metrics(x0_hat, fecg, fs=fs)
        tp, fp, fn = batch_rpeak_counts(x0_hat, fqrs, fs=fs, tolerance_ms=50.0)

        total_loss += float(total.item())
        total_mae += recon["mae"]
        total_rmse += recon["rmse"]
        total_prd += recon["prd"]
        total_pcc += recon["pcc"]
        total_spc += recon["spc"]

        total_tp += tp
        total_fp += fp
        total_fn += fn

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
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    rpk = compute_rpeak_metrics_from_counts(total_tp, total_fp, total_fn, percent=True)

    return {
        "loss": total_loss / n_batches,
        "mae": total_mae / n_batches,
        "rmse": total_rmse / n_batches,
        "prd": total_prd / n_batches,
        "pcc": total_pcc / n_batches,
        "spc": total_spc / n_batches,
        "sen": rpk["sen"],
        "ppv": rpk["ppv"],
        "f1": rpk["f1"],
        "tp": rpk["tp"],
        "fp": rpk["fp"],
        "fn": rpk["fn"],
    }


@torch.no_grad()
def evaluate_sampling(separator, diffusion, loader, device):
    separator.eval()
    diffusion.eval()

    total_mae = 0.0
    total_rmse = 0.0
    total_prd = 0.0
    total_pcc = 0.0
    total_spc = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_batches = 0

    for batch in loader:
        aecg = batch["aecg"].to(device)
        fecg = batch["fecg"].to(device)
        fqrs = batch["fqrs"]
        fs = batch["fs"]

        pred_mecg, pred_fecg = separator(aecg)
        residual = aecg - pred_mecg
        cond = torch.cat([pred_fecg, residual], dim=1)

        samples = []
        for _ in range(cfg.diffusion.infer_k):
            x_hat = diffusion.sample(
                cond=cond,
                shape=fecg.shape,
                device=device,
            )
            samples.append(x_hat)

        x_final = torch.stack(samples, dim=0).mean(dim=0)

        recon = batch_reconstruction_metrics(x_final, fecg, fs=fs)
        tp, fp, fn = batch_rpeak_counts(x_final, fqrs, fs=fs, tolerance_ms=50.0)

        total_mae += recon["mae"]
        total_rmse += recon["rmse"]
        total_prd += recon["prd"]
        total_pcc += recon["pcc"]
        total_spc += recon["spc"]

        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_batches += 1

    if n_batches == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "prd": 0.0,
            "pcc": 0.0,
            "spc": 0.0,
            "sen": 0.0,
            "ppv": 0.0,
            "f1": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    rpk = compute_rpeak_metrics_from_counts(total_tp, total_fp, total_fn, percent=True)

    return {
        "mae": total_mae / n_batches,
        "rmse": total_rmse / n_batches,
        "prd": total_prd / n_batches,
        "pcc": total_pcc / n_batches,
        "spc": total_spc / n_batches,
        "sen": rpk["sen"],
        "ppv": rpk["ppv"],
        "f1": rpk["f1"],
        "tp": rpk["tp"],
        "fp": rpk["fp"],
        "fn": rpk["fn"],
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
        batch_size=cfg.optim.diff_batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_adfecgdb,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.optim.diff_batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_adfecgdb,
    )

    separator = DualHeadSeparator(
        in_ch=cfg.model.in_ch,
        base_ch=cfg.model.base_ch,
    ).to(device)

    diffusion = build_diffusion_model(device)
    optimizer = build_joint_optimizer(separator, diffusion)

    best_test_loss = float("inf")
    best_metrics = None

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print("-" * 100)
    print(f"[FULL] Channel={channel_name} | test_record={test_record_id} | device={device}")
    print(f"Train segments={len(train_set)} | Test segments={len(test_set)}")

    num_epochs = max(cfg.optim.sep_epochs, cfg.optim.diff_epochs)

    for epoch in range(1, num_epochs + 1):
        train_stats = run_epoch_full(
            separator=separator,
            diffusion=diffusion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        test_stats = run_epoch_full(
            separator=separator,
            diffusion=diffusion,
            loader=test_loader,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        print(
            f"[FULL] [{channel_name}] [test={test_record_id}] [Epoch {epoch:03d}] "
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
                    "separator_state_dict": separator.state_dict(),
                    "diffusion_state_dict": diffusion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": best_test_loss,
                    "test_metrics": best_metrics,
                },
                ckpt_dir / f"best_fullpipe_{channel_name}_test_{test_record_id}.pt",
            )

    final_metrics = evaluate_sampling(
        separator=separator,
        diffusion=diffusion,
        loader=test_loader,
        device=device,
    )

    print(
        f"[FULL-FINAL] {channel_name} | test={test_record_id} | "
        f"MAE={final_metrics['mae']:.4f}, PRD={final_metrics['prd']:.4f}, "
        f"PCC={final_metrics['pcc']:.4f}, SPC={final_metrics['spc']:.4f}, "
        f"Sen={final_metrics['sen']:.2f}, PPV={final_metrics['ppv']:.2f}, F1={final_metrics['f1']:.2f}"
    )

    return {
        "channel_name": channel_name,
        "test_record_id": test_record_id,
        **final_metrics,
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
    if DEBUG_ONE_CHANNEL_ONLY:
        channel_indices = [DEBUG_CHANNEL_INDEX]
    else:
        channel_indices = list(range(4))

    if DEBUG_ONE_RECORD_ONLY:
        test_record_ids = [DEBUG_RECORD_ID]
    else:
        test_record_ids = cfg.data.adfecgdb_records

    all_channel_summaries = []

    for ch in channel_indices:
        channel_name = cfg.data.adfecgdb_abd_channel_names[ch]
        fold_results = []

        for test_record_id in test_record_ids:
            r = train_one_fold(channel_index=ch, test_record_id=test_record_id)
            fold_results.append(r)

        summary = summarize_channel_results(channel_name, fold_results)
        all_channel_summaries.append(summary)

    print("\n" + "=" * 100)
    print("FINAL FULL-PIPELINE LOOCV RESULTS PER SINGLE CHANNEL")
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