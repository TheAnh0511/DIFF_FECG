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
from models.losses import diffusion_hybrid_loss
from utils.seed import set_seed
from utils.metrics import batch_reconstruction_metrics, batch_rpeak_metrics


def build_separator_checkpoint_path(channel_name: str, test_record_id: str) -> Path:
    ckpt_dir = Path("checkpoints")
    return ckpt_dir / f"best_separator_{channel_name}_test_{test_record_id}.pt"


def load_separator_for_fold(channel_name: str, test_record_id: str, device: torch.device):
    ckpt_path = build_separator_checkpoint_path(channel_name, test_record_id)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing separator checkpoint: {ckpt_path}")

    model = DualHeadSeparator(
        in_ch=cfg.model.in_ch,
        base_ch=cfg.model.base_ch,
    ).to(device)

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


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


def make_condition(separator: DualHeadSeparator, aecg: torch.Tensor):
    with torch.no_grad():
        pred_mecg, pred_fecg = separator(aecg)
        residual = aecg - pred_mecg
        cond = torch.cat([pred_fecg, residual], dim=1)  # [B, 2, L]
    return pred_mecg, pred_fecg, residual, cond


def run_diffusion_epoch(
    separator,
    diffusion,
    loader,
    optimizer,
    device,
    train: bool = True,
):
    if train:
        diffusion.train()
    else:
        diffusion.eval()

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
        aecg = batch["aecg"].to(device)   # [B, 1, L]
        fecg = batch["fecg"].to(device)   # [B, 1, L]
        fqrs = batch["fqrs"]
        fs = batch["fs"]

        _, _, _, cond = make_condition(separator, aecg)

        B = fecg.shape[0]
        t = torch.randint(
            low=0,
            high=cfg.diffusion.num_steps,
            size=(B,),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(fecg)
        x_t = diffusion.q_sample(fecg, t, noise)

        with torch.set_grad_enabled(train):
            eps_hat = diffusion.predict_eps(x_t, cond, t)
            x0_hat = diffusion.predict_x0_from_eps(x_t, eps_hat, t)

            loss, _ = diffusion_hybrid_loss(
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

            if train:
                optimizer.zero_grad()
                loss.backward()
                if cfg.optim.grad_clip is not None and cfg.optim.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.optim.grad_clip)
                optimizer.step()

        recon = batch_reconstruction_metrics(x0_hat, fecg, fs=fs)
        rpk = batch_rpeak_metrics(x0_hat, fqrs, fs=fs, tolerance_ms=50.0)

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


@torch.no_grad()
def evaluate_diffusion_sampling(separator, diffusion, loader, device):
    diffusion.eval()

    total_mae = 0.0
    total_rmse = 0.0
    total_prd = 0.0
    total_pcc = 0.0
    total_spc = 0.0

    total_sen = 0.0
    total_ppv = 0.0
    total_f1 = 0.0
    n_batches = 0

    for batch in loader:
        aecg = batch["aecg"].to(device)
        fecg = batch["fecg"].to(device)
        fqrs = batch["fqrs"]
        fs = batch["fs"]

        _, _, _, cond = make_condition(separator, aecg)

        # multiple reconstructions K
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
        rpk = batch_rpeak_metrics(x_final, fqrs, fs=fs, tolerance_ms=50.0)

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

    separator = load_separator_for_fold(channel_name, test_record_id, device)
    diffusion = build_diffusion_model(device)

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

    optimizer = torch.optim.Adam(
        diffusion.parameters(),
        lr=cfg.optim.diff_lr,
        weight_decay=cfg.optim.weight_decay,
    )

    best_test_loss = float("inf")
    best_metrics = None

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print("-" * 100)
    print(f"[DIFF] Channel={channel_name} | test_record={test_record_id} | device={device}")
    print(f"Train segments={len(train_set)} | Test segments={len(test_set)}")

    for epoch in range(1, cfg.optim.diff_epochs + 1):
        train_stats = run_diffusion_epoch(
            separator=separator,
            diffusion=diffusion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
        )
        test_stats = run_diffusion_epoch(
            separator=separator,
            diffusion=diffusion,
            loader=test_loader,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        print(
            f"[DIFF] [{channel_name}] [test={test_record_id}] [Epoch {epoch:03d}] "
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
                    "model_state_dict": diffusion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": best_test_loss,
                    "test_metrics": best_metrics,
                },
                ckpt_dir / f"best_diffusion_{channel_name}_test_{test_record_id}.pt",
            )

    # final sampling-based evaluation with K reconstructions
    sampling_metrics = evaluate_diffusion_sampling(
        separator=separator,
        diffusion=diffusion,
        loader=test_loader,
        device=device,
    )

    print(
        f"[DIFF-FINAL] {channel_name} | test={test_record_id} | "
        f"MAE={sampling_metrics['mae']:.4f}, PRD={sampling_metrics['prd']:.4f}, "
        f"PCC={sampling_metrics['pcc']:.4f}, SPC={sampling_metrics['spc']:.4f}, "
        f"Sen={sampling_metrics['sen']:.2f}, PPV={sampling_metrics['ppv']:.2f}, F1={sampling_metrics['f1']:.2f}"
    )

    return {
        "channel_name": channel_name,
        "test_record_id": test_record_id,
        **sampling_metrics,
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
        channel_name = cfg.data.adfecgdb_abd_channel_names[ch]
        fold_results = []

        for test_record_id in cfg.data.adfecgdb_records:
            r = train_one_fold(channel_index=ch, test_record_id=test_record_id)
            fold_results.append(r)

        summary = summarize_channel_results(channel_name, fold_results)
        all_channel_summaries.append(summary)

    print("\n" + "=" * 100)
    print("FINAL DIFFUSION LOOCV RESULTS PER SINGLE CHANNEL")
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