from __future__ import annotations

import torch
import torch.nn.functional as F


# =========================
# basic losses
# =========================
def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def correlation_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.squeeze(1)
    target = target.squeeze(1)

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    num = torch.sum(pred * target, dim=1)
    den = torch.sqrt(torch.sum(pred ** 2, dim=1) * torch.sum(target ** 2, dim=1) + eps)
    corr = num / (den + eps)
    return 1.0 - corr.mean()


def decorrelation_loss(sig_a: torch.Tensor, sig_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Encourage pred_mecg and pred_fecg to be less correlated.
    Lower is better.
    """
    a = sig_a.squeeze(1)
    b = sig_b.squeeze(1)

    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)

    num = torch.sum(a * b, dim=1)
    den = torch.sqrt(torch.sum(a ** 2, dim=1) * torch.sum(b ** 2, dim=1) + eps)
    corr = num / (den + eps)

    return torch.mean(torch.abs(corr))


def qrs_weight_map_from_fqrs(fqrs_list, length: int, device: torch.device, radius: int = 8):
    """
    Build simple weight map emphasizing neighborhoods around reference fetal peaks.
    """
    B = len(fqrs_list)
    w = torch.ones(B, 1, length, device=device)

    for i, peaks in enumerate(fqrs_list):
        peaks = peaks.detach().cpu().numpy() if hasattr(peaks, "detach") else peaks
        for p in peaks:
            s = max(0, int(p) - radius)
            e = min(length, int(p) + radius + 1)
            w[i, 0, s:e] = 2.0
    return w


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target) * weight)


# =========================
# separator hybrid loss
# =========================
def separator_hybrid_loss(
    x_aecg: torch.Tensor,
    pred_mecg: torch.Tensor,
    pred_fecg: torch.Tensor,
    target_fecg: torch.Tensor,
    lambda_fecg_l1: float = 1.0,
    lambda_fecg_corr: float = 0.2,
    lambda_mix: float = 0.5,
    lambda_sep_decorr: float = 0.1,
):
    """
    aECG ≈ pred_mECG + pred_fECG
    pred_fECG should match target fetal ECG
    pred_mECG and pred_fECG should be less correlated
    """
    loss_fecg = l1_loss(pred_fecg, target_fecg)
    loss_corr = correlation_loss(pred_fecg, target_fecg)

    recon_mix = pred_mecg + pred_fecg
    loss_mix = l1_loss(recon_mix, x_aecg)

    loss_decorr = decorrelation_loss(pred_mecg, pred_fecg)

    total = (
        lambda_fecg_l1 * loss_fecg
        + lambda_fecg_corr * loss_corr
        + lambda_mix * loss_mix
        + lambda_sep_decorr * loss_decorr
    )

    loss_dict = {
        "loss_fecg_l1": loss_fecg,
        "loss_fecg_corr": loss_corr,
        "loss_mix": loss_mix,
        "loss_decorr": loss_decorr,
        "loss_total": total,
    }
    return total, loss_dict


# =========================
# diffusion hybrid loss
# =========================
def diffusion_hybrid_loss(
    noise: torch.Tensor,
    eps_hat: torch.Tensor,
    x0_hat: torch.Tensor,
    target_x0: torch.Tensor,
    fqrs_list=None,
    lambda_noise: float = 1.0,
    lambda_x0: float = 1.0,
    lambda_corr: float = 0.2,
    lambda_qrs: float = 0.1,
):
    loss_noise = mse_loss(eps_hat, noise)
    loss_x0 = l1_loss(x0_hat, target_x0)
    loss_corr = correlation_loss(x0_hat, target_x0)

    if fqrs_list is not None:
        w = qrs_weight_map_from_fqrs(fqrs_list, target_x0.shape[-1], target_x0.device)
        loss_qrs = weighted_l1(x0_hat, target_x0, w)
    else:
        loss_qrs = torch.tensor(0.0, device=target_x0.device)

    total = (
        lambda_noise * loss_noise
        + lambda_x0 * loss_x0
        + lambda_corr * loss_corr
        + lambda_qrs * loss_qrs
    )

    loss_dict = {
        "loss_noise": loss_noise,
        "loss_x0": loss_x0,
        "loss_corr": loss_corr,
        "loss_qrs": loss_qrs,
        "loss_total": total,
    }
    return total, loss_dict