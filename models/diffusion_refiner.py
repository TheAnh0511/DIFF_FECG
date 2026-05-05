from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: [B]
    returns: [B, dim]
    """
    device = timesteps.device
    half = dim // 2
    emb_scale = math.log(10000.0) / max(half - 1, 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb_scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)

        t_add = self.time_proj(t_emb).unsqueeze(-1)
        h = h + t_add

        h = self.conv2(h)
        h = self.bn2(h)

        return self.act(h + self.skip(x))


class ConditionalDenoiser1D(nn.Module):
    """
    Input to denoiser:
      x_t       : noisy target [B, 1, L]
      cond      : [B, 2, L] = concat(coarse_fecg, residual)
      timestep  : [B]
    Output:
      predicted noise eps_hat [B, 1, L]
    """
    def __init__(self, in_ch: int = 1, cond_ch: int = 2, base_ch: int = 64, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim

        self.in_proj = nn.Conv1d(in_ch + cond_ch, base_ch, kernel_size=3, padding=1)

        self.rb1 = ResBlock1D(base_ch, base_ch, time_dim)
        self.rb2 = ResBlock1D(base_ch, base_ch, time_dim)
        self.rb3 = ResBlock1D(base_ch, base_ch, time_dim)

        self.out_proj = nn.Conv1d(base_ch, 1, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)

        x = torch.cat([x_t, cond], dim=1)
        x = self.in_proj(x)

        x = self.rb1(x, t_emb)
        x = self.rb2(x, t_emb)
        x = self.rb3(x, t_emb)

        return self.out_proj(x)


class GaussianDiffusion1D(nn.Module):
    def __init__(self, denoiser: nn.Module, num_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 3.5e-2) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.num_steps = num_steps

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise
        """
        s1 = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return s1 * x0 + s2 * noise

    def predict_eps(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        return self.denoiser(x_t, cond, t)

    def predict_x0_from_eps(self, x_t: torch.Tensor, eps_hat: torch.Tensor, t: torch.Tensor):
        s1 = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (x_t - s2 * eps_hat) / (s1 + 1e-8)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)

        eps_hat = self.predict_eps(x_t, cond, t)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8)) * eps_hat
        )

        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1, 1)
        sample = mean + nonzero_mask * torch.sqrt(beta_t) * noise
        return sample

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, shape, device: torch.device):
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, t)
        return x