from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathConfig:
    project_root: str = r"D:\project_fetal_ecg"
    code_root: str = r"D:\project_fetal_ecg\DIFF_Fetal_ECG_project"

    adfecgdb_root: str = r"D:\project_fetal_ecg\ADFECGDB"
    bddb_root: str = r"D:\project_fetal_ecg\BDDB"
    fecgsyndb_root: str = r"D:\project_fetal_ecg\FECGSYNDB"


@dataclass
class DataConfig:
    adfecgdb_records: List[str] = field(default_factory=lambda: ["r01", "r04", "r07", "r08", "r10"])
    adfecgdb_fs: int = 1000

    use_single_channel: bool = True
    single_channel_index: int = 0
    adfecgdb_abd_channel_names: List[str] = field(default_factory=lambda: [
        "Abdomen_1", "Abdomen_2", "Abdomen_3", "Abdomen_4"
    ])

    bddb_fs: int = 500
    max_records_bddb: Optional[int] = None

    target_fs: int = 200
    bandpass_low: float = 7.5
    bandpass_high: float = 75.0
    bandpass_order: int = 3
    notch_freqs: List[float] = field(default_factory=lambda: [50.0, 60.0])
    notch_q: float = 50.0

    seg_sec: float = 5.0
    overlap: float = 0.5


@dataclass
class ModelConfig:
    in_ch: int = 1
    base_ch: int = 32
    use_dual_separator: bool = True

    diff_in_ch: int = 1
    diff_cond_ch: int = 2
    diff_base_ch: int = 64
    time_embed_dim: int = 128


@dataclass
class DiffusionConfig:
    num_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 3.5e-2
    infer_k: int = 10
    clip_denoised: bool = False


@dataclass
class OptimConfig:
    sep_lr: float = 1e-3
    sep_batch_size: int = 32
    sep_epochs: int = 50

    diff_lr: float = 2e-4
    diff_batch_size: int = 16
    diff_epochs: int = 100

    weight_decay: float = 0.0
    grad_clip: float = 1.0


@dataclass
class TrainConfig:
    num_workers: int = 0
    shuffle: bool = True
    seed: int = 42


@dataclass
class LossConfig:
    # separator loss
    lambda_fecg_l1: float = 1.0
    lambda_fecg_corr: float = 0.2
    lambda_mix: float = 0.5
    lambda_sep_decorr: float = 0.1

    # diffusion loss
    lambda_diff_noise: float = 1.0
    lambda_diff_x0: float = 1.0
    lambda_diff_corr: float = 0.2
    lambda_diff_qrs: float = 0.1

    # total loss balance
    alpha_total_sep: float = 1.0
    alpha_total_diff: float = 1.0


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)


cfg = Config()