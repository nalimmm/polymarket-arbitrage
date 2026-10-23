"""Typed configuration. A missing key raises immediately instead of falling back silently."""

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class DataCfg:
    price_poll_seconds: float
    kline_poll_seconds: float
    book_poll_seconds: float
    calibration_klines: int
    max_data_age_seconds: float


@dataclass(frozen=True)
class ModelCfg:
    ewma_lambda: float
    fit_lambda: bool
    lambda_bounds: tuple
    jump_threshold_sigma: float
    vol_stress: float


@dataclass(frozen=True)
class SignalCfg:
    min_edge: float
    max_edge: float
    min_token_price: float
    max_token_price: float
    no_entry_minutes_before_expiry: float
    cooldown_seconds: float


@dataclass(frozen=True)
class SizingCfg:
    kelly_fraction: float
    max_pct_per_outcome: float
    max_total_exposure_pct: float
    depth_tolerance: float
    max_depth_share: float
    min_order_usd: float


@dataclass(frozen=True)
class ExitCfg:
    convergence_band: float
    stop_loss_pct: float
    cutoff_minutes: float


@dataclass(frozen=True)
class ExecCfg:
    mode: str
    latency_seconds: float
    fee_on_profit: float


@dataclass(frozen=True)
class LogCfg:
    trades_csv: str
    signals_csv: str


@dataclass(frozen=True)
class Config:
    capital_usd: float
    data: DataCfg
    model: ModelCfg
    signal: SignalCfg
    sizing: SizingCfg
    exits: ExitCfg
    execution: ExecCfg
    logging: LogCfg


def load_config(path: str = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        capital_usd=float(raw["capital_usd"]),
        data=DataCfg(**raw["data"]),
        model=ModelCfg(**raw["model"]),
        signal=SignalCfg(**raw["signal"]),
        sizing=SizingCfg(**raw["sizing"]),
        exits=ExitCfg(**raw["exits"]),
        execution=ExecCfg(**raw["execution"]),
        logging=LogCfg(**raw["logging"]),
    )
