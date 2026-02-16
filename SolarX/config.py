"""Configuration management for SolarX project."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class ModelConfig:
    """LSTM model hyperparameters."""
    input_size: int = 8
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 64
    seq_length: int = 24


@dataclass
class BatteryConfig:
    """Battery vendor specifications."""
    # Format: (C-rate, round_trip_efficiency, soc_min, soc_max)
    vendors: Dict[str, Tuple[float, float, float, float]] = None

    def __post_init__(self):
        if self.vendors is None:
            self.vendors = {
                "LG": (2.0, 0.98, 0.05, 0.95),
                "Samsung": (1.8, 0.985, 0.05, 0.95),
                "Tesla": (1.5, 0.97, 0.10, 0.90)
            }


@dataclass
class SimulationConfig:
    """Simulation and decision logic parameters."""
    charge_threshold: float = 0.9   # price < avg * 0.9
    discharge_threshold: float = 1.1 # price > avg * 1.1
    min_generation_kw: float = 0.1
    allow_grid_charge: bool = True


@dataclass
class PathConfig:
    """File paths for data and models."""
    base_dir: Path = Path(__file__).parent
    data_dir: Path = None
    model_path: Path = None
    images_dir: Path = None

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.model_path is None:
            self.model_path = self.base_dir / "src" / "lstm_solar_model.pth"
        if self.images_dir is None:
            self.images_dir = self.base_dir / "images"


class Config:
    """Master configuration container."""
    def __init__(self):
        self.model = ModelConfig()
        self.battery = BatteryConfig()
        self.simulation = SimulationConfig()
        self.paths = PathConfig()
