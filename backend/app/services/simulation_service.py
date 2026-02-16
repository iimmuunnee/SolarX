"""Simulation service that wraps existing SolarX code."""
import numpy as np
import os
import sys
import pandas as pd
from typing import Dict, List, Tuple, Optional
import uuid
from functools import lru_cache
import hashlib
import json

# Add root directory to path (where src/, data/, config.py are located)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT_DIR)

from src.battery import SamsungSDI, LGEnergySolution, TeslaBattery, ESSBattery
from src.data_loader import SolarDataManager
from src.model import LSTMPredictor
from src.economics import VENDOR_COSTS, calculate_roi
from config import Config

from ..utils.serializers import numpy_to_list, ensure_finite
from ..schemas.requests import BenchmarkRequest, CustomRequest
from ..schemas.responses import (
    BenchmarkResponse,
    CustomResponse,
    VendorResult,
    BaselineResult,
    VendorInfo,
)
from ..schemas.common import PredictionMetrics, TimeSeriesData, SimulationMetadata


class SimulationService:
    """Service for running battery optimization simulations."""

    def __init__(self):
        """Initialize the simulation service."""
        self.config = Config()
        self.loader = None
        self.predictor = None
        self.test_x = None
        self.test_y = None
        self.test_smp = None
        self.test_df = None
        self.X_test = None
        self.y_test = None
        self.y_real_kw = None
        self.y_pred_kw = None
        self.real_prices = None
        self.seq_length = self.config.model.seq_length
        self.dt_hours = 1.0
        self._initialized = False

    def initialize(self):
        """Load model and data (call once on startup)."""
        if self._initialized:
            return

        # Load data
        self.loader = SolarDataManager()
        data_dir = os.path.join(ROOT_DIR, "data")
        _, _, self.test_x, self.test_y, self.test_smp = self.loader.load_and_split_standard(data_dir)

        # Create sequences
        self.X_test, self.y_test = self.loader.create_sequences(
            self.test_x, self.test_y, seq_length=self.seq_length
        )

        # Reconstruct test DataFrame for temperature
        self.test_df = pd.DataFrame(
            self.loader.scaler_x.inverse_transform(self.test_x),
            columns=["기온(℃)", "강수량(mm)", "풍속(m/s)", "습도(%)", "일조(hr)", "일사(MJ/m2)", "운량(10분위)", "발전량"]
        )

        # Load LSTM model
        model_path = os.path.join(ROOT_DIR, "src", "lstm_solar_model.pth")
        self.predictor = LSTMPredictor(
            model_path,
            input_size=self.config.model.input_size,
            hidden_size=self.config.model.hidden_size,
            num_layers=self.config.model.num_layers,
            dropout=0.0
        )

        # Run prediction once
        y_pred_scaled = self.predictor.predict(self.X_test)
        y_real_raw = self.loader.inverse_transform_y(self.y_test.reshape(-1, 1)).flatten()
        y_pred_raw = self.loader.inverse_transform_y(y_pred_scaled.reshape(-1, 1)).flatten()

        self.y_real_kw = np.maximum(y_real_raw / 1000.0, 0)
        self.y_pred_kw = np.maximum(y_pred_raw / 1000.0, 0)

        # Prepare SMP prices
        self.real_prices = self.test_smp[self.seq_length:] if self.test_smp is not None else None
        if self.real_prices is None or np.sum(self.real_prices) == 0:
            self.real_prices = np.array([100 if 10 <= i % 24 <= 16 else 200 for i in range(len(self.y_real_kw))])

        self._initialized = True

    @staticmethod
    def _get_cache_key(params: Dict) -> str:
        """Generate cache key from parameters."""
        # Convert params to sorted JSON string for consistent hashing
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()

    def _get_vendor_battery(self, vendor_id: str, capacity_kwh: float) -> ESSBattery:
        """Create battery instance for given vendor."""
        vendor_map = {
            "lg": LGEnergySolution,
            "samsung": SamsungSDI,
            "tesla": TeslaBattery,
        }
        battery_class = vendor_map.get(vendor_id)
        if not battery_class:
            raise ValueError(f"Unknown vendor: {vendor_id}")
        return battery_class(capacity_kwh)

    def _calculate_prediction_metrics(self) -> PredictionMetrics:
        """Calculate prediction quality metrics."""
        mae = float(np.mean(np.abs(self.y_real_kw - self.y_pred_kw)))
        rmse = float(np.sqrt(np.mean((self.y_real_kw - self.y_pred_kw) ** 2)))

        mape_mask = self.y_real_kw > 1e-6
        if np.any(mape_mask):
            mape = float(
                np.mean(np.abs((self.y_real_kw[mape_mask] - self.y_pred_kw[mape_mask]) / self.y_real_kw[mape_mask]))
                * 100.0
            )
        else:
            mape = 0.0

        return PredictionMetrics(mae_kw=mae, rmse_kw=rmse, mape_percent=mape)

    def _simulate_battery(
        self,
        battery: ESSBattery,
        y_real_kw: np.ndarray,
        y_pred_kw: np.ndarray,
        real_prices: np.ndarray,
        charge_threshold: float,
        discharge_threshold: float,
        allow_grid_charge: bool,
    ) -> Tuple[float, List[float]]:
        """Run simulation for a single battery.

        Returns:
            Tuple of (total_profit, profit_history)
        """
        avg_price = np.mean(real_prices)
        profit = 0.0
        history = []

        for t in range(len(y_real_kw)):
            gen_kw = y_real_kw[t]
            pred_kw = y_pred_kw[t]
            price = real_prices[t]

            # Get temperature
            temp_c = self.test_df.iloc[self.seq_length + t]["기온(℃)"] if self.seq_length + t < len(self.test_df) else 25.0

            # Decision logic
            action = 0
            if price > avg_price * discharge_threshold:
                action = -1
            elif price < avg_price * charge_threshold and pred_kw > self.config.simulation.min_generation_kw:
                action = 1

            # Execute battery action
            if action == 1:  # Charge
                charge_request_kw = battery.max_power if allow_grid_charge else min(gen_kw, battery.max_power)
                trade_kw = gen_kw + battery.update(action, charge_request_kw, self.dt_hours, temp_c=temp_c)
            elif action == -1:  # Discharge
                discharge_request_kw = battery.max_power
                trade_kw = gen_kw + battery.update(action, discharge_request_kw, self.dt_hours, temp_c=temp_c)
            else:  # Bypass
                trade_kw = gen_kw

            profit += trade_kw * self.dt_hours * price
            history.append(profit)

        return profit, history

    def _create_vendor_result(
        self,
        battery: ESSBattery,
        profit: float,
        capacity_kwh: float,
        simulation_years: float,
    ) -> VendorResult:
        """Create VendorResult from battery simulation."""
        status = battery.get_status()
        cost_model = VENDOR_COSTS[battery.name]
        capex = cost_model.total_capex(capacity_kwh)

        # Convert USD to KRW (approximate rate: 1 USD = 1320 KRW)
        usd_to_krw = 1320

        roi_metrics = calculate_roi(
            total_revenue=profit,
            capex=capex,
            opex_annual=cost_model.ohm_cost_per_year,
            years=max(simulation_years, 1)
        )

        # Map vendor names to IDs
        vendor_id_map = {
            "LG Energy Solution (NCM)": "lg",
            "Samsung SDI (NCA)": "samsung",
            "Tesla In-house (4680)": "tesla",
        }

        return VendorResult(
            vendor_id=vendor_id_map.get(battery.name, battery.name.lower()),
            vendor_name=battery.name,
            revenue_krw=profit,
            soh_percent=status['soh'] * 100,
            cycle_count=status['cycle_count'],
            throughput_kwh=status['total_throughput_kwh'],
            capex_krw=capex * usd_to_krw,
            opex_annual_krw=cost_model.ohm_cost_per_year * usd_to_krw,
            roi_percent=roi_metrics['roi_percent'],
            payback_years=roi_metrics['payback_period_years'],
            npv_krw=roi_metrics['npv'] * usd_to_krw,
        )

    def run_benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """Run benchmark simulation comparing all vendors."""
        if not self._initialized:
            self.initialize()

        # Scale generation by region factor
        y_real_scaled = self.y_real_kw * request.region_factor

        # Create batteries
        battery_capacity_kwh = request.battery_capacity_kwh
        batteries = [
            LGEnergySolution(battery_capacity_kwh),
            SamsungSDI(battery_capacity_kwh),
            TeslaBattery(battery_capacity_kwh),
        ]

        # Calculate baseline (no ESS)
        base_profit = 0.0
        baseline_history = []
        for t in range(len(y_real_scaled)):
            base_profit += y_real_scaled[t] * self.dt_hours * self.real_prices[t]
            baseline_history.append(base_profit)

        # Simulate each vendor
        vendor_results = []
        all_histories = {}

        for batt in batteries:
            profit, history = self._simulate_battery(
                battery=batt,
                y_real_kw=y_real_scaled,
                y_pred_kw=self.y_pred_kw * request.region_factor,
                real_prices=self.real_prices,
                charge_threshold=request.charge_threshold,
                discharge_threshold=request.discharge_threshold,
                allow_grid_charge=request.allow_grid_charge,
            )

            simulation_years = len(y_real_scaled) / (24 * 365)
            vendor_result = self._create_vendor_result(batt, profit, battery_capacity_kwh, simulation_years)
            vendor_results.append(vendor_result)
            all_histories[vendor_result.vendor_id] = history

        # Create time series data
        hours = list(range(len(y_real_scaled)))
        time_series = TimeSeriesData(
            hours=hours,
            actual_generation_kw=ensure_finite(y_real_scaled.tolist()),
            predicted_generation_kw=ensure_finite((self.y_pred_kw * request.region_factor).tolist()),
            lg_profit_krw=ensure_finite(all_histories.get("lg", [])),
            samsung_profit_krw=ensure_finite(all_histories.get("samsung", [])),
            tesla_profit_krw=ensure_finite(all_histories.get("tesla", [])),
            baseline_profit_krw=ensure_finite(baseline_history),
        )

        # Create metadata
        metadata = SimulationMetadata(
            duration_hours=len(y_real_scaled),
            simulation_years=len(y_real_scaled) / (24 * 365),
            avg_smp_price=float(np.mean(self.real_prices)),
            data_points=len(y_real_scaled),
        )

        return BenchmarkResponse(
            simulation_id=str(uuid.uuid4()),
            metadata=metadata,
            prediction_metrics=self._calculate_prediction_metrics(),
            vendors=vendor_results,
            baseline=BaselineResult(revenue_krw=base_profit),
            time_series=time_series,
        )

    def run_custom(self, request: CustomRequest) -> CustomResponse:
        """Run custom simulation with specific vendor."""
        if not self._initialized:
            self.initialize()

        # Scale generation by region factor
        y_real_scaled = self.y_real_kw * request.region_factor

        # Create battery
        battery = self._get_vendor_battery(request.vendor_id, request.battery_capacity_kwh)

        # Calculate baseline
        base_profit = 0.0
        baseline_history = []
        for t in range(len(y_real_scaled)):
            base_profit += y_real_scaled[t] * self.dt_hours * self.real_prices[t]
            baseline_history.append(base_profit)

        # Simulate
        profit, history = self._simulate_battery(
            battery=battery,
            y_real_kw=y_real_scaled,
            y_pred_kw=self.y_pred_kw * request.region_factor,
            real_prices=self.real_prices,
            charge_threshold=request.charge_threshold,
            discharge_threshold=request.discharge_threshold,
            allow_grid_charge=request.allow_grid_charge,
        )

        simulation_years = len(y_real_scaled) / (24 * 365)
        vendor_result = self._create_vendor_result(battery, profit, request.battery_capacity_kwh, simulation_years)

        # Create time series data (only single vendor)
        hours = list(range(len(y_real_scaled)))
        time_series = TimeSeriesData(
            hours=hours,
            actual_generation_kw=ensure_finite(y_real_scaled.tolist()),
            predicted_generation_kw=ensure_finite((self.y_pred_kw * request.region_factor).tolist()),
            baseline_profit_krw=ensure_finite(baseline_history),
        )
        # Add vendor-specific profit to appropriate field
        if request.vendor_id == "lg":
            time_series.lg_profit_krw = ensure_finite(history)
        elif request.vendor_id == "samsung":
            time_series.samsung_profit_krw = ensure_finite(history)
        elif request.vendor_id == "tesla":
            time_series.tesla_profit_krw = ensure_finite(history)

        metadata = SimulationMetadata(
            duration_hours=len(y_real_scaled),
            simulation_years=simulation_years,
            avg_smp_price=float(np.mean(self.real_prices)),
            data_points=len(y_real_scaled),
        )

        return CustomResponse(
            simulation_id=str(uuid.uuid4()),
            metadata=metadata,
            prediction_metrics=self._calculate_prediction_metrics(),
            vendor_result=vendor_result,
            baseline=BaselineResult(revenue_krw=base_profit),
            time_series=time_series,
        )

    @staticmethod
    def get_vendor_info() -> List[VendorInfo]:
        """Get information about all available vendors."""
        vendors = [
            VendorInfo(
                id="lg",
                name="LG Energy Solution (NCM)",
                c_rate=2.0,
                efficiency=0.98,
                soc_range=[0.05, 0.95],
                cost_per_kwh=180.0,
                degradation_rate=0.00008,
                chemistry="NCM (Nickel-Cobalt-Manganese)",
            ),
            VendorInfo(
                id="samsung",
                name="Samsung SDI (NCA)",
                c_rate=1.8,
                efficiency=0.985,
                soc_range=[0.05, 0.95],
                cost_per_kwh=175.0,
                degradation_rate=0.00009,
                chemistry="NCA (Nickel-Cobalt-Aluminum)",
            ),
            VendorInfo(
                id="tesla",
                name="Tesla In-house (4680)",
                c_rate=1.5,
                efficiency=0.97,
                soc_range=[0.10, 0.90],
                cost_per_kwh=200.0,
                degradation_rate=0.0001,
                chemistry="4680 Cell (High Nickel)",
            ),
        ]
        return vendors
