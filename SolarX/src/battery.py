"""Battery physics models with SOH tracking and temperature effects."""
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger("solarx.battery")


class ESSBattery:
    """Energy Storage System Battery with physics constraints."""

    def __init__(
        self,
        name: str,
        capacity: float,
        c_rate: float,
        eff: float,
        soc_range: Tuple[float, float],
        eff_is_roundtrip: bool = True
    ) -> None:
        """
        Initialize battery with vendor specifications.

        Args:
            name: Battery model name
            capacity: Nominal capacity (kWh)
            c_rate: Maximum charge/discharge rate (C-rate)
            eff: Efficiency (round-trip or one-way)
            soc_range: (soc_min, soc_max) as fractions
            eff_is_roundtrip: If True, eff is round-trip and will be split
        """
        self.name = name
        self.capacity = capacity
        self.max_power = capacity * c_rate

        # Efficiency conversion
        if eff_is_roundtrip:
            one_way = float(np.sqrt(eff))
            self.charge_eff = one_way
            self.discharge_eff = one_way
        else:
            self.charge_eff = eff
            self.discharge_eff = eff

        self.soc_min, self.soc_max = soc_range
        self.current_kwh = capacity * self.soc_min

        # SOH (State of Health) tracking
        self.soh: float = 1.0  # 1.0 = 100%
        self.cycle_count: float = 0.0  # Equivalent full cycles
        self.total_throughput_kwh: float = 0.0  # Total energy throughput

        # Degradation parameters (can be overridden by subclasses)
        self.degradation_rate: float = 0.0001  # Per cycle
        self.calendar_aging_rate: float = 0.00005  # Per day

        # Temperature parameters
        self.reference_temp_c: float = 25.0  # Reference temperature
        self.temp_coefficient: float = 0.005  # Efficiency change per °C

    def validate_params(self, power_kw: float, dt_hours: float) -> None:
        """
        Validate battery operation parameters.

        Args:
            power_kw: Requested power (kW)
            dt_hours: Time step (hours)

        Raises:
            ValueError: If parameters are invalid
        """
        if power_kw < 0:
            raise ValueError(f"power_kw must be non-negative, got {power_kw}")
        if dt_hours <= 0:
            raise ValueError(f"dt_hours must be positive, got {dt_hours}")

    def temperature_efficiency_factor(self, temp_c: float) -> float:
        """
        Calculate efficiency adjustment based on temperature.

        Args:
            temp_c: Ambient temperature (°C)

        Returns:
            Efficiency multiplier (e.g., 0.95 = 5% reduction)
        """
        temp_delta = temp_c - self.reference_temp_c

        # Efficiency decreases in cold, slightly decreases in extreme heat
        if temp_c < 0:
            # Below freezing: significant efficiency loss
            factor = 1.0 - 0.15 * (abs(temp_delta) / 25.0)
        elif temp_c > 35:
            # High heat: moderate efficiency loss
            factor = 1.0 - 0.05 * ((temp_c - 35) / 10.0)
        else:
            # Normal range: minimal impact
            factor = 1.0 - abs(temp_delta) * self.temp_coefficient

        return max(0.7, min(1.0, factor))  # Clamp to [0.7, 1.0]

    def update_soh(self, energy_kwh: float, dt_hours: float) -> None:
        """
        Update State of Health based on cycle counting.

        Args:
            energy_kwh: Energy charged or discharged
            dt_hours: Time step
        """
        # Cycle counting (simple method)
        cycle_fraction = energy_kwh / self.capacity
        self.cycle_count += cycle_fraction
        self.total_throughput_kwh += energy_kwh

        # Cycle degradation
        cycle_degradation = self.cycle_count * self.degradation_rate

        # Calendar aging
        days_elapsed = dt_hours / 24.0
        calendar_degradation = days_elapsed * self.calendar_aging_rate

        # Update SOH (cannot go below 0.7 = 70%)
        self.soh = max(0.7, 1.0 - cycle_degradation - calendar_degradation)

    def effective_capacity(self) -> float:
        """Get current effective capacity accounting for SOH."""
        return self.capacity * self.soh

    def update(
        self,
        action: int,
        amount_kw: float,
        dt_hours: float = 1.0,
        temp_c: float = 25.0
    ) -> float:
        """
        Update battery state with charge/discharge action.

        Args:
            action: 1 (charge), -1 (discharge), 0 (idle)
            amount_kw: Requested power (kW)
            dt_hours: Time step (hours)
            temp_c: Ambient temperature (°C)

        Returns:
            Grid trade power (+: discharge/sell, -: charge/buy)
        """
        self.validate_params(amount_kw, dt_hours)

        amount_kw = min(amount_kw, self.max_power)
        actual_trade = 0.0

        # Temperature efficiency adjustment
        temp_factor = self.temperature_efficiency_factor(temp_c)

        if action == 1:  # Charge
            adjusted_charge_eff = self.charge_eff * temp_factor

            max_storable = (self.capacity * self.soc_max) - self.current_kwh
            real_in_kwh = min(amount_kw * dt_hours, max_storable)
            self.current_kwh += real_in_kwh * adjusted_charge_eff
            actual_trade = -(real_in_kwh / dt_hours)

            # Update SOH
            self.update_soh(real_in_kwh, dt_hours)

        elif action == -1:  # Discharge
            adjusted_discharge_eff = self.discharge_eff * temp_factor

            max_outable = self.current_kwh - (self.capacity * self.soc_min)
            real_out_kwh = min(max_outable, amount_kw * dt_hours)
            self.current_kwh -= real_out_kwh
            grid_out_kw = (real_out_kwh * adjusted_discharge_eff) / dt_hours
            actual_trade = grid_out_kw

            # Update SOH
            self.update_soh(real_out_kwh, dt_hours)

        return actual_trade

    def get_soc(self) -> float:
        """Get current State of Charge as fraction."""
        return self.current_kwh / self.capacity

    def get_status(self) -> dict:
        """Get comprehensive battery status."""
        return {
            "name": self.name,
            "current_kwh": self.current_kwh,
            "capacity_kwh": self.capacity,
            "soc": self.get_soc(),
            "soh": self.soh,
            "cycle_count": self.cycle_count,
            "total_throughput_kwh": self.total_throughput_kwh
        }


class SamsungSDI(ESSBattery):
    """Samsung SDI NCA battery."""

    def __init__(self, capacity: float) -> None:
        super().__init__(
            name="Samsung SDI (NCA)",
            capacity=capacity,
            c_rate=1.8,
            eff=0.985,
            soc_range=(0.05, 0.95),
            eff_is_roundtrip=True,
        )
        # Samsung has moderate degradation
        self.degradation_rate = 0.00009
        self.calendar_aging_rate = 0.000045


class LGEnergySolution(ESSBattery):
    """LG Energy Solution NCM battery."""

    def __init__(self, capacity: float) -> None:
        super().__init__(
            name="LG Energy Solution (NCM)",
            capacity=capacity,
            c_rate=2.0,
            eff=0.980,
            soc_range=(0.05, 0.95),
            eff_is_roundtrip=True,
        )
        # LG has low degradation (best longevity)
        self.degradation_rate = 0.00008
        self.calendar_aging_rate = 0.00004


class TeslaBattery(ESSBattery):
    """Tesla 4680 in-house battery."""

    def __init__(self, capacity: float) -> None:
        super().__init__(
            name="Tesla In-house (4680)",
            capacity=capacity,
            c_rate=1.5,
            eff=0.970,
            soc_range=(0.10, 0.90),
            eff_is_roundtrip=True,
        )
        # Tesla has slightly higher degradation
        self.degradation_rate = 0.0001
        self.calendar_aging_rate = 0.00005
