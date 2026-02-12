"""Economic analysis including CAPEX and ROI calculations."""
from dataclasses import dataclass
from typing import Dict
import logging

logger = logging.getLogger("solarx.economics")


@dataclass
class BatteryCost:
    """Battery CAPEX model."""
    cost_per_kwh: float      # $/kWh
    installation_ratio: float = 0.15  # Installation adds 15% to hardware cost
    ohm_cost_per_year: float = 0.0    # Annual O&M cost

    def total_capex(self, capacity_kwh: float) -> float:
        """Calculate total initial investment."""
        hardware_cost = capacity_kwh * self.cost_per_kwh
        installation_cost = hardware_cost * self.installation_ratio
        return hardware_cost + installation_cost


# Vendor costs (2026 estimates)
VENDOR_COSTS = {
    "LG Energy Solution (NCM)": BatteryCost(cost_per_kwh=180.0, ohm_cost_per_year=5000.0),
    "Samsung SDI (NCA)": BatteryCost(cost_per_kwh=175.0, ohm_cost_per_year=4800.0),
    "Tesla In-house (4680)": BatteryCost(cost_per_kwh=200.0, ohm_cost_per_year=6000.0)
}


def calculate_roi(
    total_revenue: float,
    capex: float,
    opex_annual: float,
    years: int
) -> Dict[str, float]:
    """
    Calculate ROI metrics.

    Args:
        total_revenue: Total revenue over simulation period
        capex: Initial investment
        opex_annual: Annual O&M cost
        years: Number of years in simulation

    Returns:
        Dictionary with roi, payback_period, npv
    """
    total_opex = opex_annual * years
    net_profit = total_revenue - capex - total_opex
    roi_percent = (net_profit / capex) * 100 if capex > 0 else 0.0

    # Simple payback period (years)
    annual_net_revenue = total_revenue / years - opex_annual
    payback_period = capex / annual_net_revenue if annual_net_revenue > 0 else float('inf')

    # NPV with 5% discount rate
    discount_rate = 0.05
    npv = -capex
    for year in range(1, years + 1):
        npv += annual_net_revenue / ((1 + discount_rate) ** year)

    return {
        "roi_percent": roi_percent,
        "payback_period_years": payback_period,
        "npv": npv,
        "net_profit": net_profit
    }
