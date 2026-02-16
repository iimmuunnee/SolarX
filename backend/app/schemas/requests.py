"""Request schemas for API endpoints."""
from pydantic import BaseModel, Field, field_validator


class BenchmarkRequest(BaseModel):
    """Request parameters for benchmark simulation comparing all vendors."""

    battery_capacity_kwh: float = Field(
        default=2280,
        ge=500,
        le=10000,
        description="Battery capacity in kWh (500-10000)",
    )
    charge_threshold: float = Field(
        default=0.9,
        ge=0.5,
        le=1.5,
        description="Charge when price < avg_price * threshold (0.5-1.5)",
    )
    discharge_threshold: float = Field(
        default=1.1,
        ge=0.5,
        le=2.0,
        description="Discharge when price > avg_price * threshold (0.5-2.0)",
    )
    allow_grid_charge: bool = Field(
        default=True,
        description="Allow battery to charge from grid when profitable",
    )
    region_factor: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Solar generation scaling factor for different regions (0.1-3.0)",
    )

    @field_validator("discharge_threshold")
    @classmethod
    def validate_discharge_threshold(cls, v, info):
        """Ensure discharge threshold is greater than charge threshold."""
        if "charge_threshold" in info.data and v <= info.data["charge_threshold"]:
            raise ValueError("discharge_threshold must be greater than charge_threshold")
        return v


class CustomRequest(BenchmarkRequest):
    """Request parameters for custom simulation with specific vendor."""

    vendor_id: str = Field(
        ...,
        description="Vendor ID (lg, samsung, or tesla)",
        pattern="^(lg|samsung|tesla)$",
    )
