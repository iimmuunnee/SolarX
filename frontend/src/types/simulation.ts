/**
 * Simulation request and response types
 */

export interface PredictionMetrics {
  mae_kw: number;
  rmse_kw: number;
  mape_percent: number;
}

export interface TimeSeriesData {
  hours: number[];
  actual_generation_kw: number[];
  predicted_generation_kw: number[];
  lg_profit_krw?: number[];
  samsung_profit_krw?: number[];
  tesla_profit_krw?: number[];
  baseline_profit_krw: number[];
}

export interface SimulationMetadata {
  duration_hours: number;
  simulation_years: number;
  avg_smp_price: number;
  data_points: number;
}

export interface VendorResult {
  vendor_id: string;
  vendor_name: string;
  revenue_krw: number;
  soh_percent: number;
  cycle_count: number;
  throughput_kwh: number;
  capex_krw: number;
  opex_annual_krw: number;
  roi_percent: number;
  payback_years: number;
  npv_krw: number;
}

export interface BaselineResult {
  revenue_krw: number;
}

export interface BenchmarkRequest {
  battery_capacity_kwh: number;
  charge_threshold: number;
  discharge_threshold: number;
  allow_grid_charge: boolean;
  region_factor: number;
}

export interface BenchmarkResponse {
  simulation_id: string;
  metadata: SimulationMetadata;
  prediction_metrics: PredictionMetrics;
  vendors: VendorResult[];
  baseline: BaselineResult;
  time_series: TimeSeriesData;
}

export interface CustomRequest extends BenchmarkRequest {
  vendor_id: string;
}

export interface CustomResponse {
  simulation_id: string;
  metadata: SimulationMetadata;
  prediction_metrics: PredictionMetrics;
  vendor_result: VendorResult;
  baseline: BaselineResult;
  time_series: TimeSeriesData;
}

export interface ScenarioResult {
  name: string;
  factor: number;
  vendor_id: string;
  revenue_krw: number;
  soh_percent: number;
  time_series: TimeSeriesData;
}

export interface ScalabilityResponse {
  scenarios: ScenarioResult[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  data_loaded: boolean;
}
