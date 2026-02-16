/**
 * Vendor information types
 */

export interface VendorInfo {
  id: string;
  name: string;
  c_rate: number;
  efficiency: number;
  soc_range: [number, number];
  cost_per_kwh: number;
  degradation_rate: number;
  chemistry: string;
}

export interface VendorsResponse {
  vendors: VendorInfo[];
}
