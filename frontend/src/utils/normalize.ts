/**
 * Normalize vendor data for radar chart (0–1 scale)
 */
import type { VendorResult } from '../types/simulation';

export interface RadarDataPoint {
  axis: string;
  axisKey: string;
  [vendorId: string]: number | string;
}

const minMax = (values: number[]) => ({
  min: Math.min(...values),
  max: Math.max(...values),
});

const normValue = (val: number, min: number, max: number): number => {
  if (max === min) return 1;
  return (val - min) / (max - min);
};

export const normalizeForRadar = (
  vendors: VendorResult[],
  axisLabels: Record<string, string>,
): RadarDataPoint[] => {
  const revenues = minMax(vendors.map((v) => v.revenue_krw));
  const rois = minMax(vendors.map((v) => v.roi_percent));
  const sohs = minMax(vendors.map((v) => v.soh_percent));
  const paybacks = minMax(vendors.map((v) => v.payback_years));
  const npvs = minMax(vendors.map((v) => v.npv_krw));

  const axes: { key: string; getter: (v: VendorResult) => number; range: { min: number; max: number }; invert?: boolean }[] = [
    { key: 'revenue', getter: (v) => v.revenue_krw, range: revenues },
    { key: 'roi', getter: (v) => v.roi_percent, range: rois },
    { key: 'soh', getter: (v) => v.soh_percent, range: sohs },
    { key: 'payback', getter: (v) => v.payback_years, range: paybacks, invert: true },
    { key: 'npv', getter: (v) => v.npv_krw, range: npvs },
  ];

  return axes.map(({ key, getter, range, invert }) => {
    const point: RadarDataPoint = {
      axis: axisLabels[key] || key,
      axisKey: key,
    };
    vendors.forEach((v) => {
      let val = normValue(getter(v), range.min, range.max);
      if (invert) val = 1 - val;
      // Clamp to minimum 0.1 for visual clarity
      point[v.vendor_id] = Math.max(0.1, val * 0.9 + 0.1);
    });
    return point;
  });
};
