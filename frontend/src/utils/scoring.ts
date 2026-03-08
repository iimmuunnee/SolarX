/**
 * Composite scoring logic for vendor comparison
 */
import type { VendorResult } from '../types/simulation';

const WEIGHTS = {
  revenue: 0.30,
  roi: 0.25,
  soh: 0.20,
  payback: 0.15,
  npv: 0.10,
};

/** Normalize a value between min and max to 0–100 */
const norm = (value: number, min: number, max: number): number => {
  if (max === min) return 100;
  return ((value - min) / (max - min)) * 100;
};

export interface VendorScore {
  vendorId: string;
  score: number;
  grade: string;
  rank: number;
}

export const getLetterGrade = (score: number): string => {
  if (score >= 90) return 'A+';
  if (score >= 80) return 'A';
  if (score >= 70) return 'B+';
  if (score >= 60) return 'B';
  if (score >= 50) return 'C+';
  return 'C';
};

export const calculateVendorScores = (vendors: VendorResult[]): VendorScore[] => {
  if (vendors.length === 0) return [];

  const revenues = vendors.map((v) => v.revenue_krw);
  const rois = vendors.map((v) => v.roi_percent);
  const sohs = vendors.map((v) => v.soh_percent);
  const paybacks = vendors.map((v) => v.payback_years);
  const npvs = vendors.map((v) => v.npv_krw);

  const scores = vendors.map((v) => {
    const revenueScore = norm(v.revenue_krw, Math.min(...revenues), Math.max(...revenues));
    const roiScore = norm(v.roi_percent, Math.min(...rois), Math.max(...rois));
    const sohScore = norm(v.soh_percent, Math.min(...sohs), Math.max(...sohs));
    // Payback: lower is better, so invert
    const paybackScore = 100 - norm(v.payback_years, Math.min(...paybacks), Math.max(...paybacks));
    const npvScore = norm(v.npv_krw, Math.min(...npvs), Math.max(...npvs));

    const composite =
      revenueScore * WEIGHTS.revenue +
      roiScore * WEIGHTS.roi +
      sohScore * WEIGHTS.soh +
      paybackScore * WEIGHTS.payback +
      npvScore * WEIGHTS.npv;

    return { vendorId: v.vendor_id, score: composite };
  });

  // Sort descending by score to assign ranks
  const sorted = [...scores].sort((a, b) => b.score - a.score);
  return sorted.map((s, i) => ({
    ...s,
    grade: getLetterGrade(s.score),
    rank: i + 1,
  }));
};
