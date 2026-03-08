/**
 * Hook for vendor comparison: normalization, ranking, scoring
 */
import { useMemo } from 'react';
import type { VendorResult } from '../types/simulation';
import { calculateVendorScores, type VendorScore } from '../utils/scoring';

export const useVendorComparison = (vendors: VendorResult[]) => {
  const scores = useMemo(() => calculateVendorScores(vendors), [vendors]);

  const winner = useMemo(() => {
    if (scores.length === 0) return null;
    return scores[0];
  }, [scores]);

  const getScore = (vendorId: string): VendorScore | undefined =>
    scores.find((s) => s.vendorId === vendorId);

  const getVendor = (vendorId: string): VendorResult | undefined =>
    vendors.find((v) => v.vendor_id === vendorId);

  const winnerVendor = winner ? getVendor(winner.vendorId) : null;

  return { scores, winner, winnerVendor, getScore, getVendor };
};
