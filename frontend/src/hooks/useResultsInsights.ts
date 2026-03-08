/**
 * Auto-generate data-driven insights from benchmark results
 */
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { VendorResult, BaselineResult } from '../types/simulation';
import { formatKRW, formatPercent, formatNumber } from '../utils/formatters';

export interface Insight {
  id: string;
  text: string;
  category: 'winner' | 'comparison' | 'health' | 'economic';
}

export const useResultsInsights = (
  vendors: VendorResult[],
  baseline: BaselineResult,
): Insight[] => {
  const { i18n } = useTranslation();
  const isKo = i18n.language === 'ko';

  return useMemo(() => {
    if (vendors.length === 0) return [];

    const insights: Insight[] = [];
    const sorted = [...vendors].sort((a, b) => b.roi_percent - a.roi_percent);
    const best = sorted[0];

    // 1. Winner ROI insight
    insights.push({
      id: 'roi-winner',
      text: isKo
        ? `${best.vendor_name}가 ROI ${formatPercent(best.roi_percent, 2)}로 최고 — 낮은 CAPEX(${formatKRW(best.capex_krw, 0)}) 덕분`
        : `${best.vendor_name} leads with ${formatPercent(best.roi_percent, 2)} ROI — thanks to lower CAPEX (${formatKRW(best.capex_krw, 0)})`,
      category: 'winner',
    });

    // 2. Revenue spread insight
    const revenues = vendors.map((v) => v.revenue_krw);
    const maxRev = Math.max(...revenues);
    const minRev = Math.min(...revenues);
    const spread = ((maxRev - minRev) / maxRev) * 100;
    if (spread < 2) {
      insights.push({
        id: 'revenue-tight',
        text: isKo
          ? `세 벤더 간 수익 차이 ${spread.toFixed(1)}% 이내 — 실질적으로 동등한 성능`
          : `Revenue spread within ${spread.toFixed(1)}% across vendors — virtually equal performance`,
        category: 'comparison',
      });
    }

    // 3. SOH insight
    const avgSoh = vendors.reduce((s, v) => s + v.soh_percent, 0) / vendors.length;
    if (avgSoh > 99) {
      insights.push({
        id: 'soh-excellent',
        text: isKo
          ? `SOH ${formatPercent(avgSoh, 1)}+ — 시뮬레이션에서 배터리 열화 무시 가능`
          : `SOH ${formatPercent(avgSoh, 1)}+ — negligible battery degradation in simulation`,
        category: 'health',
      });
    }

    // 4. Fastest payback
    const fastestPayback = [...vendors].sort((a, b) => a.payback_years - b.payback_years)[0];
    insights.push({
      id: 'payback-fastest',
      text: isKo
        ? `${fastestPayback.vendor_name}은 회수기간 ${formatNumber(fastestPayback.payback_years, 2)}년으로 가장 빠른 투자 회수`
        : `${fastestPayback.vendor_name} has fastest payback at ${formatNumber(fastestPayback.payback_years, 2)} years`,
      category: 'economic',
    });

    // 5. ESS vs baseline
    const bestRevenue = Math.max(...revenues);
    const delta = bestRevenue - baseline.revenue_krw;
    if (delta > 0) {
      insights.push({
        id: 'ess-advantage',
        text: isKo
          ? `ESS 도입 시 기준선 대비 +${formatKRW(delta, 0)} 추가 수익 발생`
          : `ESS adds +${formatKRW(delta, 0)} revenue over baseline (no ESS)`,
        category: 'economic',
      });
    }

    return insights;
  }, [vendors, baseline, isKo]);
};
