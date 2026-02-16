/**
 * Utility functions for formatting numbers and currency
 */

/**
 * Format number with thousands separator
 */
export const formatNumber = (value: number, decimals: number = 0): string => {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

/**
 * Format currency in KRW
 */
export const formatKRW = (value: number, decimals: number = 0): string => {
  return `₩${formatNumber(value, decimals)}`;
};

/**
 * Format currency in USD
 */
export const formatUSD = (value: number, decimals: number = 0): string => {
  return `$${formatNumber(value, decimals)}`;
};

/**
 * Format percentage
 */
export const formatPercent = (value: number, decimals: number = 1): string => {
  return `${value.toFixed(decimals)}%`;
};

/**
 * Format large numbers with K/M/B suffixes
 */
export const formatCompact = (value: number): string => {
  if (value >= 1e9) {
    return `${(value / 1e9).toFixed(1)}B`;
  }
  if (value >= 1e6) {
    return `${(value / 1e6).toFixed(1)}M`;
  }
  if (value >= 1e3) {
    return `${(value / 1e3).toFixed(1)}K`;
  }
  return value.toFixed(0);
};

/**
 * Format energy values
 */
export const formatEnergy = (value: number, unit: 'kW' | 'kWh' = 'kW'): string => {
  return `${formatNumber(value, 2)} ${unit}`;
};
