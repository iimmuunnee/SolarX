/**
 * Centralized vendor color definitions
 */
export const VENDOR_COLORS: Record<string, { primary: string; glow: string; light: string }> = {
  samsung: {
    primary: '#3B82F6',
    glow: 'rgba(59, 130, 246, 0.4)',
    light: 'rgba(59, 130, 246, 0.15)',
  },
  lg: {
    primary: '#10B981',
    glow: 'rgba(16, 185, 129, 0.4)',
    light: 'rgba(16, 185, 129, 0.15)',
  },
  tesla: {
    primary: '#EF4444',
    glow: 'rgba(239, 68, 68, 0.4)',
    light: 'rgba(239, 68, 68, 0.15)',
  },
};

export const BASELINE_COLOR = '#718096';

export const getVendorColor = (vendorId: string) =>
  VENDOR_COLORS[vendorId] ?? { primary: '#A0AEC0', glow: 'rgba(160,174,192,0.4)', light: 'rgba(160,174,192,0.15)' };
