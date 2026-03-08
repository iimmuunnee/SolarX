/**
 * RadarChart - Multi-axis vendor comparison using Recharts
 */
import {
  RadarChart as RechartsRadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  Legend,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { Box } from '@chakra-ui/react';
import type { VendorResult } from '../../types/simulation';
import type { RadarDataPoint } from '../../utils/normalize';
import { normalizeForRadar } from '../../utils/normalize';
import { getVendorColor } from '../../utils/vendorColors';

interface RadarChartProps {
  vendors: VendorResult[];
  axisLabels: Record<string, string>;
  height?: number;
}

export const VendorRadarChart = ({
  vendors,
  axisLabels,
  height = 400,
}: RadarChartProps) => {
  const data: RadarDataPoint[] = normalizeForRadar(vendors, axisLabels);

  return (
    <Box>
      <ResponsiveContainer width="100%" height={height}>
        <RechartsRadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="rgba(255,255,255,0.15)" />
          <PolarAngleAxis
            dataKey="axis"
            tick={{ fill: '#e5e7eb', fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1c2536',
              border: '1px solid #4a5568',
              borderRadius: '0',
              color: '#ffffff',
            }}
          />
          {vendors.map((vendor) => {
            const color = getVendorColor(vendor.vendor_id);
            return (
              <Radar
                key={vendor.vendor_id}
                name={vendor.vendor_name}
                dataKey={vendor.vendor_id}
                stroke={color.primary}
                fill={color.primary}
                fillOpacity={0.15}
                strokeWidth={2}
              />
            );
          })}
          <Legend />
        </RechartsRadarChart>
      </ResponsiveContainer>
    </Box>
  );
};
