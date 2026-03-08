/**
 * Profit chart component - Shows cumulative profit over time for all vendors
 */
import { useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
} from 'recharts';
import { Box, Heading, Text, useColorModeValue } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import type { TimeSeriesData } from '../../types/simulation';
import { formatKRW } from '../../utils/formatters';
import { VENDOR_COLORS, BASELINE_COLOR } from '../../utils/vendorColors';

interface ProfitChartProps {
  data: TimeSeriesData;
  title?: string;
  height?: number;
}

// Space allocated to the Y-axis column (tick labels + axis line)
const Y_AXIS_WIDTH = 115;
// Chart margins: left is small since Y-axis width handles the space
const CHART_MARGIN = { top: 5, right: 30, left: 10, bottom: 20 };

export const ProfitChart = ({ data, title, height = 400 }: ProfitChartProps) => {
  const { t } = useTranslation('charts');
  const gridColor = useColorModeValue('#e2e8f0', '#4a5568');
  const textColor = useColorModeValue('#2d3748', '#e2e8f0');
  const [hiddenVendors, setHiddenVendors] = useState<Set<string>>(new Set());

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleLegendClick = useCallback((e: any) => {
    const key = String(e?.dataKey || '');
    if (!key) return;
    setHiddenVendors((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  // Create vendor array with final profits for sorting (distinct brand colors)
  const vendors = [
    {
      key: 'samsung',
      name: 'Samsung SDI',
      stroke: VENDOR_COLORS.samsung.primary,
      data: data.samsung_profit_krw,
      finalProfit: data.samsung_profit_krw?.[data.samsung_profit_krw.length - 1] || 0,
    },
    {
      key: 'lg',
      name: 'LG Energy Solution',
      stroke: VENDOR_COLORS.lg.primary,
      data: data.lg_profit_krw,
      finalProfit: data.lg_profit_krw?.[data.lg_profit_krw.length - 1] || 0,
    },
    {
      key: 'tesla',
      name: 'Tesla',
      stroke: VENDOR_COLORS.tesla.primary,
      data: data.tesla_profit_krw,
      finalProfit: data.tesla_profit_krw?.[data.tesla_profit_krw.length - 1] || 0,
    },
  ]
    .filter((v) => v.data && v.data.length > 0)
    .sort((a, b) => b.finalProfit - a.finalProfit);

  // Transform data for Recharts
  const chartData = data.hours.map((hour, index) => ({
    hour,
    lg: data.lg_profit_krw?.[index] || null,
    samsung: data.samsung_profit_krw?.[index] || null,
    tesla: data.tesla_profit_krw?.[index] || null,
    baseline: data.baseline_profit_krw[index],
  }));

  // Sample data for better performance (show every 10th point if > 500 points)
  const sampledData = chartData.length > 500
    ? chartData.filter((_, index) => index % 10 === 0)
    : chartData;

  // Calculate Y-axis domain for better zoom
  const allProfits = [
    ...(data.baseline_profit_krw || []),
    ...(data.lg_profit_krw || []),
    ...(data.samsung_profit_krw || []),
    ...(data.tesla_profit_krw || []),
  ].filter((v) => v !== null && v !== undefined && v !== 0);

  const minProfit = Math.min(...allProfits);
  const maxProfit = Math.max(...allProfits);

  const yAxisDomain = [
    Math.floor(minProfit * 0.9),
    Math.ceil(maxProfit * 1.05),
  ];

  // Left offset used to horizontally align axis labels with the chart area
  const labelPaddingLeft = `${CHART_MARGIN.left + Y_AXIS_WIDTH}px`;

  return (
    <Box>
      <Heading size="md" mb={4} color="white">
        {title || t('profitChart.title')}
      </Heading>

      {/* Y-axis label — horizontal, positioned above the chart area */}
      <Text
        fontSize="xs"
        color={textColor}
        fontWeight="medium"
        pl={labelPaddingLeft}
        mb="2px"
      >
        {t('profitChart.yAxis')}
      </Text>

      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={sampledData} margin={CHART_MARGIN}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis
            dataKey="hour"
            stroke={textColor}
            tickFormatter={(v: number) => String(Math.round(v / 24))}
          />
          <YAxis
            width={Y_AXIS_WIDTH}
            domain={yAxisDomain}
            stroke={textColor}
            tickFormatter={(value) => formatKRW(value, 0)}
          />
          <Tooltip
            formatter={(value: number | undefined, name: string | undefined) =>
              value !== undefined ? [formatKRW(value, 0), name || ''] : ['N/A', name || '']
            }
            labelFormatter={(label) =>
              t('profitChart.tooltip.hour', { day: Math.round(label / 24) })
            }
            itemSorter={(item) => -(item.value || 0)}
            contentStyle={{
              backgroundColor: '#1c2536',
              border: '1px solid #4a5568',
              borderRadius: '0',
              color: '#ffffff',
            }}
          />
          <Legend onClick={handleLegendClick} wrapperStyle={{ cursor: 'pointer' }} />
          <Line
            type="monotone"
            dataKey="baseline"
            stroke={BASELINE_COLOR}
            hide={hiddenVendors.has('baseline')}
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name={t('common:vendor.baseline')}
          />
          {vendors.map((vendor) => (
            <Line
              key={vendor.key}
              type="monotone"
              dataKey={vendor.key}
              stroke={vendor.stroke}
              strokeWidth={2}
              dot={false}
              name={vendor.name}
              hide={hiddenVendors.has(vendor.key)}
              strokeOpacity={hiddenVendors.size > 0 && !hiddenVendors.has(vendor.key) ? 1 : undefined}
            />
          ))}
          <Brush
            dataKey="hour"
            height={30}
            stroke="#4a5568"
            fill="#1c2536"
            tickFormatter={(v: number) => String(Math.round(v / 24))}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* X-axis label — centered under the chart area (excluding Y-axis column) */}
      <Text
        fontSize="xs"
        color={textColor}
        fontWeight="medium"
        textAlign="center"
        pl={labelPaddingLeft}
        pr={`${CHART_MARGIN.right}px`}
        mt="2px"
      >
        {t('profitChart.xAxis')}
      </Text>
    </Box>
  );
};
