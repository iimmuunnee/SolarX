/**
 * Profit chart component - Shows cumulative profit over time for all vendors
 */
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Box, Heading, useColorModeValue } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import type { TimeSeriesData } from '../../types/simulation';
import { formatKRW } from '../../utils/formatters';

interface ProfitChartProps {
  data: TimeSeriesData;
  title?: string;
  height?: number;
}

export const ProfitChart = ({ data, title, height = 400 }: ProfitChartProps) => {
  const { t } = useTranslation('charts');
  const gridColor = useColorModeValue('#e2e8f0', '#4a5568');
  const textColor = useColorModeValue('#2d3748', '#e2e8f0');

  // Create vendor array with final profits for sorting
  const vendors = [
    {
      key: 'samsung',
      name: 'Samsung SDI',
      stroke: '#3182ce', // blue
      data: data.samsung_profit_krw,
      finalProfit: data.samsung_profit_krw?.[data.samsung_profit_krw.length - 1] || 0,
    },
    {
      key: 'lg',
      name: 'LG Energy Solution',
      stroke: '#e53e3e', // red
      data: data.lg_profit_krw,
      finalProfit: data.lg_profit_krw?.[data.lg_profit_krw.length - 1] || 0,
    },
    {
      key: 'tesla',
      name: 'Tesla',
      stroke: '#38a169', // green
      data: data.tesla_profit_krw,
      finalProfit: data.tesla_profit_krw?.[data.tesla_profit_krw.length - 1] || 0,
    },
  ]
    .filter((v) => v.data && v.data.length > 0) // Only include if data exists
    .sort((a, b) => b.finalProfit - a.finalProfit); // Sort descending by final profit

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

  // Start from 90% of min to zoom in on differences
  const yAxisDomain = [
    Math.floor(minProfit * 0.9),
    Math.ceil(maxProfit * 1.05), // Add 5% padding at top
  ];

  return (
    <Box>
      <Heading size="md" mb={4}>
        {title || t('profitChart.title')}
      </Heading>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={sampledData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis
            dataKey="hour"
            label={{ value: t('profitChart.xAxis'), position: 'insideBottom', offset: -5 }}
            stroke={textColor}
          />
          <YAxis
            domain={yAxisDomain}
            label={{ value: t('profitChart.yAxis'), angle: -90, position: 'insideLeft' }}
            stroke={textColor}
            tickFormatter={(value) => formatKRW(value, 0)}
          />
          <Tooltip
            formatter={(value: number | undefined, name: string | undefined) =>
              value !== undefined ? [formatKRW(value, 0), name || ''] : ['N/A', name || '']
            }
            labelFormatter={(label) => t('profitChart.tooltip.hour', { hour: label })}
            itemSorter={(item) => -(item.value || 0)} // Sort items by value descending in tooltip
            contentStyle={{
              backgroundColor: useColorModeValue('white', 'gray.800'),
              border: `1px solid ${gridColor}`,
            }}
          />
          <Legend />
          {/* Baseline always first per user preference */}
          <Line
            type="monotone"
            dataKey="baseline"
            stroke="#718096"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name={t('common:vendor.baseline')}
          />
          {/* Vendors in descending profit order */}
          {vendors.map((vendor) => (
            <Line
              key={vendor.key}
              type="monotone"
              dataKey={vendor.key}
              stroke={vendor.stroke}
              strokeWidth={2}
              dot={false}
              name={vendor.name}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};
