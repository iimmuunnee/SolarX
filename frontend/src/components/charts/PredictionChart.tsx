/**
 * Prediction chart component - Shows actual vs predicted solar generation
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
import { formatEnergy } from '../../utils/formatters';

interface PredictionChartProps {
  data: TimeSeriesData;
  title?: string;
  height?: number;
  maxPoints?: number;
}

export const PredictionChart = ({
  data,
  title,
  height = 400,
  maxPoints = 200,
}: PredictionChartProps) => {
  const { t } = useTranslation('charts');
  const gridColor = useColorModeValue('#e2e8f0', '#4a5568');
  const textColor = useColorModeValue('#2d3748', '#e2e8f0');

  // Transform data for Recharts (limit to first maxPoints for readability)
  const chartData = data.hours.slice(0, maxPoints).map((hour, index) => ({
    hour,
    actual: data.actual_generation_kw[index],
    predicted: data.predicted_generation_kw[index],
  }));

  return (
    <Box>
      <Heading size="md" mb={4}>
        {title || t('predictionChart.title')}
      </Heading>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis
            dataKey="hour"
            label={{ value: t('predictionChart.xAxis'), position: 'insideBottom', offset: -5 }}
            stroke={textColor}
          />
          <YAxis
            label={{ value: t('predictionChart.yAxis'), angle: -90, position: 'insideLeft' }}
            stroke={textColor}
          />
          <Tooltip
            formatter={(value: number | undefined) =>
              value !== undefined ? formatEnergy(value, 'kW') : 'N/A'
            }
            labelFormatter={(label) => t('predictionChart.tooltip.hour', { hour: label })}
            contentStyle={{
              backgroundColor: useColorModeValue('white', 'gray.800'),
              border: `1px solid ${gridColor}`,
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#3182ce"
            strokeWidth={2}
            dot={false}
            name={t('predictionChart.legend.actual')}
          />
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#ed8936"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name={t('predictionChart.legend.predicted')}
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};
