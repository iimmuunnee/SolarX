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
import { Box, Heading, Text, useColorModeValue } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import type { TimeSeriesData } from '../../types/simulation';
import { formatEnergy } from '../../utils/formatters';

interface PredictionChartProps {
  data: TimeSeriesData;
  title?: string;
  height?: number;
  maxPoints?: number;
}

// Space allocated to the Y-axis column (tick labels + axis line)
const Y_AXIS_WIDTH = 65;
// Chart margins: left is small since Y-axis width handles the space
const CHART_MARGIN = { top: 5, right: 30, left: 10, bottom: 20 };

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

  // Left offset used to horizontally align axis labels with the chart area
  const labelPaddingLeft = `${CHART_MARGIN.left + Y_AXIS_WIDTH}px`;

  return (
    <Box>
      <Heading size="md" mb={4} color="white">
        {title || t('predictionChart.title')}
      </Heading>

      {/* Y-axis label — horizontal, positioned above the chart area */}
      <Text
        fontSize="xs"
        color={textColor}
        fontWeight="medium"
        pl={labelPaddingLeft}
        mb="2px"
      >
        {t('predictionChart.yAxis')}
      </Text>

      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={CHART_MARGIN}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis
            dataKey="hour"
            stroke={textColor}
          />
          <YAxis
            width={Y_AXIS_WIDTH}
            stroke={textColor}
          />
          <Tooltip
            formatter={(value: number | undefined) =>
              value !== undefined ? formatEnergy(value, 'kW') : 'N/A'
            }
            labelFormatter={(label) =>
              t('predictionChart.tooltip.hour', { hour: label })
            }
            contentStyle={{
              backgroundColor: '#000000',
              border: '1px solid #ffffff',
              borderRadius: '0',
              color: '#ffffff',
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#ffffff"
            strokeWidth={2}
            dot={false}
            name={t('predictionChart.legend.actual')}
          />
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#FFD700"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            name={t('predictionChart.legend.predicted')}
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
        {t('predictionChart.xAxis')}
      </Text>
    </Box>
  );
};
