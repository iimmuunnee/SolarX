/**
 * SparkLine - Mini inline chart for trend display
 */
import { Box } from '@chakra-ui/react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

interface SparkLineProps {
  data: number[];
  color?: string;
  height?: number;
  width?: string;
}

export const SparkLine = ({
  data,
  color = '#FFD700',
  height = 40,
  width = '100%',
}: SparkLineProps) => {
  // Sample to max 30 points for performance
  const step = Math.max(1, Math.floor(data.length / 30));
  const chartData = data
    .filter((_, i) => i % step === 0)
    .map((value, i) => ({ i, value }));

  return (
    <Box w={width} h={`${height}px`}>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};
