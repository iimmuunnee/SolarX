/**
 * ConfidenceIndicator - Color bar showing metric confidence level
 */
import { HStack, Box, Text } from '@chakra-ui/react';

interface ConfidenceIndicatorProps {
  value: number;
  thresholds: { good: number; fair: number }; // below good = green, below fair = yellow, else red
  invert?: boolean; // if true, lower is better
}

const getConfidenceColor = (
  value: number,
  thresholds: { good: number; fair: number },
  invert: boolean,
) => {
  const isGood = invert ? value <= thresholds.good : value >= thresholds.good;
  const isFair = invert ? value <= thresholds.fair : value >= thresholds.fair;

  if (isGood) return { color: '#22C55E', label: 'Good' };
  if (isFair) return { color: '#F59E0B', label: 'Fair' };
  return { color: '#EF4444', label: 'Low' };
};

export const ConfidenceIndicator = ({
  value,
  thresholds,
  invert = false,
}: ConfidenceIndicatorProps) => {
  const { color, label } = getConfidenceColor(value, thresholds, invert);

  return (
    <HStack spacing={2}>
      <Box w="40px" h="4px" bg={color} borderRadius="full" />
      <Text fontSize="xs" color={color} fontWeight="medium">
        {label}
      </Text>
    </HStack>
  );
};
