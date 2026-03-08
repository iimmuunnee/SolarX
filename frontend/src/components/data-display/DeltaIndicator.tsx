/**
 * DeltaIndicator - Shows change amount with arrow
 */
import { HStack, Text } from '@chakra-ui/react';
import { TriangleUpIcon, TriangleDownIcon } from '@chakra-ui/icons';

interface DeltaIndicatorProps {
  value: number;
  format?: (v: number) => string;
  invert?: boolean; // if true, negative is good
}

export const DeltaIndicator = ({
  value,
  format = (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`,
  invert = false,
}: DeltaIndicatorProps) => {
  const isPositive = value >= 0;
  const isGood = invert ? !isPositive : isPositive;

  return (
    <HStack spacing={1}>
      {isPositive ? (
        <TriangleUpIcon boxSize={3} color={isGood ? 'battery.excellent' : 'battery.critical'} />
      ) : (
        <TriangleDownIcon boxSize={3} color={isGood ? 'battery.excellent' : 'battery.critical'} />
      )}
      <Text
        fontSize="sm"
        fontWeight="semibold"
        color={isGood ? 'battery.excellent' : 'battery.critical'}
      >
        {format(value)}
      </Text>
    </HStack>
  );
};
