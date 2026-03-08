/**
 * MetricCard - Label + value + optional delta + tooltip
 */
import { Box, Text, Tooltip } from '@chakra-ui/react';
import { InfoOutlineIcon } from '@chakra-ui/icons';

interface MetricCardProps {
  label: string;
  value: string;
  delta?: string;
  deltaColor?: string;
  tooltip?: string;
  accentColor?: string;
}

export const MetricCard = ({
  label,
  value,
  delta,
  deltaColor = 'battery.excellent',
  tooltip,
  accentColor,
}: MetricCardProps) => {
  return (
    <Box
      bg="spacex.darkGray"
      border="1px solid"
      borderColor="spacex.borderGray"
      p={4}
      position="relative"
      transition="all 0.3s ease"
      _hover={{ borderColor: accentColor || 'white', transform: 'translateY(-2px)' }}
    >
      {accentColor && (
        <Box position="absolute" top={0} left={0} right={0} h="3px" bg={accentColor} />
      )}
      <Text
        fontSize="xs"
        color="spacex.textGray"
        textTransform="uppercase"
        letterSpacing="wider"
        mb={1}
      >
        {label}
        {tooltip && (
          <Tooltip label={tooltip} hasArrow placement="top" maxW="300px">
            <InfoOutlineIcon ml={1} boxSize={3} cursor="help" />
          </Tooltip>
        )}
      </Text>
      <Text fontSize="2xl" fontWeight="bold" color="white" fontFamily="mono">
        {value}
      </Text>
      {delta && (
        <Text fontSize="sm" color={deltaColor} fontWeight="medium">
          {delta}
        </Text>
      )}
    </Box>
  );
};
