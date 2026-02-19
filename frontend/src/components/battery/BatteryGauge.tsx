/**
 * BatteryGauge Component - SpaceX Minimal Style
 * Simple vertical battery with clean white borders
 */
import { Box, Text, VStack } from '@chakra-ui/react';
import { motion, useReducedMotion } from 'framer-motion';

const MotionBox = motion(Box);

const getFillColor = (soc: number) => {
  if (soc >= 80) return '#10B981'; // battery.excellent
  if (soc >= 50) return '#22C55E'; // battery.good
  if (soc >= 20) return '#F59E0B'; // battery.warning
  return '#EF4444'; // battery.critical
};

export const SOC_EXPLANATION =
  '배터리는 SMP(계통한계가격)가 평균보다 낮을 때 충전하고, 높을 때 방전하는 ' +
  '차익거래 방식으로 운영됩니다. 가격 신호에 따라 충·방전이 반복되므로 SOC가 ' +
  '항상 일정 수준을 유지하지 않는 것이 정상입니다. 표시된 값은 시뮬레이션 전체 ' +
  '운영 기간의 평균 SOC입니다.';

export const SOC_LOW_THRESHOLD = 50;

interface BatteryGaugeProps {
  soc: number; // State of Charge (0-100%)
  vendor?: string;
  width?: string;
  height?: string;
}

export const BatteryGauge = ({
  soc,
  vendor,
  width = '80px',
  height = '180px',
}: BatteryGaugeProps) => {
  const shouldReduceMotion = useReducedMotion();

  // Clamp SOC between 0 and 100
  const clampedSOC = Math.max(0, Math.min(100, soc));

  // Calculate color based on SOC
  const fillColor = getFillColor(clampedSOC);

  return (
    <VStack spacing={2}>
      {/* Battery Body */}
      <Box position="relative" width={width} height={height}>
        {/* Battery Terminal (Top Nub) */}
        <Box
          position="absolute"
          top="-8px"
          left="50%"
          transform="translateX(-50%)"
          width="40%"
          height="8px"
          bg="spacex.borderGray"
          borderRadius="0"
          border="1px solid"
          borderColor="white"
        />

        {/* Battery Container */}
        <Box
          position="relative"
          width="100%"
          height="100%"
          bg="spacex.black"
          border="2px solid"
          borderColor="white"
          borderRadius="0"
          overflow="hidden"
        >
          {/* Battery Fill (Animated) */}
          <MotionBox
            position="absolute"
            bottom={0}
            left={0}
            right={0}
            bg={fillColor}
            boxShadow={`0 0 10px ${fillColor}40`}
            initial={shouldReduceMotion ? { height: `${clampedSOC}%` } : { height: 0 }}
            animate={{ height: `${clampedSOC}%` }}
            transition={{
              duration: shouldReduceMotion ? 0 : 1.2,
              ease: 'easeOut',
            }}
            opacity={0.9}
          />

          {/* SOC Percentage Text (Centered) */}
          <Box
            position="absolute"
            top="50%"
            left="50%"
            transform="translate(-50%, -50%)"
            zIndex={2}
          >
            <Text
              fontSize="xl"
              fontWeight="bold"
              color="white"
              fontFamily="mono"
              mixBlendMode="difference"
            >
              {clampedSOC.toFixed(0)}%
            </Text>
          </Box>

          {/* Horizontal Level Lines */}
          {[25, 50, 75].map((level) => (
            <Box
              key={level}
              position="absolute"
              bottom={`${level}%`}
              left={0}
              right={0}
              height="1px"
              bg="spacex.borderGray"
              opacity={0.3}
            />
          ))}
        </Box>
      </Box>

      {/* SOC Label */}
      <Text
        fontSize="xs"
        color="spacex.textGray"
        textTransform="uppercase"
        letterSpacing="wider"
        fontWeight="semibold"
      >
        SOC
      </Text>

      {/* Vendor Name (Optional) */}
      {vendor && (
        <Text
          fontSize="sm"
          color="white"
          fontWeight="bold"
          textAlign="center"
        >
          {vendor}
        </Text>
      )}
    </VStack>
  );
};

export default BatteryGauge;
