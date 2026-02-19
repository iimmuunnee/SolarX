/**
 * CircularSOHGauge Component - SpaceX Minimal Style
 * Simple circular progress with clean white stroke
 */
import { Box, Text, VStack } from '@chakra-ui/react';
import { motion, useReducedMotion } from 'framer-motion';

interface CircularSOHGaugeProps {
  soh: number; // State of Health (0-100%)
  size?: number; // diameter in pixels
  strokeWidth?: number;
}

export const CircularSOHGauge = ({
  soh,
  size = 120,
  strokeWidth = 8,
}: CircularSOHGaugeProps) => {
  const shouldReduceMotion = useReducedMotion();

  // Clamp SOH between 0 and 100
  const clampedSOH = Math.max(0, Math.min(100, soh));

  // Calculate SVG circle properties
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clampedSOH / 100) * circumference;

  return (
    <VStack spacing={2}>
      <Box position="relative" width={`${size}px`} height={`${size}px`}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          style={{ transform: 'rotate(-90deg)' }}
        >
          {/* Background Circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth={strokeWidth}
          />

          {/* Progress Circle - Simple white stroke */}
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="#ffffff"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={
              shouldReduceMotion
                ? { strokeDashoffset: offset }
                : { strokeDashoffset: circumference }
            }
            animate={{ strokeDashoffset: offset }}
            transition={{
              duration: shouldReduceMotion ? 0 : 2,
              ease: 'easeOut',
              delay: shouldReduceMotion ? 0 : 0.5,
            }}
          />
        </svg>

        {/* Center Text */}
        <Box
          position="absolute"
          top="50%"
          left="50%"
          transform="translate(-50%, -50%)"
          textAlign="center"
        >
          <Text
            fontSize="2xl"
            fontWeight="bold"
            color="white"
            fontFamily="mono"
          >
            {clampedSOH.toFixed(1)}%
          </Text>
          <Text
            fontSize="xs"
            color="spacex.textGray"
            textTransform="uppercase"
            letterSpacing="wider"
            mt={-1}
          >
            SOH
          </Text>
        </Box>
      </Box>

      {/* Health Status Label */}
      <Text
        fontSize="xs"
        color="spacex.textGray"
        fontWeight="semibold"
        textTransform="uppercase"
        letterSpacing="wide"
      >
        {clampedSOH >= 90 && 'Excellent'}
        {clampedSOH >= 75 && clampedSOH < 90 && 'Good'}
        {clampedSOH >= 50 && clampedSOH < 75 && 'Degraded'}
        {clampedSOH < 50 && 'Critical'}
      </Text>
    </VStack>
  );
};

export default CircularSOHGauge;
