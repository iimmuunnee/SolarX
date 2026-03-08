/**
 * ScoreGauge - Circular progress for composite score (SVG)
 */
import { Box, Text, VStack } from '@chakra-ui/react';
import { motion, useReducedMotion } from 'framer-motion';

interface ScoreGaugeProps {
  score: number; // 0-100
  grade: string;
  size?: number;
  strokeWidth?: number;
  color?: string;
}

export const ScoreGauge = ({
  score,
  grade,
  size = 100,
  strokeWidth = 6,
  color = '#FFD700',
}: ScoreGaugeProps) => {
  const shouldReduceMotion = useReducedMotion();
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (Math.min(100, Math.max(0, score)) / 100) * circumference;

  return (
    <VStack spacing={0}>
      <Box position="relative" width={`${size}px`} height={`${size}px`}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={strokeWidth}
          />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={shouldReduceMotion ? { strokeDashoffset: offset } : { strokeDashoffset: circumference }}
            whileInView={{ strokeDashoffset: offset }}
            viewport={{ once: true }}
            // @ts-ignore
            transition={{ duration: shouldReduceMotion ? 0 : 1.5, ease: 'easeOut', delay: 0.3 }}
          />
        </svg>
        <Box position="absolute" top="50%" left="50%" transform="translate(-50%, -50%)" textAlign="center">
          <Text fontSize="xl" fontWeight="bold" color="white" fontFamily="mono" lineHeight="1">
            {grade}
          </Text>
          <Text fontSize="xs" color="spacex.textGray" mt={0.5}>
            {score.toFixed(0)}
          </Text>
        </Box>
      </Box>
    </VStack>
  );
};
