/**
 * ChartEntrance Component
 * Framer Motion-based chart entrance animation with stagger effect
 */
import { Box, type BoxProps } from '@chakra-ui/react';
import { motion, useReducedMotion, type Transition } from 'framer-motion';

const MotionBox = motion(Box);

interface ChartEntranceProps extends BoxProps {
  children: React.ReactNode;
  delay?: number; // delay in seconds
}

export const ChartEntrance = ({
  children,
  delay = 0,
  ...props
}: ChartEntranceProps) => {
  const shouldReduceMotion = useReducedMotion();

  const transition: Transition = {
    duration: shouldReduceMotion ? 0 : 0.6,
    delay: shouldReduceMotion ? 0 : delay,
    ease: 'easeOut' as const,
  };

  return (
    <MotionBox
      initial={shouldReduceMotion ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      // @ts-ignore - Chakra/Framer Motion type conflict
      transition={transition}
      {...props}
    >
      {children}
    </MotionBox>
  );
};

export default ChartEntrance;
