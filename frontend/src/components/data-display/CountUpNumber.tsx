/**
 * CountUpNumber - Animated counting number display
 */
import { motion, useSpring, useTransform, useReducedMotion, useInView } from 'framer-motion';
import { useEffect, useRef } from 'react';
import { Text, type TextProps } from '@chakra-ui/react';

interface CountUpNumberProps extends Omit<TextProps, 'children'> {
  end: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  delay?: number;
}

export const CountUpNumber = ({
  end,
  duration = 1.5,
  decimals = 0,
  prefix = '',
  suffix = '',
  delay = 0,
  ...textProps
}: CountUpNumberProps) => {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true });
  const shouldReduceMotion = useReducedMotion();

  const spring = useSpring(0, {
    duration: shouldReduceMotion ? 0 : duration * 1000,
    bounce: 0,
  });

  const display = useTransform(spring, (v) => {
    const formatted = v.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
    return `${prefix}${formatted}${suffix}`;
  });

  useEffect(() => {
    if (isInView) {
      const timeout = setTimeout(() => spring.set(end), delay * 1000);
      return () => clearTimeout(timeout);
    }
  }, [isInView, end, spring, delay]);

  return (
    <Text as="span" ref={ref} {...textProps}>
      <motion.span>{display}</motion.span>
    </Text>
  );
};
