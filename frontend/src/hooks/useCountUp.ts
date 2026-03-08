/**
 * Count-up animation hook using Framer Motion useSpring
 */
import { useSpring, useTransform, useReducedMotion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface UseCountUpOptions {
  end: number;
  duration?: number;
  decimals?: number;
  startOnMount?: boolean;
}

export const useCountUp = ({ end, duration = 1.5, decimals = 0, startOnMount = true }: UseCountUpOptions) => {
  const shouldReduceMotion = useReducedMotion();
  const [hasStarted, setHasStarted] = useState(false);

  const spring = useSpring(0, {
    duration: shouldReduceMotion ? 0 : duration * 1000,
    bounce: 0,
  });

  const display = useTransform(spring, (v) => v.toFixed(decimals));

  useEffect(() => {
    if (startOnMount && !hasStarted) {
      spring.set(end);
      setHasStarted(true);
    }
  }, [startOnMount, end, spring, hasStarted]);

  const start = () => {
    spring.set(end);
    setHasStarted(true);
  };

  return { display, start, spring };
};
