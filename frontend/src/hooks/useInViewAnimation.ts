/**
 * IntersectionObserver + whileInView wrapper for scroll-triggered animations
 */
import { useReducedMotion, type Variants } from 'framer-motion';

interface UseInViewAnimationOptions {
  y?: number;
  x?: number;
  delay?: number;
  duration?: number;
  once?: boolean;
}

export const useInViewAnimation = ({
  y = 40,
  x = 0,
  delay = 0,
  duration = 0.6,
  once = true,
}: UseInViewAnimationOptions = {}) => {
  const shouldReduceMotion = useReducedMotion();

  const variants: Variants = {
    hidden: shouldReduceMotion
      ? { opacity: 1, y: 0, x: 0 }
      : { opacity: 0, y, x },
    visible: {
      opacity: 1,
      y: 0,
      x: 0,
      transition: {
        duration: shouldReduceMotion ? 0 : duration,
        delay: shouldReduceMotion ? 0 : delay,
        ease: 'easeOut' as const,
      },
    },
  };

  const viewportOptions = {
    once,
    margin: '-50px' as const,
  };

  return { variants, viewportOptions };
};

/** Create stagger container variants */
export const staggerContainer = (staggerDelay = 0.15): Variants => ({
  hidden: {},
  visible: {
    transition: {
      staggerChildren: staggerDelay,
    },
  },
});

/** Standard fade-up child variants */
export const fadeUpChild: Variants = {
  hidden: { opacity: 0, y: 40 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: 'easeOut' as const },
  },
};
