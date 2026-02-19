/**
 * DotPattern Component
 * CSS-based dot pattern for performance optimization (GPU-accelerated)
 * Used for battery gauges and decorative backgrounds
 */
import { Box, type BoxProps } from '@chakra-ui/react';

interface DotPatternProps extends BoxProps {
  color?: string;
  opacity?: number;
  size?: number; // dot size in pixels
  spacing?: number; // spacing between dots in pixels
}

export const DotPattern = ({
  color = 'rgba(0, 217, 255, 0.3)',
  opacity = 0.3,
  size = 2,
  spacing = 20,
  ...props
}: DotPatternProps) => {
  return (
    <Box
      position="absolute"
      top={0}
      left={0}
      right={0}
      bottom={0}
      pointerEvents="none"
      opacity={opacity}
      backgroundImage={`radial-gradient(circle, ${color} ${size}px, transparent ${size}px)`}
      backgroundSize={`${spacing}px ${spacing}px`}
      backgroundPosition="0 0, ${spacing / 2}px ${spacing / 2}px"
      {...props}
    />
  );
};

export default DotPattern;
