/**
 * GridPattern Component
 * CSS-based cyberpunk grid pattern for chart backgrounds
 * Optimized for GPU acceleration
 */
import { Box, type BoxProps } from '@chakra-ui/react';

interface GridPatternProps extends BoxProps {
  color?: string;
  opacity?: number;
  gridSize?: number; // grid cell size in pixels
  lineWidth?: number; // grid line width in pixels
}

export const GridPattern = ({
  color = 'rgba(0, 102, 255, 0.2)',
  opacity = 0.08,
  gridSize = 40,
  lineWidth = 1,
  ...props
}: GridPatternProps) => {
  const gradientLine = `${color} ${lineWidth}px, transparent ${lineWidth}px`;

  return (
    <Box
      position="absolute"
      top={0}
      left={0}
      right={0}
      bottom={0}
      pointerEvents="none"
      opacity={opacity}
      backgroundImage={`
        linear-gradient(0deg, ${gradientLine}),
        linear-gradient(90deg, ${gradientLine})
      `}
      backgroundSize={`${gridSize}px ${gridSize}px`}
      {...props}
    />
  );
};

export default GridPattern;
