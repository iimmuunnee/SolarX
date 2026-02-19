/**
 * GlowPulse Component
 * Neon glow pulse animation for CTA buttons and badges
 * Supports prefers-reduced-motion
 */
import { Box, type BoxProps } from '@chakra-ui/react';
import { keyframes } from '@emotion/react';

const glowPulse = keyframes`
  0%, 100% {
    box-shadow: 0 0 10px rgba(0, 217, 255, 0.4), 0 0 20px rgba(0, 217, 255, 0.2);
  }
  50% {
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.6), 0 0 40px rgba(0, 217, 255, 0.3), 0 0 60px rgba(0, 217, 255, 0.1);
  }
`;

interface GlowPulseProps extends BoxProps {
  children: React.ReactNode;
  duration?: string; // e.g., "2s"
  glowColor?: string;
}

export const GlowPulse = ({
  children,
  duration = '2s',
  glowColor = 'rgba(0, 217, 255, 0.5)',
  ...props
}: GlowPulseProps) => {
  return (
    <Box
      animation={`${glowPulse} ${duration} ease-in-out infinite`}
      sx={{
        '@media (prefers-reduced-motion: reduce)': {
          animation: 'none',
        },
      }}
      {...props}
    >
      {children}
    </Box>
  );
};

export default GlowPulse;
