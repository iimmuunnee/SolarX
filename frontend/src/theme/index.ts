/**
 * SpaceX-inspired Minimal Dark Theme for SolarX
 * Ultra-clean, minimal design with black/white/gray palette
 */
import { extendTheme, type ThemeConfig } from '@chakra-ui/react';

// Theme configuration
const config: ThemeConfig = {
  initialColorMode: 'dark',
  useSystemColorMode: false,
};

// SpaceX minimal color palette - pure black and white
const colors = {
  spacex: {
    black: '#111827',
    darkGray: '#1c2536',
    mediumGray: '#2d3748',
    lightGray: '#3d4f6a',
    borderGray: '#4a5568',
    textGray: '#e5e7eb',
    white: '#ffffff',
  },
  // Solar energy accent colors
  solar: {
    gold: '#FFD700',      // Primary solar accent
    amber: '#FFA500',     // Secondary solar accent
    dim: '#B8860B',       // Muted solar accent
  },
  // Battery health status colors
  battery: {
    excellent: '#10B981', // SOH ≥ 90%
    good: '#22C55E',      // SOH 75-90%
    warning: '#F59E0B',   // SOH 50-75%
    critical: '#EF4444',  // SOH < 50%
    charging: '#3B82F6',  // Charging state
    discharging: '#8B5CF6', // Discharging state
  },
  // Chart accent colors
  chart: {
    prediction: '#FFD700', // Solar predictions
    actual: '#FFFFFF',     // Actual data (white)
    profit: '#10B981',     // Profit/revenue
  },
};

// Minimal shadows - no glow effects
const shadows = {
  minimal: '0 2px 8px rgba(0, 0, 0, 0.5)',
  card: '0 4px 12px rgba(0, 0, 0, 0.6)',
};

// Clean typography - sans-serif focus
const fonts = {
  heading: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`,
  body: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`,
  mono: `'Fira Code', 'Courier New', monospace`,
};

// Component style customizations - SpaceX minimal style
const components = {
  Button: {
    variants: {
      spacex: {
        bg: 'transparent',
        color: 'white',
        border: '2px solid',
        borderColor: 'white',
        fontWeight: 'bold',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        fontSize: 'sm',
        px: 8,
        py: 6,
        '& svg': {
          transition: 'transform 0.3s ease',
        },
        _hover: {
          bg: 'white',
          color: 'black',
          transform: 'translateY(-2px)',
          '& svg': {
            transform: 'translateX(4px)',
          },
        },
        _active: {
          transform: 'translateY(0)',
        },
        transition: 'all 0.3s ease',
      },
      spacexSolid: {
        bg: 'white',
        color: 'black',
        fontWeight: 'bold',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        fontSize: 'sm',
        px: 8,
        py: 6,
        _hover: {
          bg: 'spacex.textGray',
          transform: 'translateY(-2px)',
        },
        _active: {
          transform: 'translateY(0)',
        },
        transition: 'all 0.3s ease',
      },
      spacexSolar: {
        bg: 'transparent',
        color: 'white',
        border: '2px solid',
        borderColor: 'solar.gold',
        fontWeight: 'bold',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        fontSize: 'sm',
        px: 8,
        py: 6,
        _hover: {
          bg: 'solar.gold',
          color: 'black',
          transform: 'translateY(-2px)',
          boxShadow: '0 0 20px rgba(255, 215, 0, 0.3)',
        },
        _active: {
          transform: 'translateY(0)',
        },
        transition: 'all 0.3s ease',
      },
    },
  },
  Card: {
    variants: {
      spacex: {
        container: {
          bg: 'spacex.darkGray',
          borderWidth: '1px',
          borderColor: 'spacex.borderGray',
          borderRadius: 'none',
          transition: 'all 0.3s ease',
          _hover: {
            borderColor: 'white',
            transform: 'translateY(-8px)',
          },
        },
      },
      minimal: {
        container: {
          bg: 'transparent',
          borderWidth: '1px',
          borderColor: 'spacex.borderGray',
          borderRadius: 'none',
        },
      },
    },
  },
  Heading: {
    baseStyle: {
      fontFamily: 'heading',
      fontWeight: 'bold',
      color: 'white',
    },
  },
  Text: {
    baseStyle: {
      color: 'spacex.textGray',
    },
  },
  Stat: {
    baseStyle: {
      container: {
        color: 'spacex.textGray',
      },
      label: {
        color: 'spacex.textGray',
        textTransform: 'uppercase',
        fontSize: 'xs',
        fontWeight: 'medium',
        letterSpacing: '0.1em',
      },
      number: {
        color: 'white',
        fontSize: '2xl',
        fontWeight: 'bold',
      },
    },
  },
};

// Global styles
const styles = {
  global: {
    body: {
      bg: 'spacex.black',
      color: 'spacex.textGray',
      fontFamily: 'body',
    },
    '*::placeholder': {
      color: 'spacex.borderGray',
    },
    '*, *::before, *::after': {
      borderColor: 'spacex.borderGray',
    },
  },
};

// Extended theme
export const theme = extendTheme({
  config,
  colors,
  shadows,
  fonts,
  components,
  styles,
});

export default theme;
