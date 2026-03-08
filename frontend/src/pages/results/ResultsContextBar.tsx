/**
 * ResultsContextBar - Simulation context metadata banner
 */
import { HStack, Text, Box, Container, Icon } from '@chakra-ui/react';
import { TimeIcon, InfoOutlineIcon } from '@chakra-ui/icons';
import { motion, useReducedMotion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import type { SimulationMetadata } from '../../types/simulation';
import { formatKRW, formatNumber } from '../../utils/formatters';

const MotionBox = motion(Box);

interface ResultsContextBarProps {
  metadata: SimulationMetadata;
  isFallback: boolean;
}

export const ResultsContextBar = ({ metadata, isFallback }: ResultsContextBarProps) => {
  const { t } = useTranslation('pages');
  const shouldReduceMotion = useReducedMotion();

  const items = [
    {
      icon: TimeIcon,
      label: t('results.context.duration', { defaultValue: 'Duration' }),
      value: `${formatNumber(metadata.duration_hours, 0)}h / ${Math.round(metadata.duration_hours / 24)}${t('results.context.days', { defaultValue: 'd' })}`,
    },
    {
      icon: InfoOutlineIcon,
      label: t('results.context.avgSmp', { defaultValue: 'Avg SMP' }),
      value: `${formatKRW(metadata.avg_smp_price, 2)}/kWh`,
    },
    {
      icon: InfoOutlineIcon,
      label: t('results.context.dataPoints', { defaultValue: 'Data Points' }),
      value: formatNumber(metadata.data_points, 0),
    },
  ];

  return (
    <MotionBox
      initial={shouldReduceMotion ? {} : { opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      // @ts-ignore
      transition={{ duration: 0.5, delay: 0 }}
      bg="rgba(255, 255, 255, 0.03)"
      borderBottom="1px solid"
      borderColor="spacex.borderGray"
      py={3}
    >
      <Container maxW="7xl" px={{ base: 4, md: 6, lg: 8 }}>
        <HStack
          spacing={{ base: 4, md: 8 }}
          justify="center"
          flexWrap="wrap"
        >
          {items.map((item) => (
            <HStack key={item.label} spacing={2}>
              <Icon as={item.icon} boxSize={3} color="solar.gold" />
              <Text fontSize="xs" color="spacex.textGray" fontFamily="mono">
                <Text as="span" color="spacex.borderGray" textTransform="uppercase" letterSpacing="wider">
                  {item.label}
                </Text>
                {' '}
                {item.value}
              </Text>
            </HStack>
          ))}
          {isFallback && (
            <Text fontSize="xs" color="solar.amber" fontFamily="mono">
              CACHED DATA
            </Text>
          )}
        </HStack>
      </Container>
    </MotionBox>
  );
};
