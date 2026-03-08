/**
 * ResultsPredictionSection - Prediction accuracy metrics + chart
 */
import { Box, Container, Heading, SimpleGrid, VStack, Text, Collapse, Button } from '@chakra-ui/react';
import { ChevronDownIcon, ChevronUpIcon } from '@chakra-ui/icons';
import { useState } from 'react';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import type { PredictionMetrics, TimeSeriesData } from '../../types/simulation';
import { PredictionChart } from '../../components/charts/PredictionChart';
import { MetricCard } from '../../components/data-display/MetricCard';
import { ConfidenceIndicator } from '../../components/data-display/ConfidenceIndicator';
import { formatNumber, formatPercent } from '../../utils/formatters';
import { useInViewAnimation } from '../../hooks/useInViewAnimation';

const MotionBox = motion(Box);

interface ResultsPredictionSectionProps {
  metrics: PredictionMetrics;
  timeSeries: TimeSeriesData;
}

export const ResultsPredictionSection = ({ metrics, timeSeries }: ResultsPredictionSectionProps) => {
  const { t } = useTranslation(['pages', 'charts']);
  const [isOpen, setIsOpen] = useState(true);
  const { variants, viewportOptions } = useInViewAnimation();

  return (
    <Box py={16}>
      <Container maxW="7xl" px={{ base: 4, md: 6, lg: 8 }}>
        <MotionBox
          variants={variants}
          initial="hidden"
          whileInView="visible"
          viewport={viewportOptions}
        >
          <Button
            variant="unstyled"
            display="flex"
            alignItems="center"
            gap={2}
            mb={6}
            onClick={() => setIsOpen(!isOpen)}
            _hover={{ opacity: 0.8 }}
          >
            <Heading size="lg" color="white">
              {t('pages:results.predictionQuality')}
            </Heading>
            {isOpen ? (
              <ChevronUpIcon boxSize={6} color="white" />
            ) : (
              <ChevronDownIcon boxSize={6} color="white" />
            )}
          </Button>

          <Collapse in={isOpen} animateOpacity>
            <SimpleGrid columns={{ base: 1, lg: 3 }} spacing={6}>
              {/* Left: Metrics */}
              <VStack spacing={4} align="stretch">
                <Box>
                  <MetricCard
                    label="MAE"
                    value={`${formatNumber(metrics.mae_kw, 2)} kW`}
                    tooltip={t('charts:metrics.mae.help', { defaultValue: 'Mean Absolute Error' })}
                  />
                  <Box mt={2}>
                    <ConfidenceIndicator
                      value={metrics.mae_kw}
                      thresholds={{ good: 50, fair: 100 }}
                      invert
                    />
                  </Box>
                </Box>

                <Box>
                  <MetricCard
                    label="RMSE"
                    value={`${formatNumber(metrics.rmse_kw, 2)} kW`}
                    tooltip={t('charts:metrics.rmse.help', { defaultValue: 'Root Mean Square Error' })}
                  />
                  <Box mt={2}>
                    <ConfidenceIndicator
                      value={metrics.rmse_kw}
                      thresholds={{ good: 80, fair: 150 }}
                      invert
                    />
                  </Box>
                </Box>

                <Box>
                  <MetricCard
                    label="MAPE"
                    value={formatPercent(metrics.mape_percent, 2)}
                    tooltip={t('charts:metrics.mape.help', { defaultValue: 'Mean Absolute Percentage Error' })}
                  />
                  <Box mt={2}>
                    <ConfidenceIndicator
                      value={metrics.mape_percent}
                      thresholds={{ good: 30, fair: 70 }}
                      invert
                    />
                  </Box>
                  <Text fontSize="xs" color="spacex.borderGray" mt={2}>
                    {t('pages:results.prediction.mapeNote', {
                      defaultValue: 'MAPE 60% range is typical for solar generation prediction with weather variability',
                    })}
                  </Text>
                </Box>
              </VStack>

              {/* Right: Chart */}
              <Box gridColumn={{ lg: 'span 2' }}>
                <PredictionChart data={timeSeries} maxPoints={200} height={380} />
              </Box>
            </SimpleGrid>
          </Collapse>
        </MotionBox>
      </Container>
    </Box>
  );
};
