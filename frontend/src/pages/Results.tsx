/**
 * Results page - Solar Observatory redesign
 * Orchestrator for 8 visual sections
 */
import { Box, VStack, Spinner, Text, Alert, AlertIcon } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import { Section } from '../components/layout/Section';
import { usePrecomputed } from '../hooks/usePrecomputed';
import { useVendorComparison } from '../hooks/useVendorComparison';
import { ResultsContextBar } from './results/ResultsContextBar';
import { ResultsWinnerHero } from './results/ResultsWinnerHero';
import { ResultsVendorCards } from './results/ResultsVendorCards';
import { ResultsChartSection } from './results/ResultsChartSection';
import { ResultsPredictionSection } from './results/ResultsPredictionSection';
import { ResultsEconomicTable } from './results/ResultsEconomicTable';
import { ResultsInsightPanel } from './results/ResultsInsightPanel';
import { ResultsActionBar } from './results/ResultsActionBar';

export const Results = () => {
  const { t } = useTranslation(['common', 'pages']);
  const { data, loading, error, isFallback } = usePrecomputed();
  const { scores, winnerVendor } = useVendorComparison(data?.vendors ?? []);

  if (loading) {
    return (
      <Section>
        <VStack spacing={8} py={20}>
          <Spinner size="xl" color="solar.gold" />
          <Text>{t('pages:results.loading')}</Text>
        </VStack>
      </Section>
    );
  }

  if (error || !data) {
    return (
      <Section>
        <Alert status="error" borderRadius="0">
          <AlertIcon />
          {error || t('common:noData')}
        </Alert>
      </Section>
    );
  }

  if (!winnerVendor) return null;

  return (
    <Box>
      <ResultsContextBar metadata={data.metadata} isFallback={isFallback} />
      <ResultsWinnerHero winner={winnerVendor} baseline={data.baseline} />
      <ResultsVendorCards vendors={data.vendors} scores={scores} timeSeries={data.time_series} />
      <ResultsChartSection vendors={data.vendors} timeSeries={data.time_series} />
      <ResultsPredictionSection metrics={data.prediction_metrics} timeSeries={data.time_series} />
      <ResultsEconomicTable vendors={data.vendors} baseline={data.baseline} winnerId={winnerVendor.vendor_id} />
      <ResultsInsightPanel vendors={data.vendors} baseline={data.baseline} />
      <ResultsActionBar />
    </Box>
  );
};
