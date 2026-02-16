/**
 * Results page - Benchmark showcase
 */
import {
  Heading,
  Text,
  Spinner,
  Alert,
  AlertIcon,
  VStack,
  Box,
  SimpleGrid,
  Card,
  CardBody,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Skeleton,
  SkeletonText,
} from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import { Section } from '../components/layout/Section';
import { usePrecomputed } from '../hooks/usePrecomputed';
import { ProfitChart } from '../components/charts/ProfitChart';
import { PredictionChart } from '../components/charts/PredictionChart';
import { MetricsGrid } from '../components/charts/MetricsGrid';
import { formatKRW, formatPercent, formatNumber } from '../utils/formatters';

export const Results = () => {
  const { t } = useTranslation(['common', 'pages']);
  const { data, loading, error } = usePrecomputed();

  if (loading) {
    return (
      <Section>
        <VStack spacing={8}>
          <Spinner size="xl" color="blue.500" />
          <Text>{t('pages:results.loading')}</Text>
          <Skeleton height="300px" width="100%" />
          <SkeletonText noOfLines={6} spacing={4} width="100%" />
        </VStack>
      </Section>
    );
  }

  if (error) {
    return (
      <Section>
        <Alert status="error">
          <AlertIcon />
          {error}
        </Alert>
      </Section>
    );
  }

  if (!data) {
    return (
      <Section>
        <Alert status="info">
          <AlertIcon />
          {t('common:noData')}
        </Alert>
      </Section>
    );
  }

  // Find the winner (highest revenue)
  const winner = data.vendors.reduce((prev, current) =>
    current.revenue_krw > prev.revenue_krw ? current : prev
  );

  return (
    <Box>
      {/* Winner Announcement */}
      <Section bg="blue.50">
        <Card bg="blue.500" color="white" size="lg">
          <CardBody>
            <VStack spacing={4} align="center">
              <Badge colorScheme="yellow" fontSize="lg" px={4} py={2}>
                {t('pages:results.winner.badge')}
              </Badge>
              <Heading size="xl">{winner.vendor_name}</Heading>
              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8} w="full" textAlign="center">
                <Box>
                  <Text fontSize="3xl" fontWeight="bold">
                    {formatKRW(winner.revenue_krw, 0)}
                  </Text>
                  <Text fontSize="sm">{t('pages:results.winner.totalRevenue')}</Text>
                </Box>
                <Box>
                  <Text fontSize="3xl" fontWeight="bold">
                    {formatPercent(winner.soh_percent, 2)}
                  </Text>
                  <Text fontSize="sm">{t('pages:results.winner.stateOfHealth')}</Text>
                </Box>
                <Box>
                  <Text fontSize="3xl" fontWeight="bold">
                    {formatPercent(winner.roi_percent, 0)}
                  </Text>
                  <Text fontSize="sm">{t('pages:results.winner.roi')}</Text>
                </Box>
              </SimpleGrid>
            </VStack>
          </CardBody>
        </Card>
      </Section>

      {/* Prediction Quality Metrics */}
      <Section>
        <Heading size="lg" mb={6}>
          {t('pages:results.predictionQuality')}
        </Heading>
        <MetricsGrid predictionMetrics={data.prediction_metrics} metadata={data.metadata} />
      </Section>

      {/* Charts */}
      <Section>
        <VStack spacing={8} align="stretch">
          <ProfitChart data={data.time_series} />
          <PredictionChart data={data.time_series} maxPoints={200} />
        </VStack>
      </Section>

      {/* Vendor Comparison Table */}
      <Section>
        <Heading size="lg" mb={6}>
          {t('pages:results.vendorComparison')}
        </Heading>
        <Box overflowX="auto">
          <Table variant="simple">
            <Thead>
              <Tr>
                <Th>{t('pages:results.table.vendor')}</Th>
                <Th isNumeric>{t('pages:results.table.revenue')}</Th>
                <Th isNumeric>{t('pages:results.table.soh')}</Th>
                <Th isNumeric>{t('pages:results.table.cycles')}</Th>
                <Th isNumeric>{t('pages:results.table.roi')}</Th>
                <Th isNumeric>{t('pages:results.table.payback')}</Th>
                <Th isNumeric>{t('pages:results.table.npv')}</Th>
              </Tr>
            </Thead>
            <Tbody>
              {data.vendors.map((vendor) => (
                <Tr key={vendor.vendor_id} bg={vendor.vendor_id === winner.vendor_id ? 'blue.50' : undefined}>
                  <Td fontWeight={vendor.vendor_id === winner.vendor_id ? 'bold' : 'normal'}>
                    {vendor.vendor_name}
                    {vendor.vendor_id === winner.vendor_id && (
                      <Badge ml={2} colorScheme="blue">
                        {t('pages:results.table.best')}
                      </Badge>
                    )}
                  </Td>
                  <Td isNumeric>{formatKRW(vendor.revenue_krw, 0)}</Td>
                  <Td isNumeric>{formatPercent(vendor.soh_percent, 2)}</Td>
                  <Td isNumeric>{formatNumber(vendor.cycle_count, 1)}</Td>
                  <Td isNumeric>{formatPercent(vendor.roi_percent, 0)}</Td>
                  <Td isNumeric>{formatNumber(vendor.payback_years, 1)} {t('pages:results.table.years')}</Td>
                  <Td isNumeric>{formatKRW(vendor.npv_krw, 0)}</Td>
                </Tr>
              ))}
              <Tr bg="gray.100" fontWeight="bold">
                <Td>{t('pages:results.table.baseline')}</Td>
                <Td isNumeric>{formatKRW(data.baseline.revenue_krw, 0)}</Td>
                <Td isNumeric colSpan={5}>
                  -
                </Td>
              </Tr>
            </Tbody>
          </Table>
        </Box>
      </Section>
    </Box>
  );
};
