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
  HStack,
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
import { BatteryGauge, SOC_EXPLANATION, SOC_LOW_THRESHOLD } from '../components/battery/BatteryGauge';
import { CircularSOHGauge } from '../components/battery/CircularSOHGauge';
import { SOCLowInfoTooltip } from '../components/common/SOCLowInfoTooltip';
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

  const winnerSOCLow = winner.avg_soc_percent < SOC_LOW_THRESHOLD;

  return (
    <Box>
      {/* Winner Announcement - SpaceX Minimal Style */}
      <Section>
        <Card
          bg="spacex.darkGray"
          color="white"
          size="lg"
          borderWidth="1px"
          borderColor="white"
          borderRadius="0"
        >
          <CardBody>
            <VStack spacing={6} align="center">
              <Badge
                bg="white"
                color="black"
                fontSize="md"
                px={6}
                py={2}
                borderRadius="0"
                textTransform="uppercase"
                fontWeight="bold"
                letterSpacing="widest"
              >
                {t('pages:results.winner.badge')}
              </Badge>
              <Heading size="2xl" fontWeight="800" color="white">
                {winner.vendor_name}
              </Heading>

              <SimpleGrid columns={{ base: 1, md: 4 }} spacing={8} w="full" textAlign="center">
                <Box>
                  <Text fontSize="3xl" fontWeight="bold" color="white">
                    {formatKRW(winner.revenue_krw, 0)}
                  </Text>
                  <Text
                    fontSize="xs"
                    color="spacex.textGray"
                    textTransform="uppercase"
                    letterSpacing="wider"
                  >
                    {t('pages:results.winner.totalRevenue')}
                  </Text>
                </Box>
                <Box>
                  <Text fontSize="3xl" fontWeight="bold" color="white">
                    {formatPercent(winner.soh_percent, 2)}
                  </Text>
                  <Text
                    fontSize="xs"
                    color="spacex.textGray"
                    textTransform="uppercase"
                    letterSpacing="wider"
                  >
                    {t('pages:results.winner.stateOfHealth')}
                  </Text>
                </Box>
                <Box>
                  <Text fontSize="3xl" fontWeight="bold" color="white">
                    {formatPercent(winner.roi_percent, 0)}
                  </Text>
                  <Text
                    fontSize="xs"
                    color="spacex.textGray"
                    textTransform="uppercase"
                    letterSpacing="wider"
                  >
                    {t('pages:results.winner.roi')}
                  </Text>
                </Box>
                <Box position="relative">
                  <HStack justify="flex-end" mb={3}>
                    {winnerSOCLow && <SOCLowInfoTooltip label={SOC_EXPLANATION} />}
                  </HStack>
                  <HStack spacing={6} justify="center">
                    <BatteryGauge soc={winner.avg_soc_percent} width="70px" height="140px" />
                    <CircularSOHGauge soh={winner.soh_percent} size={100} />
                  </HStack>
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

      {/* Vendor Comparison Table - SpaceX Minimal */}
      <Section>
        <Heading size="lg" mb={6} color="white">
          {t('pages:results.vendorComparison')}
        </Heading>
        <Box
          overflowX="auto"
          borderWidth="1px"
          borderColor="spacex.borderGray"
          borderRadius="0"
          bg="spacex.darkGray"
        >
          <Table variant="simple">
            <Thead>
              <Tr borderBottom="2px solid" borderColor="white">
                <Th color="white" textTransform="uppercase">
                  {t('pages:results.table.vendor')}
                </Th>
                <Th color="white" isNumeric textTransform="uppercase">
                  {t('pages:results.table.revenue')}
                </Th>
                <Th color="white" isNumeric textTransform="uppercase">
                  {t('pages:results.table.soh')}
                </Th>
                <Th color="white" isNumeric textTransform="uppercase">
                  {t('pages:results.table.cycles')}
                </Th>
                <Th color="white" isNumeric textTransform="uppercase">
                  {t('pages:results.table.roi')}
                </Th>
                <Th color="white" isNumeric textTransform="uppercase">
                  {t('pages:results.table.payback')}
                </Th>
                <Th color="white" isNumeric textTransform="uppercase">
                  {t('pages:results.table.npv')}
                </Th>
              </Tr>
            </Thead>
            <Tbody>
              {data.vendors.map((vendor) => (
                <Tr
                  key={vendor.vendor_id}
                  bg={vendor.vendor_id === winner.vendor_id ? 'rgba(0, 217, 255, 0.05)' : undefined}
                  borderBottom="1px solid"
                  borderColor="gray.700"
                  _hover={{
                    bg: 'rgba(0, 217, 255, 0.1)',
                    transition: 'background 0.3s ease',
                  }}
                >
                  <Td fontWeight={vendor.vendor_id === winner.vendor_id ? 'bold' : 'normal'} color="gray.200">
                    {vendor.vendor_name}
                    {vendor.vendor_id === winner.vendor_id && (
                      <Badge ml={2} bg="white" color="black" boxShadow="0 0 8px rgba(255, 215, 0, 0.5)">
                        {t('pages:results.table.best')}
                      </Badge>
                    )}
                  </Td>
                  <Td isNumeric color="gray.300">{formatKRW(vendor.revenue_krw, 0)}</Td>
                  <Td isNumeric color="gray.300">{formatPercent(vendor.soh_percent, 2)}</Td>
                  <Td isNumeric color="gray.300">{formatNumber(vendor.cycle_count, 1)}</Td>
                  <Td isNumeric color="gray.300">{formatPercent(vendor.roi_percent, 0)}</Td>
                  <Td isNumeric color="gray.300">
                    {formatNumber(vendor.payback_years, 1)} {t('pages:results.table.years')}
                  </Td>
                  <Td isNumeric color="gray.300">{formatKRW(vendor.npv_krw, 0)}</Td>
                </Tr>
              ))}
              <Tr bg="rgba(0, 102, 255, 0.1)" fontWeight="bold" borderTop="2px solid" borderColor="white">
                <Td color="gray.200">{t('pages:results.table.baseline')}</Td>
                <Td isNumeric color="gray.300">{formatKRW(data.baseline.revenue_krw, 0)}</Td>
                <Td isNumeric colSpan={5} color="gray.500">
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

