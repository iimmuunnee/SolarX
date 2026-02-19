/**
 * Demo page - Interactive simulation
 */
import { useState } from 'react';
import {
  Box,
  Grid,
  Heading,
  VStack,
  HStack,
  FormControl,
  FormLabel,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Switch,
  Button,
  Alert,
  AlertIcon,
  Spinner,
  Text,
  Select,
  Skeleton,
  SkeletonText,
  SimpleGrid,
  Tooltip,
} from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import { Section } from '../components/layout/Section';
import { useSimulation } from '../hooks/useSimulation';
import { ProfitChart } from '../components/charts/ProfitChart';
import { PredictionChart } from '../components/charts/PredictionChart';
import { MetricsGrid } from '../components/charts/MetricsGrid';
import { BatteryGauge, SOC_EXPLANATION, SOC_LOW_THRESHOLD } from '../components/battery/BatteryGauge';
import { CircularSOHGauge } from '../components/battery/CircularSOHGauge';
import { MethodologyPanel } from '../components/charts/MethodologyPanel';
import type { BenchmarkRequest } from '../types/simulation';

export const Demo = () => {
  const { t } = useTranslation(['common', 'pages']);
  const { loading, error, result, simulateBenchmark } = useSimulation();

  // Simulation parameters
  const [params, setParams] = useState<BenchmarkRequest>({
    battery_capacity_kwh: 2280,
    charge_threshold: 0.9,
    discharge_threshold: 1.1,
    allow_grid_charge: true,
    region_factor: 1.0,
  });

  const handleRunSimulation = async () => {
    try {
      await simulateBenchmark(params);
    } catch (err) {
      console.error('Simulation failed:', err);
    }
  };

  const benchmarkResult = result && 'vendors' in result ? result : null;

  const hasLowSOC =
    benchmarkResult?.vendors.some((v) => v.avg_soc_percent < SOC_LOW_THRESHOLD) ?? false;

  return (
    <Box>
      <Section>
        <Heading mb={6}>{t('pages:demo.title')}</Heading>
        <Text mb={8} color="gray.600">
          {t('pages:demo.subtitle')}
        </Text>

        <Grid templateColumns={{ base: '1fr', lg: '1fr 2fr' }} gap={8}>
          {/* Parameter Panel - Neon Style */}
          <VStack spacing={6} align="stretch">
            <Box
              p={6}
              bg="spacex.darkGray"
              borderWidth="2px"
              borderColor="spacex.borderGray"
              borderRadius="lg"
              boxShadow="0 0 10px rgba(0, 102, 255, 0.3)"
              backdropFilter="blur(10px)"
            >
              <Heading size="md" mb={4} color="white">
                {t('pages:demo.parameters.title')}
              </Heading>

              {/* Battery Capacity */}
              <FormControl mb={4}>
                <FormLabel color="gray.300">
                  {t('pages:demo.parameters.batteryCapacity')}: <Text as="span" color="white" fontWeight="bold">{params.battery_capacity_kwh} kWh</Text>
                </FormLabel>
                <Slider
                  value={params.battery_capacity_kwh}
                  min={500}
                  max={5000}
                  step={100}
                  onChange={(value) => setParams({ ...params, battery_capacity_kwh: value })}
                >
                  <SliderTrack bg="gray.700">
                    <SliderFilledTrack bg="white" boxShadow="0 0 8px rgba(0, 217, 255, 0.5)" />
                  </SliderTrack>
                  <SliderThumb boxSize={6} bg="white" />
                </Slider>
                <Text fontSize="sm" color="gray.500" mt={1}>
                  {t('pages:demo.parameters.batteryCapacityRange')}
                </Text>
              </FormControl>

              {/* Charge Threshold */}
              <FormControl mb={4}>
                <FormLabel color="gray.300">
                  {t('pages:demo.parameters.chargeThreshold')}: <Text as="span" color="white" fontWeight="bold">{params.charge_threshold}×</Text>
                </FormLabel>
                <Slider
                  value={params.charge_threshold}
                  min={0.5}
                  max={1.5}
                  step={0.05}
                  onChange={(value) => setParams({ ...params, charge_threshold: value })}
                >
                  <SliderTrack bg="gray.700">
                    <SliderFilledTrack bg="white" boxShadow="0 0 8px rgba(0, 217, 255, 0.5)" />
                  </SliderTrack>
                  <SliderThumb boxSize={6} bg="white" />
                </Slider>
                <Text fontSize="sm" color="gray.500" mt={1}>
                  {t('pages:demo.parameters.chargeThresholdHelp', { threshold: params.charge_threshold })}
                </Text>
              </FormControl>

              {/* Discharge Threshold */}
              <FormControl mb={4}>
                <FormLabel color="gray.300">
                  {t('pages:demo.parameters.dischargeThreshold')}: <Text as="span" color="white" fontWeight="bold">{params.discharge_threshold}×</Text>
                </FormLabel>
                <Slider
                  value={params.discharge_threshold}
                  min={0.7}
                  max={2.0}
                  step={0.05}
                  onChange={(value) => setParams({ ...params, discharge_threshold: value })}
                >
                  <SliderTrack bg="gray.700">
                    <SliderFilledTrack bg="white" boxShadow="0 0 8px rgba(0, 217, 255, 0.5)" />
                  </SliderTrack>
                  <SliderThumb boxSize={6} bg="white" />
                </Slider>
                <Text fontSize="sm" color="gray.500" mt={1}>
                  {t('pages:demo.parameters.dischargeThresholdHelp', { threshold: params.discharge_threshold })}
                </Text>
              </FormControl>

              {/* Region Factor */}
              <FormControl mb={4}>
                <FormLabel>{t('pages:demo.parameters.regionFactor')}: {params.region_factor}×</FormLabel>
                <Select
                  value={params.region_factor}
                  onChange={(e) => setParams({ ...params, region_factor: parseFloat(e.target.value) })}
                >
                  <option value="0.6">{t('pages:demo.parameters.regionSeattle')}</option>
                  <option value="1.0">{t('pages:demo.parameters.regionDonghae')}</option>
                  <option value="1.3">{t('pages:demo.parameters.regionJeju')}</option>
                </Select>
              </FormControl>

              {/* Allow Grid Charge */}
              <FormControl display="flex" alignItems="center" mb={4}>
                <FormLabel mb={0}>{t('pages:demo.parameters.allowGridCharge')}</FormLabel>
                <Switch
                  isChecked={params.allow_grid_charge}
                  onChange={(e) => setParams({ ...params, allow_grid_charge: e.target.checked })}
                />
              </FormControl>

              {/* Run Button */}
              <Button
                variant="spacexSolid"
                size="lg"
                w="full"
                onClick={handleRunSimulation}
                isLoading={loading}
                loadingText={t('common:loadingSimulation')}
              >
                {t('common:buttons.runSimulation')}
              </Button>
            </Box>

            {/* Battery Gauges (show when results exist) */}
            {benchmarkResult && !loading && (
              <Box
                position="relative"
                p={6}
                bg="spacex.darkGray"
                borderWidth="1px"
                borderColor="spacex.borderGray"
                borderRadius="0"
              >
                <Heading size="md" mb={4} color="white" textAlign="center">
                  Battery Status
                </Heading>

                {/* Single SOC tooltip — shown only when any battery SOC is low */}
                {hasLowSOC && (
                  <Tooltip
                    label={SOC_EXPLANATION}
                    placement="top-end"
                    hasArrow
                    bg="gray.700"
                    color="white"
                    fontSize="xs"
                    maxW="260px"
                    p={3}
                    borderRadius="md"
                    textAlign="left"
                  >
                    <Text
                      position="absolute"
                      top={3}
                      right={4}
                      fontSize="2xs"
                      color="spacex.textGray"
                      textTransform="uppercase"
                      letterSpacing="wide"
                      cursor="help"
                      borderBottom="1px dotted"
                      borderBottomColor="spacex.textGray"
                      _hover={{ color: 'white', borderBottomColor: 'white' }}
                    >
                      왜 SOC가 낮나요?
                    </Text>
                  </Tooltip>
                )}

                <SimpleGrid columns={1} spacing={6}>
                  {benchmarkResult.vendors.map((vendor) => (
                    <VStack key={vendor.vendor_id} spacing={4}>
                      <Text color="gray.300" fontWeight="bold" fontSize="sm" textTransform="uppercase">
                        {vendor.vendor_name}
                      </Text>
                      <HStack spacing={6} justify="center">
                        <BatteryGauge soc={vendor.avg_soc_percent} width="70px" height="140px" />
                        <CircularSOHGauge soh={vendor.soh_percent} size={100} />
                      </HStack>
                    </VStack>
                  ))}
                </SimpleGrid>
              </Box>
            )}
          </VStack>

          {/* Results Panel */}
          <VStack spacing={6} align="stretch">
            {loading && (
              <VStack spacing={4} py={12}>
                <Spinner size="xl" color="blue.500" />
                <Text>{t('pages:demo.loadingMessage')}</Text>
                <Skeleton height="200px" width="100%" />
                <SkeletonText noOfLines={4} spacing={4} width="100%" />
              </VStack>
            )}

            {error && (
              <Alert status="error">
                <AlertIcon />
                {error}
              </Alert>
            )}

            {benchmarkResult && !loading && (
              <>
                <Box>
                  <Heading size="md" mb={4}>
                    {t('pages:demo.results.predictionQuality')}
                  </Heading>
                  <MetricsGrid
                    predictionMetrics={benchmarkResult.prediction_metrics}
                    metadata={benchmarkResult.metadata}
                  />
                </Box>

                <Box>
                  <ProfitChart data={benchmarkResult.time_series} />
                </Box>

                <Box>
                  <PredictionChart data={benchmarkResult.time_series} maxPoints={200} />
                </Box>

                <Box>
                  <Heading size="md" mb={6}>
                    {t('pages:demo.results.vendorResults')}
                  </Heading>
                  {benchmarkResult.vendors.map((vendor) => {
                    const vendorColors: Record<string, string> = {
                      lg: '#10B981',
                      samsung: '#3B82F6',
                      tesla: '#EF4444',
                    };
                    const color = vendorColors[vendor.vendor_id] ?? 'white';
                    return (
                      <Box
                        key={vendor.vendor_id}
                        mb={10}
                        borderLeft="4px solid"
                        borderColor={color}
                        pl={5}
                      >
                        <HStack mb={4} spacing={3}>
                          <Box w="8px" h="8px" borderRadius="full" bg={color} flexShrink={0} />
                          <Heading
                            size="sm"
                            color={color}
                            textTransform="uppercase"
                            letterSpacing="wider"
                          >
                            {vendor.vendor_name}
                          </Heading>
                        </HStack>
                        <MetricsGrid vendorResult={vendor} />
                      </Box>
                    );
                  })}
                </Box>

                <Box>
                  <MethodologyPanel
                    capacityKwh={params.battery_capacity_kwh}
                    vendors={benchmarkResult.vendors}
                  />
                </Box>
              </>
            )}

            {!benchmarkResult && !loading && !error && (
              <Box textAlign="center" py={12} color="gray.500">
                <Text fontSize="lg">{t('pages:demo.results.emptyState')}</Text>
              </Box>
            )}
          </VStack>
        </Grid>
      </Section>
    </Box>
  );
};
