/**
 * Metrics grid component - Displays key performance metrics
 */
import { SimpleGrid, Stat, StatLabel, StatNumber, StatHelpText, Card, CardBody } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import type { VendorResult, PredictionMetrics, SimulationMetadata } from '../../types/simulation';
import { formatKRW, formatPercent, formatNumber, formatEnergy } from '../../utils/formatters';

interface MetricsGridProps {
  vendorResult?: VendorResult;
  predictionMetrics?: PredictionMetrics;
  metadata?: SimulationMetadata;
}

export const MetricsGrid = ({ vendorResult, predictionMetrics, metadata }: MetricsGridProps) => {
  const { t } = useTranslation('charts');

  // SpaceX minimal card styling
  const minimalCardStyle = {
    bg: 'spacex.darkGray',
    borderWidth: '1px',
    borderColor: 'spacex.borderGray',
    borderRadius: '0',
    transition: 'all 0.3s ease',
    _hover: {
      borderColor: 'white',
      transform: 'translateY(-2px)',
    },
  };

  return (
    <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
      {/* Prediction Metrics */}
      {predictionMetrics && (
        <>
          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.mae.label')}
                </StatLabel>
                <StatNumber color="white" fontSize="2xl" fontWeight="bold">
                  {formatNumber(predictionMetrics.mae_kw, 2)} kW
                </StatNumber>
                <StatHelpText>{t('metrics.mae.help')}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.rmse.label')}
                </StatLabel>
                <StatNumber color="white" fontSize="2xl" fontWeight="bold">
                  {formatNumber(predictionMetrics.rmse_kw, 2)} kW
                </StatNumber>
                <StatHelpText>{t('metrics.rmse.help')}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.mape.label')}
                </StatLabel>
                <StatNumber color="white" fontSize="2xl" fontWeight="bold">
                  {formatPercent(predictionMetrics.mape_percent, 2)}
                </StatNumber>
                <StatHelpText>{t('metrics.mape.help')}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>
        </>
      )}

      {/* Vendor Results */}
      {vendorResult && (
        <>
          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.revenue.label')}
                </StatLabel>
                <StatNumber color="white" fontSize="2xl" fontWeight="bold">
                  {formatKRW(vendorResult.revenue_krw, 0)}
                </StatNumber>
                <StatHelpText>{t('metrics.revenue.help', { vendor: vendorResult.vendor_name })}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.soh.label')}
                </StatLabel>
                <StatNumber color="white" fontSize="2xl" fontWeight="bold">
                  {formatPercent(vendorResult.soh_percent, 2)}
                </StatNumber>
                <StatHelpText>{t('metrics.soh.help', { cycles: formatNumber(vendorResult.cycle_count, 1) })}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.throughput.label')}
                </StatLabel>
                <StatNumber color="white" fontSize="2xl" fontWeight="bold">
                  {formatEnergy(vendorResult.throughput_kwh, 'kWh')}
                </StatNumber>
                <StatHelpText>{t('metrics.throughput.help')}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.roi.label')}
                </StatLabel>
                <StatNumber
                  color={vendorResult.roi_percent >= 0 ? 'battery.excellent' : 'battery.critical'}
                  fontSize="2xl"
                  fontWeight="bold"
                >
                  {formatPercent(vendorResult.roi_percent, 2)}
                </StatNumber>
                <StatHelpText>{t('metrics.roi.help', { years: formatNumber(vendorResult.payback_years, 1) })}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.capex.label')}
                </StatLabel>
                <StatNumber color="white" fontSize="2xl" fontWeight="bold">
                  {formatKRW(vendorResult.capex_krw, 0)}
                </StatNumber>
                <StatHelpText>{t('metrics.capex.help')}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.npv.label')}
                </StatLabel>
                <StatNumber
                  color={vendorResult.npv_krw >= 0 ? 'battery.excellent' : 'battery.critical'}
                  fontSize="2xl"
                  fontWeight="bold"
                >
                  {formatKRW(vendorResult.npv_krw, 0)}
                </StatNumber>
                <StatHelpText>{t('metrics.npv.help')}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>
        </>
      )}

      {/* Simulation Metadata */}
      {metadata && (
        <>
          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.duration.label')}
                </StatLabel>
                <StatNumber color="white" fontSize="2xl" fontWeight="bold">
                  {metadata.duration_hours} {t('metrics.duration.hours')}
                </StatNumber>
                <StatHelpText>{t('metrics.duration.years', { years: formatNumber(metadata.simulation_years, 2) })}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>

          <Card {...minimalCardStyle}>
            <CardBody>
              <Stat>
                <StatLabel color="spacex.textGray" textTransform="uppercase" fontSize="xs">
                  {t('metrics.avgPrice.label')}
                </StatLabel>
                <StatNumber color="white" fontSize="2xl" fontWeight="bold">
                  {formatKRW(metadata.avg_smp_price, 2)}/kWh
                </StatNumber>
                <StatHelpText>{t('metrics.avgPrice.help')}</StatHelpText>
              </Stat>
            </CardBody>
          </Card>
        </>
      )}
    </SimpleGrid>
  );
};
