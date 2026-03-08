/**
 * ResultsChartSection - Tabbed comparative visualization (Profit + Radar)
 */
import { useState } from 'react';
import { Box, Container, Heading, Tabs, TabList, Tab, TabPanels, TabPanel } from '@chakra-ui/react';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import type { VendorResult, TimeSeriesData } from '../../types/simulation';
import { ProfitChart } from '../../components/charts/ProfitChart';
import { VendorRadarChart } from '../../components/charts/RadarChart';
import { useInViewAnimation } from '../../hooks/useInViewAnimation';

const MotionBox = motion(Box);

interface ResultsChartSectionProps {
  vendors: VendorResult[];
  timeSeries: TimeSeriesData;
}

export const ResultsChartSection = ({ vendors, timeSeries }: ResultsChartSectionProps) => {
  const { t } = useTranslation('pages');
  const [tabIndex, setTabIndex] = useState(0);
  const shouldReduceMotion = useReducedMotion();
  const { variants, viewportOptions } = useInViewAnimation();

  const radarLabels: Record<string, string> = {
    revenue: t('results.table.revenue', { defaultValue: 'Revenue' }),
    roi: 'ROI',
    soh: 'SOH',
    payback: t('results.table.payback', { defaultValue: 'Payback' }),
    npv: 'NPV',
  };

  return (
    <Box py={16}>
      <Container maxW="7xl" px={{ base: 4, md: 6, lg: 8 }}>
        <MotionBox
          variants={variants}
          initial="hidden"
          whileInView="visible"
          viewport={viewportOptions}
        >
          <Heading size="lg" mb={6} color="white">
            {t('results.charts.title', { defaultValue: 'Comparative Analysis' })}
          </Heading>

          <Box
            bg="spacex.darkGray"
            border="1px solid"
            borderColor="spacex.borderGray"
            p={{ base: 4, md: 6 }}
          >
            <Tabs
              index={tabIndex}
              onChange={setTabIndex}
              variant="unstyled"
              isLazy
            >
              <TabList borderBottom="1px solid" borderColor="spacex.borderGray" mb={6}>
                <Tab
                  color="spacex.textGray"
                  fontWeight="bold"
                  fontSize="sm"
                  textTransform="uppercase"
                  letterSpacing="wider"
                  pb={3}
                  borderBottom="2px solid"
                  borderColor="transparent"
                  _selected={{ color: 'white', borderColor: 'solar.gold' }}
                  _hover={{ color: 'white' }}
                >
                  {t('results.charts.profitTab', { defaultValue: 'Cumulative Profit' })}
                </Tab>
                <Tab
                  color="spacex.textGray"
                  fontWeight="bold"
                  fontSize="sm"
                  textTransform="uppercase"
                  letterSpacing="wider"
                  pb={3}
                  borderBottom="2px solid"
                  borderColor="transparent"
                  _selected={{ color: 'white', borderColor: 'solar.gold' }}
                  _hover={{ color: 'white' }}
                >
                  {t('results.charts.radarTab', { defaultValue: 'Radar Comparison' })}
                </Tab>
              </TabList>

              <TabPanels>
                <TabPanel p={0}>
                  <AnimatePresence mode="wait">
                    <motion.div
                      key="profit"
                      initial={shouldReduceMotion ? {} : { opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <ProfitChart data={timeSeries} height={450} />
                    </motion.div>
                  </AnimatePresence>
                </TabPanel>
                <TabPanel p={0}>
                  <AnimatePresence mode="wait">
                    <motion.div
                      key="radar"
                      initial={shouldReduceMotion ? {} : { opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <VendorRadarChart
                        vendors={vendors}
                        axisLabels={radarLabels}
                        height={450}
                      />
                    </motion.div>
                  </AnimatePresence>
                </TabPanel>
              </TabPanels>
            </Tabs>
          </Box>
        </MotionBox>
      </Container>
    </Box>
  );
};
