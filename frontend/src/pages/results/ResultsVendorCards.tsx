/**
 * ResultsVendorCards - Score cards with composite score + sparkline
 */
import { SimpleGrid, Box, Text, VStack, HStack, Container } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import type { VendorResult, TimeSeriesData } from '../../types/simulation';
import type { VendorScore } from '../../utils/scoring';
import { ScoreGauge } from '../../components/data-display/ScoreGauge';
import { RankBadge } from '../../components/data-display/RankBadge';
import { SparkLine } from '../../components/data-display/SparkLine';
import { getVendorColor } from '../../utils/vendorColors';
import { formatKRW, formatPercent, formatNumber } from '../../utils/formatters';
import { staggerContainer, fadeUpChild } from '../../hooks/useInViewAnimation';

const MotionBox = motion(Box);
const MotionSimpleGrid = motion(SimpleGrid);

interface ResultsVendorCardsProps {
  vendors: VendorResult[];
  scores: VendorScore[];
  timeSeries: TimeSeriesData;
}

const getProfitData = (vendorId: string, ts: TimeSeriesData): number[] => {
  switch (vendorId) {
    case 'lg': return ts.lg_profit_krw || [];
    case 'samsung': return ts.samsung_profit_krw || [];
    case 'tesla': return ts.tesla_profit_krw || [];
    default: return [];
  }
};

export const ResultsVendorCards = ({ vendors, scores, timeSeries }: ResultsVendorCardsProps) => {
  const { t } = useTranslation('pages');

  return (
    <Box py={16}>
      <Container maxW="7xl" px={{ base: 4, md: 6, lg: 8 }}>
        <MotionSimpleGrid
          columns={{ base: 1, md: 3 }}
          spacing={6}
          variants={staggerContainer(0.15)}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
        >
          {scores.map((score) => {
            const vendor = vendors.find((v) => v.vendor_id === score.vendorId);
            if (!vendor) return null;
            const color = getVendorColor(vendor.vendor_id);
            const isBest = score.rank === 1;
            const profitData = getProfitData(vendor.vendor_id, timeSeries);

            return (
              <MotionBox
                key={vendor.vendor_id}
                variants={fadeUpChild}
                bg="spacex.darkGray"
                border="1px solid"
                borderColor={isBest ? color.primary : 'spacex.borderGray'}
                position="relative"
                overflow="hidden"
                // @ts-ignore - Chakra/Framer transition conflict
                transition="all 0.3s ease"
                _hover={{
                  transform: 'translateY(-4px)',
                  borderColor: color.primary,
                  boxShadow: isBest ? `0 0 20px ${color.glow}` : `0 4px 12px rgba(0,0,0,0.6)`,
                }}
                cursor="default"
              >
                {/* Accent bar */}
                <Box h="4px" bg={color.primary} />

                <VStack spacing={4} p={6} align="stretch">
                  {/* Header */}
                  <HStack justify="space-between" align="center">
                    <Text fontSize="md" fontWeight="bold" color="white">
                      {vendor.vendor_name}
                    </Text>
                    <RankBadge rank={score.rank} isBest={isBest} />
                  </HStack>

                  {/* Score gauge */}
                  <Box display="flex" justifyContent="center">
                    <ScoreGauge
                      score={score.score}
                      grade={score.grade}
                      color={color.primary}
                      size={90}
                    />
                  </Box>

                  {/* Key metrics */}
                  <VStack spacing={2} align="stretch">
                    <HStack justify="space-between">
                      <Text fontSize="xs" color="spacex.textGray" textTransform="uppercase">
                        {t('results.table.revenue', { defaultValue: 'Revenue' })}
                      </Text>
                      <Text fontSize="sm" fontWeight="bold" color="white" fontFamily="mono">
                        {formatKRW(vendor.revenue_krw, 0)}
                      </Text>
                    </HStack>
                    <HStack justify="space-between">
                      <Text fontSize="xs" color="spacex.textGray" textTransform="uppercase">
                        ROI
                      </Text>
                      <Text fontSize="sm" fontWeight="bold" color="battery.excellent" fontFamily="mono">
                        {formatPercent(vendor.roi_percent, 1)}
                      </Text>
                    </HStack>
                    <HStack justify="space-between">
                      <Text fontSize="xs" color="spacex.textGray" textTransform="uppercase">
                        {t('results.table.payback', { defaultValue: 'Payback' })}
                      </Text>
                      <Text fontSize="sm" fontWeight="bold" color="white" fontFamily="mono">
                        {formatNumber(vendor.payback_years, 1)} {t('results.table.years', { defaultValue: 'yr' })}
                      </Text>
                    </HStack>
                  </VStack>

                  {/* Sparkline */}
                  {profitData.length > 0 && (
                    <SparkLine data={profitData} color={color.primary} height={35} />
                  )}
                </VStack>

                {/* Best glow */}
                {isBest && (
                  <Box
                    position="absolute"
                    inset={0}
                    pointerEvents="none"
                    boxShadow={`inset 0 0 30px ${color.glow}`}
                    opacity={0.15}
                  />
                )}
              </MotionBox>
            );
          })}
        </MotionSimpleGrid>
      </Container>
    </Box>
  );
};
