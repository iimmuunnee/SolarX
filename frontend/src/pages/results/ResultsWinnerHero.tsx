/**
 * ResultsWinnerHero - Full-screen winner announcement with animations
 */
import { Box, Text, VStack, HStack, Badge, Container } from '@chakra-ui/react';
import { motion, useReducedMotion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { ChevronDownIcon } from '@chakra-ui/icons';
import type { VendorResult, BaselineResult } from '../../types/simulation';
import { CountUpNumber } from '../../components/data-display/CountUpNumber';
import { DotPattern } from '../../components/patterns/DotPattern';
import { getVendorColor } from '../../utils/vendorColors';
import { formatNumber, formatKRW } from '../../utils/formatters';

const MotionBox = motion(Box);
const MotionBadge = motion(Badge);
const MotionText = motion(Text);

interface ResultsWinnerHeroProps {
  winner: VendorResult;
  baseline: BaselineResult;
}

export const ResultsWinnerHero = ({ winner, baseline }: ResultsWinnerHeroProps) => {
  const { t } = useTranslation('pages');
  const shouldReduceMotion = useReducedMotion();
  const vendorColor = getVendorColor(winner.vendor_id);
  const delta = winner.revenue_krw - baseline.revenue_krw;

  const subMetrics = [
    { label: 'SOH', value: `${winner.soh_percent.toFixed(1)}%` },
    {
      label: t('results.winner.payback', { defaultValue: 'Payback' }),
      value: `${formatNumber(winner.payback_years, 1)} ${t('results.table.years', { defaultValue: 'yr' })}`,
    },
    { label: 'NPV', value: formatKRW(winner.npv_krw, 0) },
  ];

  return (
    <Box
      position="relative"
      minH={{ base: '50vh', md: '55vh' }}
      display="flex"
      alignItems="center"
      overflow="hidden"
    >
      {/* Background */}
      <Box
        position="absolute"
        inset={0}
        bgGradient={`radial(ellipse at 50% 30%, ${vendorColor.glow}, transparent 70%)`}
        opacity={0.3}
      />
      <DotPattern color="rgba(255, 215, 0, 0.15)" opacity={0.2} spacing={30} size={1} />

      <Container maxW="7xl" px={{ base: 4, md: 6, lg: 8 }} position="relative" zIndex={1}>
        <VStack spacing={8} align="center" textAlign="center">
          {/* Badge */}
          <MotionBadge
            initial={shouldReduceMotion ? {} : { scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            // @ts-ignore
            transition={{ type: 'spring', stiffness: 300, damping: 20, delay: 0.2 }}
            bg="solar.gold"
            color="black"
            fontSize="sm"
            px={6}
            py={2}
            borderRadius="0"
            textTransform="uppercase"
            fontWeight="bold"
            letterSpacing="widest"
          >
            OPTIMAL BATTERY
          </MotionBadge>

          {/* Vendor Name - blur to sharp */}
          <MotionText
            initial={shouldReduceMotion ? {} : { filter: 'blur(10px)', opacity: 0 }}
            animate={{ filter: 'blur(0px)', opacity: 1 }}
            // @ts-ignore
            transition={{ duration: 0.8, delay: 0.4 }}
            fontSize={{ base: '4xl', md: '6xl', lg: '8xl' }}
            fontWeight="800"
            color="white"
            letterSpacing="tight"
            lineHeight="1"
          >
            {winner.vendor_name}
          </MotionText>

          {/* Primary metric with count-up */}
          <MotionBox
            initial={shouldReduceMotion ? {} : { opacity: 0 }}
            animate={{ opacity: 1 }}
            // @ts-ignore
            transition={{ delay: 0.6, duration: 0.4 }}
          >
            <VStack spacing={1}>
              <CountUpNumber
                end={winner.revenue_krw}
                duration={1.5}
                decimals={0}
                prefix="₩"
                delay={0.6}
                fontSize={{ base: '3xl', md: '4xl' }}
                fontWeight="bold"
                color="white"
                fontFamily="mono"
              />
              <Text
                fontSize="sm"
                color="spacex.textGray"
                textTransform="uppercase"
                letterSpacing="wider"
              >
                {t('results.winner.totalRevenue')}
              </Text>
              {delta > 0 && (
                <Text fontSize="sm" color="battery.excellent" fontWeight="medium">
                  +{formatKRW(delta, 0)} vs no ESS
                </Text>
              )}
            </VStack>
          </MotionBox>

          {/* Sub metrics */}
          <HStack spacing={{ base: 6, md: 12 }} flexWrap="wrap" justify="center">
            {subMetrics.map((m, i) => (
              <MotionBox
                key={m.label}
                initial={shouldReduceMotion ? {} : { opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                // @ts-ignore
                transition={{ delay: 1.0 + i * 0.15, duration: 0.5 }}
                textAlign="center"
              >
                <Text fontSize={{ base: 'xl', md: '2xl' }} fontWeight="bold" color="white">
                  {m.value}
                </Text>
                <Text
                  fontSize="xs"
                  color="spacex.textGray"
                  textTransform="uppercase"
                  letterSpacing="wider"
                >
                  {m.label}
                </Text>
              </MotionBox>
            ))}
          </HStack>

          {/* Scroll indicator */}
          <MotionBox
            initial={shouldReduceMotion ? {} : { opacity: 0 }}
            animate={{ opacity: 0.6 }}
            // @ts-ignore
            transition={{ delay: 1.5 }}
            mt={4}
          >
            <VStack spacing={1}>
              <Text fontSize="xs" color="spacex.textGray" letterSpacing="wider" textTransform="uppercase">
                {t('results.hero.scrollHint', { defaultValue: 'scroll to discover why' })}
              </Text>
              <motion.div
                animate={shouldReduceMotion ? {} : { y: [0, 8, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
              >
                <ChevronDownIcon boxSize={6} color="spacex.textGray" />
              </motion.div>
            </VStack>
          </MotionBox>
        </VStack>
      </Container>

      {/* Bottom glow border */}
      <Box
        position="absolute"
        bottom={0}
        left={0}
        right={0}
        h="1px"
        bgGradient={`linear(to-r, transparent, ${vendorColor.primary}, transparent)`}
        opacity={0.5}
      />
    </Box>
  );
};
