/**
 * ResultsEconomicTable - Sortable expanded economic comparison table
 */
import {
  Box, Container, Heading, Table, Thead, Tbody, Tr, Th, Td,
  Badge, HStack, Icon,
} from '@chakra-ui/react';
import { TriangleUpIcon, TriangleDownIcon } from '@chakra-ui/icons';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import type { VendorResult, BaselineResult } from '../../types/simulation';
import { useSortableData } from '../../hooks/useSortableData';
import { getVendorColor } from '../../utils/vendorColors';
import { formatKRW, formatPercent, formatNumber, formatEnergy } from '../../utils/formatters';
import { useInViewAnimation, staggerContainer, fadeUpChild } from '../../hooks/useInViewAnimation';
import { MethodologyPanel } from '../../components/charts/MethodologyPanel';

const MotionBox = motion(Box);
const MotionTr = motion(Tr);

interface ResultsEconomicTableProps {
  vendors: VendorResult[];
  baseline: BaselineResult;
  winnerId: string;
}

interface SortableVendor extends Record<string, unknown> {
  vendor_id: string;
  vendor_name: string;
  revenue_krw: number;
  soh_percent: number;
  cycle_count: number;
  roi_percent: number;
  payback_years: number;
  npv_krw: number;
  capex_krw: number;
  opex_annual_krw: number;
  throughput_kwh: number;
}

const columns: { key: string; labelKey: string; format: (v: SortableVendor) => string; isNumeric: boolean }[] = [
  { key: 'vendor_name', labelKey: 'vendor', format: (v) => v.vendor_name, isNumeric: false },
  { key: 'revenue_krw', labelKey: 'revenue', format: (v) => formatKRW(v.revenue_krw, 0), isNumeric: true },
  { key: 'soh_percent', labelKey: 'soh', format: (v) => formatPercent(v.soh_percent, 2), isNumeric: true },
  { key: 'cycle_count', labelKey: 'cycles', format: (v) => formatNumber(v.cycle_count, 1), isNumeric: true },
  { key: 'roi_percent', labelKey: 'roi', format: (v) => formatPercent(v.roi_percent, 1), isNumeric: true },
  { key: 'payback_years', labelKey: 'payback', format: (v) => `${formatNumber(v.payback_years, 1)}y`, isNumeric: true },
  { key: 'npv_krw', labelKey: 'npv', format: (v) => formatKRW(v.npv_krw, 0), isNumeric: true },
  { key: 'capex_krw', labelKey: 'capex', format: (v) => formatKRW(v.capex_krw, 0), isNumeric: true },
  { key: 'opex_annual_krw', labelKey: 'opex', format: (v) => formatKRW(v.opex_annual_krw, 0), isNumeric: true },
  { key: 'throughput_kwh', labelKey: 'throughput', format: (v) => formatEnergy(v.throughput_kwh, 'kWh'), isNumeric: true },
];

export const ResultsEconomicTable = ({ vendors, baseline, winnerId }: ResultsEconomicTableProps) => {
  const { t } = useTranslation('pages');
  const { variants, viewportOptions } = useInViewAnimation();

  const sortableVendors: SortableVendor[] = vendors.map((v) => ({
    vendor_id: v.vendor_id,
    vendor_name: v.vendor_name,
    revenue_krw: v.revenue_krw,
    soh_percent: v.soh_percent,
    cycle_count: v.cycle_count,
    roi_percent: v.roi_percent,
    payback_years: v.payback_years,
    npv_krw: v.npv_krw,
    capex_krw: v.capex_krw,
    opex_annual_krw: v.opex_annual_krw,
    throughput_kwh: v.throughput_kwh,
  }));

  const { sortedItems, sortConfig, requestSort } = useSortableData(sortableVendors);

  // Find best values for highlighting
  const bestValues: Record<string, number> = {};
  columns.forEach((col) => {
    if (col.isNumeric && col.key !== 'payback_years' && col.key !== 'capex_krw' && col.key !== 'opex_annual_krw') {
      bestValues[col.key] = Math.max(...vendors.map((v) => v[col.key as keyof VendorResult] as number));
    }
    if (col.key === 'payback_years') {
      bestValues[col.key] = Math.min(...vendors.map((v) => v.payback_years));
    }
  });

  const labelMap: Record<string, string> = {
    vendor: t('results.table.vendor', { defaultValue: 'Vendor' }),
    revenue: t('results.table.revenue', { defaultValue: 'Revenue' }),
    soh: t('results.table.soh', { defaultValue: 'SOH' }),
    cycles: t('results.table.cycles', { defaultValue: 'Cycles' }),
    roi: 'ROI',
    payback: t('results.table.payback', { defaultValue: 'Payback' }),
    npv: 'NPV',
    capex: 'CAPEX',
    opex: 'OPEX',
    throughput: t('results.table.throughput', { defaultValue: 'Throughput' }),
  };

  return (
    <Box py={16} bg="rgba(0,0,0,0.2)">
      <Container maxW="7xl" px={{ base: 4, md: 6, lg: 8 }}>
        <MotionBox
          variants={variants}
          initial="hidden"
          whileInView="visible"
          viewport={viewportOptions}
        >
          <Heading size="lg" mb={4} color="white">
            {t('results.vendorComparison')}
          </Heading>

          <MethodologyPanel capacityKwh={2740} vendors={vendors} />

          <Box
            overflowX="auto"
            mt={4}
            border="1px solid"
            borderColor="spacex.borderGray"
            bg="spacex.darkGray"
          >
            <Table variant="simple" size="sm">
              <Thead position="sticky" top={0} bg="spacex.darkGray" zIndex={1}>
                <Tr borderBottom="2px solid" borderColor="white">
                  {columns.map((col) => (
                    <Th
                      key={col.key}
                      color="white"
                      textTransform="uppercase"
                      fontSize="xs"
                      letterSpacing="wider"
                      isNumeric={col.isNumeric}
                      cursor="pointer"
                      onClick={() => requestSort(col.key)}
                      _hover={{ color: 'solar.gold' }}
                      whiteSpace="nowrap"
                      userSelect="none"
                    >
                      <HStack spacing={1} justify={col.isNumeric ? 'flex-end' : 'flex-start'}>
                        <span>{labelMap[col.labelKey] || col.labelKey}</span>
                        {sortConfig?.key === col.key && (
                          <Icon
                            as={sortConfig.direction === 'asc' ? TriangleUpIcon : TriangleDownIcon}
                            boxSize={3}
                            color="solar.gold"
                          />
                        )}
                      </HStack>
                    </Th>
                  ))}
                </Tr>
              </Thead>
              <Tbody
                as={motion.tbody}
                variants={staggerContainer(0.05)}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
              >
                {sortedItems.map((vendor) => {
                  const isWinner = vendor.vendor_id === winnerId;
                  const vendorColor = getVendorColor(vendor.vendor_id as string);

                  return (
                    <MotionTr
                      key={vendor.vendor_id as string}
                      variants={fadeUpChild}
                      bg={isWinner ? `${vendorColor.light}` : undefined}
                      borderBottom="1px solid"
                      borderColor="gray.700"
                      _hover={{ bg: `${vendorColor.light}` }}
                      // @ts-ignore - Chakra/Framer transition conflict
                      transition="background 0.2s"
                      position="relative"
                    >
                      {columns.map((col) => {
                        const isVendorCol = col.key === 'vendor_name';
                        const numVal = vendor[col.key];
                        const isBestVal = col.isNumeric && bestValues[col.key] !== undefined &&
                          ((col.key === 'payback_years' && numVal === bestValues[col.key]) ||
                           (col.key !== 'payback_years' && col.key !== 'capex_krw' && col.key !== 'opex_annual_krw' && numVal === bestValues[col.key]));

                        return (
                          <Td
                            key={col.key}
                            isNumeric={col.isNumeric}
                            color={isBestVal ? vendorColor.primary : 'gray.300'}
                            fontWeight={isVendorCol && isWinner ? 'bold' : isBestVal ? 'semibold' : 'normal'}
                            whiteSpace="nowrap"
                            fontSize="sm"
                          >
                            {col.format(vendor)}
                            {isVendorCol && isWinner && (
                              <Badge
                                ml={2}
                                bg="solar.gold"
                                color="black"
                                fontSize="xs"
                                boxShadow="0 0 8px rgba(255, 215, 0, 0.5)"
                              >
                                {t('results.table.best', { defaultValue: 'BEST' })}
                              </Badge>
                            )}
                          </Td>
                        );
                      })}
                    </MotionTr>
                  );
                })}
                {/* Baseline row */}
                <Tr
                  bg="rgba(0, 102, 255, 0.05)"
                  borderTop="2px solid"
                  borderColor="spacex.borderGray"
                >
                  <Td fontWeight="bold" color="gray.400" fontSize="sm">
                    {t('results.table.baseline', { defaultValue: 'Baseline (No ESS)' })}
                  </Td>
                  <Td isNumeric color="gray.500" fontSize="sm">{formatKRW(baseline.revenue_krw, 0)}</Td>
                  <Td colSpan={8} isNumeric color="gray.600" fontSize="sm">-</Td>
                </Tr>
              </Tbody>
            </Table>
          </Box>
        </MotionBox>
      </Container>
    </Box>
  );
};
