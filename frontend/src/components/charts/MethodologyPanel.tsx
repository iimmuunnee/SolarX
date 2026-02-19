/**
 * MethodologyPanel - Inline collapsible panel explaining calculation methodology
 */
import { useState } from 'react';
import {
  Box,
  Button,
  Collapse,
  Text,
  Code,
  VStack,
  HStack,
  Divider,
  SimpleGrid,
} from '@chakra-ui/react';
import { ChevronDownIcon, ChevronUpIcon } from '@chakra-ui/icons';
import type { VendorResult } from '../../types/simulation';

interface MethodologyPanelProps {
  capacityKwh: number;
  vendors: VendorResult[];
}

// BloombergNEF 2025 기준 단가 ($/kWh)
const UNIT_COST_USD: Record<string, number> = {
  lg: 142,
  samsung: 148,
  tesla: 135,
};
const INSTALL_FACTOR = 1.15;
const USD_TO_KRW = 1320;

function formatKrw(value: number): string {
  if (Math.abs(value) >= 1e8) {
    return `₩${(value / 1e8).toFixed(1)}억`;
  }
  if (Math.abs(value) >= 1e4) {
    return `₩${(value / 1e4).toFixed(0)}만`;
  }
  return `₩${value.toFixed(0)}`;
}

export const MethodologyPanel = ({ capacityKwh, vendors }: MethodologyPanelProps) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Box>
      <Button
        size="xs"
        variant="ghost"
        color="spacex.textGray"
        rightIcon={isOpen ? <ChevronUpIcon /> : <ChevronDownIcon />}
        onClick={() => setIsOpen(!isOpen)}
        _hover={{ color: 'white', bg: 'whiteAlpha.100' }}
      >
        계산 근거
      </Button>

      <Collapse in={isOpen} animateOpacity>
        <Box
          maxH="420px"
          overflowY="auto"
          p={5}
          bg="spacex.darkGray"
          border="1px solid"
          borderColor="spacex.borderGray"
          borderRadius="md"
          mt={2}
          mb={4}
          fontSize="sm"
          color="gray.300"
        >
          <VStack align="stretch" spacing={5}>

            {/* 1. 수익 계산 */}
            <Box>
              <Text fontWeight="bold" color="white" mb={2}>
                1. 수익(Revenue) 계산 공식
              </Text>
              <Code
                display="block"
                p={3}
                bg="blackAlpha.500"
                borderRadius="sm"
                fontSize="xs"
                color="cyan.300"
                whiteSpace="pre"
              >
                {`profit += (gen_kw + battery_trade_kw) × price × dt_hours`}
              </Code>
              <VStack align="stretch" spacing={1} mt={3} pl={2}>
                <Text>• 테스트 데이터(실제 기상 기록) 기반 시뮬레이션</Text>
                <Text>• LSTM 예측값 → 충방전 의사결정 → 실제 SMP 가격 적용</Text>
                <Text>• <Code fontSize="xs">battery_trade_kw</Code>: 양수=방전(판매), 음수=충전(구매)</Text>
                <Text>• <Code fontSize="xs">dt_hours = 1.0</Code> (시간별 데이터 기준)</Text>
              </VStack>
            </Box>

            <Divider borderColor="spacex.borderGray" />

            {/* 2. CAPEX */}
            <Box>
              <Text fontWeight="bold" color="white" mb={2}>
                2. CAPEX 계산 근거
              </Text>
              <Code
                display="block"
                p={3}
                bg="blackAlpha.500"
                borderRadius="sm"
                fontSize="xs"
                color="cyan.300"
                whiteSpace="pre"
              >
                {`CAPEX = 용량(kWh) × 단가($/kWh) × ${INSTALL_FACTOR}(설치비) × ${USD_TO_KRW}(환율)`}
              </Code>
              <Text mt={2} fontSize="xs" color="gray.500">
                출처: BloombergNEF 2025 Battery Price Survey
              </Text>

              {/* 실시간 계산 테이블 */}
              <Box mt={3} border="1px solid" borderColor="spacex.borderGray" borderRadius="sm" overflow="hidden">
                <SimpleGrid columns={4} bg="blackAlpha.400" px={3} py={2}>
                  <Text fontWeight="bold" color="gray.400" fontSize="xs">벤더</Text>
                  <Text fontWeight="bold" color="gray.400" fontSize="xs">단가($/kWh)</Text>
                  <Text fontWeight="bold" color="gray.400" fontSize="xs">용량</Text>
                  <Text fontWeight="bold" color="gray.400" fontSize="xs">CAPEX</Text>
                </SimpleGrid>
                {vendors.map((v) => {
                  const unitCost = UNIT_COST_USD[v.vendor_id] ?? 140;
                  const capexCalc = capacityKwh * unitCost * INSTALL_FACTOR * USD_TO_KRW;
                  return (
                    <SimpleGrid
                      key={v.vendor_id}
                      columns={4}
                      px={3}
                      py={2}
                      borderTop="1px solid"
                      borderColor="spacex.borderGray"
                    >
                      <Text fontSize="xs">{v.vendor_name}</Text>
                      <Text fontSize="xs">${unitCost}</Text>
                      <Text fontSize="xs">{capacityKwh.toLocaleString()} kWh</Text>
                      <Text fontSize="xs" color="cyan.300">{formatKrw(capexCalc)}</Text>
                    </SimpleGrid>
                  );
                })}
              </Box>
            </Box>

            <Divider borderColor="spacex.borderGray" />

            {/* 3. ROI / NPV */}
            <Box>
              <Text fontWeight="bold" color="white" mb={2}>
                3. ROI / NPV (10년 프로젝션)
              </Text>
              <Code
                display="block"
                p={3}
                bg="blackAlpha.500"
                borderRadius="sm"
                fontSize="xs"
                color="cyan.300"
                whiteSpace="pre"
              >
                {`연간수익 = 시뮬수익 / 시뮬기간(년)
10년수익 = 연간수익 × 10

ROI% = (10년수익 - CAPEX - O&M×10) / CAPEX × 100

NPV = -CAPEX + Σ (연간수익 - O&M) / 1.05^t  [t=1..10]`}
              </Code>

              <HStack mt={3} spacing={2} align="start">
                <Box
                  px={2}
                  py={1}
                  bg="orange.900"
                  border="1px solid"
                  borderColor="orange.700"
                  borderRadius="sm"
                  flexShrink={0}
                >
                  <Text fontSize="xs" color="orange.300" fontWeight="bold">한계</Text>
                </Box>
                <VStack align="stretch" spacing={1} fontSize="xs" color="gray.400">
                  <Text>• 선형 프로젝션 — 실제 수익은 연도별로 변동</Text>
                  <Text>• O&M 비율이 낮게 설정됨 (CAPEX의 1~2%/년)</Text>
                  <Text>• SMP 가격 고정 가정 (미래 가격 변동 미반영)</Text>
                  <Text>• 초기 손실 구간에서 NPV가 음수일 수 있음</Text>
                </VStack>
              </HStack>
            </Box>

          </VStack>
        </Box>
      </Collapse>
    </Box>
  );
};
