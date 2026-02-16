/**
 * Architecture page - Technical deep-dive
 */
import {
  Box,
  Heading,
  Text,
  VStack,
  Code,
  Divider,
  List,
  ListItem,
  ListIcon,
  useColorModeValue,
} from '@chakra-ui/react';
import { CheckCircleIcon } from '@chakra-ui/icons';
import { useTranslation } from 'react-i18next';
import { Section } from '../components/layout/Section';

export const Architecture = () => {
  const { t } = useTranslation('pages');
  const codeBg = useColorModeValue('gray.100', 'gray.700');
  const accentColor = useColorModeValue('blue.500', 'blue.300');

  return (
    <Box>
      <Section>
        <VStack spacing={8} align="start" maxW="5xl" mx="auto">
          <Box>
            <Heading size="xl" mb={4}>
              {t('architecture.title')}
            </Heading>
            <Text fontSize="lg" color="gray.600">
              {t('architecture.subtitle')}
            </Text>
          </Box>

          <Divider />

          {/* Data Flow */}
          <Box w="full">
            <Heading size="lg" mb={4}>
              {t('architecture.dataFlowTitle')}
            </Heading>
            <Text mb={4}>
              {t('architecture.dataFlowText')}
            </Text>
            <Code display="block" p={4} bg={codeBg} borderRadius="md" whiteSpace="pre" fontSize="sm">
{`Data Loader → LSTM Predictor → Battery Simulator → Economics Calculator

1. Data Loader (src/data_loader.py)
   - Auto-detects CSV/Excel files
   - Merges weather + generation + SMP data
   - Creates 24-hour sequences for LSTM

2. LSTM Predictor (src/model.py)
   - Input: [temp, rain, wind, humidity, sun hours, solar radiation, clouds, generation]
   - Architecture: LSTM(8→64→1) with LayerNorm + Dropout
   - Output: Predicted solar generation (kW)

3. Battery Simulator (src/battery.py)
   - Physics constraints: C-rate, SoC limits, temperature effects
   - Decision logic: Charge/Discharge based on price thresholds
   - SOH tracking: Cycle counting + degradation modeling

4. Economics (src/economics.py)
   - CAPEX: battery cost + installation + BMS
   - OPEX: O&M + insurance
   - Metrics: ROI, payback period, NPV (5% discount)`}
            </Code>
          </Box>

          <Divider />

          {/* LSTM Architecture */}
          <Box w="full">
            <Heading size="lg" mb={4}>
              LSTM Neural Network
            </Heading>
            <List spacing={3}>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>Input Size:</strong> 8 features (weather + solar radiation)
              </ListItem>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>Hidden Size:</strong> 64 neurons
              </ListItem>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>Sequence Length:</strong> 24 hours (look-back window)
              </ListItem>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>Regularization:</strong> LayerNorm + Dropout (0.2)
              </ListItem>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>Training:</strong> AdamW optimizer, Early Stopping (patience=15)
              </ListItem>
            </List>
            <Text mt={4} fontSize="sm" color="gray.600">
              Model achieves MAE: 53.78 kW, RMSE: 92.56 kW, MAPE: 60.08% on test set
            </Text>
          </Box>

          <Divider />

          {/* Battery Physics */}
          <Box w="full">
            <Heading size="lg" mb={4}>
              Battery Physics Model
            </Heading>
            <Text mb={4}>Key equations governing battery behavior:</Text>
            <Code display="block" p={4} bg={codeBg} borderRadius="md" whiteSpace="pre" fontSize="sm">
{`1. Maximum Power (C-rate constraint):
   P_max = Capacity (kWh) × C-rate

2. Efficiency (temperature-adjusted):
   eff_one_way = sqrt(eff_round_trip)
   eff_adjusted = eff_one_way × temp_factor(T)

3. State of Health (degradation):
   SOH_new = SOH_old - (cycles × degradation_rate)
   cycles = throughput_kwh / (2 × capacity_kwh)

4. Energy Balance:
   E_stored(t+1) = E_stored(t) + eff × P_charge × dt - P_discharge/eff × dt`}
            </Code>
          </Box>

          <Divider />

          {/* Vendor Specifications */}
          <Box w="full">
            <Heading size="lg" mb={4}>
              Vendor Specifications
            </Heading>
            <Code display="block" p={4} bg={codeBg} borderRadius="md" whiteSpace="pre" fontSize="sm">
{`LG Energy Solution (NCM):
  - C-rate: 2.0 (fastest charging)
  - Efficiency: 98.0%
  - Degradation: 0.00008/cycle (best longevity)
  - Cost: $180/kWh

Samsung SDI (NCA):
  - C-rate: 1.8 (balanced)
  - Efficiency: 98.5% (highest)
  - Degradation: 0.00009/cycle
  - Cost: $175/kWh (cheapest)

Tesla In-house (4680):
  - C-rate: 1.5 (conservative)
  - Efficiency: 97.0%
  - Degradation: 0.0001/cycle (worst)
  - Cost: $200/kWh (most expensive)`}
            </Code>
          </Box>

          <Divider />

          {/* Web Stack */}
          <Box w="full">
            <Heading size="lg" mb={4}>
              Web Application Stack
            </Heading>
            <List spacing={3}>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>Backend:</strong> FastAPI + Uvicorn (async Python web framework)
              </ListItem>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>Frontend:</strong> React 18 + TypeScript + Vite (fast build tool)
              </ListItem>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>UI Library:</strong> Chakra UI (accessible, responsive components)
              </ListItem>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>Charts:</strong> Recharts (React-friendly visualization library)
              </ListItem>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>API Communication:</strong> Axios (HTTP client with TypeScript types)
              </ListItem>
              <ListItem>
                <ListIcon as={CheckCircleIcon} color={accentColor} />
                <strong>Deployment:</strong> Docker containers (backend + frontend served together)
              </ListItem>
            </List>
          </Box>
        </VStack>
      </Section>
    </Box>
  );
};
