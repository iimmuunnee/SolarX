/**
 * ResultsInsightPanel - AI-generated data insights
 */
import { Box, Container, Heading, VStack } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import type { VendorResult, BaselineResult } from '../../types/simulation';
import { useResultsInsights } from '../../hooks/useResultsInsights';
import { InsightCard } from '../../components/feedback/InsightCard';
import { staggerContainer } from '../../hooks/useInViewAnimation';

const MotionBox = motion(Box);
const MotionVStack = motion(VStack);

const slideInChild = {
  hidden: { opacity: 0, x: -30 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.5, ease: 'easeOut' as const },
  },
};

interface ResultsInsightPanelProps {
  vendors: VendorResult[];
  baseline: BaselineResult;
}

export const ResultsInsightPanel = ({ vendors, baseline }: ResultsInsightPanelProps) => {
  const { t } = useTranslation('pages');
  const insights = useResultsInsights(vendors, baseline);

  if (insights.length === 0) return null;

  return (
    <Box py={16}>
      <Container maxW="7xl" px={{ base: 4, md: 6, lg: 8 }}>
        <Heading size="lg" mb={6} color="white">
          {t('results.insights.title', { defaultValue: 'AI Insights' })}
        </Heading>

        <MotionVStack
          spacing={4}
          align="stretch"
          variants={staggerContainer(0.1)}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: '-50px' }}
        >
          {insights.map((insight) => (
            <MotionBox key={insight.id} variants={slideInChild}>
              <InsightCard text={insight.text} />
            </MotionBox>
          ))}
        </MotionVStack>
      </Container>
    </Box>
  );
};
