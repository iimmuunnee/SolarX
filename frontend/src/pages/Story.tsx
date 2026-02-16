/**
 * Story page - Project narrative
 */
import {
  Box,
  Heading,
  Text,
  VStack,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  List,
  ListItem,
  ListIcon,
  useColorModeValue,
} from '@chakra-ui/react';
import { CheckCircleIcon } from '@chakra-ui/icons';
import { useTranslation } from 'react-i18next';
import { Section } from '../components/layout/Section';

export const Story = () => {
  const { t } = useTranslation('pages');
  const accentColor = useColorModeValue('blue.500', 'blue.300');

  return (
    <Box>
      {/* Problem Statement */}
      <Section bg={useColorModeValue('gray.50', 'gray.900')}>
        <VStack spacing={6} align="start" maxW="4xl" mx="auto">
          <Heading size="xl">{t('story.problemTitle')}</Heading>
          <Text fontSize="lg" lineHeight="tall" dangerouslySetInnerHTML={{ __html: t('story.problemText1') }} />
          <Text fontSize="lg" lineHeight="tall">
            {t('story.problemText2')}
          </Text>
        </VStack>
      </Section>

      {/* Solution Approach */}
      <Section>
        <VStack spacing={6} align="start" maxW="4xl" mx="auto">
          <Heading size="xl">{t('story.solutionTitle')}</Heading>
          <Text fontSize="lg" lineHeight="tall">
            {t('story.solutionIntro')}
          </Text>
          <List spacing={4} fontSize="lg">
            <ListItem>
              <ListIcon as={CheckCircleIcon} color={accentColor} />
              <strong>{t('story.solutionItem1Title')}:</strong> {t('story.solutionItem1Desc')}
            </ListItem>
            <ListItem>
              <ListIcon as={CheckCircleIcon} color={accentColor} />
              <strong>{t('story.solutionItem2Title')}:</strong> {t('story.solutionItem2Desc')}
            </ListItem>
            <ListItem>
              <ListIcon as={CheckCircleIcon} color={accentColor} />
              <strong>{t('story.solutionItem3Title')}:</strong> {t('story.solutionItem3Desc')}
            </ListItem>
          </List>
        </VStack>
      </Section>

      {/* Project Evolution */}
      <Section bg={useColorModeValue('gray.50', 'gray.900')}>
        <VStack spacing={6} align="start" maxW="4xl" mx="auto">
          <Heading size="xl">{t('story.evolutionTitle')}</Heading>
          <Accordion allowToggle w="full">
            <AccordionItem>
              <h2>
                <AccordionButton>
                  <Box flex="1" textAlign="left" fontWeight="bold">
                    {t('story.v1Title')}
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
              </h2>
              <AccordionPanel pb={4}>
                <VStack align="start" spacing={2}>
                  <Text><strong>{t('story.v1Achievement')}:</strong> {t('story.v1AchievementText')}</Text>
                  <Text><strong>{t('story.v1Learning')}:</strong> {t('story.v1LearningText')}</Text>
                  <Text><strong>{t('story.v1Metrics')}:</strong> {t('story.v1MetricsText')}</Text>
                </VStack>
              </AccordionPanel>
            </AccordionItem>

            <AccordionItem>
              <h2>
                <AccordionButton>
                  <Box flex="1" textAlign="left" fontWeight="bold">
                    {t('story.v2Title')}
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
              </h2>
              <AccordionPanel pb={4}>
                <VStack align="start" spacing={2}>
                  <Text><strong>{t('story.v2Achievement')}:</strong> {t('story.v2AchievementText')}</Text>
                  <Text><strong>{t('story.v2Learning')}:</strong> {t('story.v2LearningText')}</Text>
                  <Text><strong>{t('story.v2Metrics')}:</strong> {t('story.v2MetricsText')}</Text>
                </VStack>
              </AccordionPanel>
            </AccordionItem>

            <AccordionItem>
              <h2>
                <AccordionButton>
                  <Box flex="1" textAlign="left" fontWeight="bold">
                    {t('story.v3Title')}
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
              </h2>
              <AccordionPanel pb={4}>
                <VStack align="start" spacing={2}>
                  <Text><strong>{t('story.v3Achievement')}:</strong> {t('story.v3AchievementText')}</Text>
                  <Text><strong>{t('story.v3Learning')}:</strong> {t('story.v3LearningText')}</Text>
                  <Text><strong>{t('story.v3Metrics')}:</strong> {t('story.v3MetricsText')}</Text>
                </VStack>
              </AccordionPanel>
            </AccordionItem>

            <AccordionItem>
              <h2>
                <AccordionButton>
                  <Box flex="1" textAlign="left" fontWeight="bold">
                    {t('story.v4Title')}
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
              </h2>
              <AccordionPanel pb={4}>
                <VStack align="start" spacing={2}>
                  <Text><strong>{t('story.v4Achievement')}:</strong> {t('story.v4AchievementText')}</Text>
                  <Text><strong>{t('story.v4Learning')}:</strong> {t('story.v4LearningText')}</Text>
                  <Text><strong>{t('story.v4Metrics')}:</strong> {t('story.v4MetricsText')}</Text>
                </VStack>
              </AccordionPanel>
            </AccordionItem>

            <AccordionItem>
              <h2>
                <AccordionButton>
                  <Box flex="1" textAlign="left" fontWeight="bold">
                    {t('story.v5Title')}
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
              </h2>
              <AccordionPanel pb={4}>
                <VStack align="start" spacing={2}>
                  <Text><strong>{t('story.v5Achievement')}:</strong> {t('story.v5AchievementText')}</Text>
                  <Text><strong>{t('story.v5Learning')}:</strong> {t('story.v5LearningText')}</Text>
                  <Text><strong>{t('story.v5Metrics')}:</strong> {t('story.v5MetricsText')}</Text>
                </VStack>
              </AccordionPanel>
            </AccordionItem>
          </Accordion>
        </VStack>
      </Section>

      {/* Key Insights */}
      <Section>
        <VStack spacing={6} align="start" maxW="4xl" mx="auto">
          <Heading size="xl">{t('story.keyInsights')}</Heading>
          <List spacing={4} fontSize="lg">
            <ListItem>
              <ListIcon as={CheckCircleIcon} color={accentColor} />
              <strong>{t('story.insight1Title')}:</strong> {t('story.insight1Text')}
            </ListItem>
            <ListItem>
              <ListIcon as={CheckCircleIcon} color={accentColor} />
              <strong>{t('story.insight2Title')}:</strong> {t('story.insight2Text')}
            </ListItem>
            <ListItem>
              <ListIcon as={CheckCircleIcon} color={accentColor} />
              <strong>{t('story.insight3Title')}:</strong> {t('story.insight3Text')}
            </ListItem>
          </List>
        </VStack>
      </Section>
    </Box>
  );
};
