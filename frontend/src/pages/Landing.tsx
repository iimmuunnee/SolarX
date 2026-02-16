/**
 * Landing page
 */
import {
  Box,
  Heading,
  Text,
  Button,
  VStack,
  HStack,
  SimpleGrid,
  Card,
  CardBody,
  useColorModeValue,
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Section } from '../components/layout/Section';

export const Landing = () => {
  const { t } = useTranslation(['common', 'pages']);
  const gradientBg = useColorModeValue(
    'linear(to-r, blue.400, purple.500)',
    'linear(to-r, blue.600, purple.700)'
  );

  return (
    <Box>
      {/* Hero Section */}
      <Box bgGradient={gradientBg} color="white" py={20}>
        <Section maxW="5xl">
          <VStack spacing={6} textAlign="center">
            <Heading as="h1" size="2xl" fontWeight="bold">
              {t('pages:landing.hero.title')}
            </Heading>
            <Text fontSize="xl" maxW="3xl">
              {t('pages:landing.hero.subtitle')}
            </Text>

            {/* Animated Metrics */}
            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8} mt={8} w="full">
              <VStack>
                <Heading size="xl">3</Heading>
                <Text>{t('pages:landing.hero.metrics.vendors')}</Text>
              </VStack>
              <VStack>
                <Heading size="xl">99.3%</Heading>
                <Text>{t('pages:landing.hero.metrics.soh')}</Text>
              </VStack>
              <VStack>
                <Heading size="xl">12,765%</Heading>
                <Text>{t('pages:landing.hero.metrics.roi')}</Text>
              </VStack>
            </SimpleGrid>

            {/* CTA Buttons */}
            <HStack spacing={4} mt={8}>
              <Button
                as={RouterLink}
                to="/demo"
                size="lg"
                colorScheme="white"
                variant="solid"
                bg="white"
                color="blue.500"
                _hover={{ bg: 'gray.100' }}
              >
                {t('common:buttons.tryDemo')}
              </Button>
              <Button
                as={RouterLink}
                to="/results"
                size="lg"
                variant="outline"
                color="white"
                borderColor="white"
                _hover={{ bg: 'whiteAlpha.200' }}
              >
                {t('common:buttons.viewResults')}
              </Button>
            </HStack>
          </VStack>
        </Section>
      </Box>

      {/* Tech Stack */}
      <Section py={12}>
        <HStack justify="center" spacing={4} flexWrap="wrap">
          <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python" />
          <img src="https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch" alt="PyTorch" />
          <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react" alt="React" />
          <img src="https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi" alt="FastAPI" />
        </HStack>
      </Section>

      {/* Quick Navigation Cards */}
      <Section>
        <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6}>
          <Card as={RouterLink} to="/story" _hover={{ shadow: 'lg', transform: 'translateY(-4px)' }} transition="all 0.2s">
            <CardBody>
              <Heading size="md" mb={2}>{t('pages:landing.cards.story.title')}</Heading>
              <Text fontSize="sm" color="gray.600">
                {t('pages:landing.cards.story.description')}
              </Text>
            </CardBody>
          </Card>

          <Card as={RouterLink} to="/demo" _hover={{ shadow: 'lg', transform: 'translateY(-4px)' }} transition="all 0.2s">
            <CardBody>
              <Heading size="md" mb={2}>{t('pages:landing.cards.demo.title')}</Heading>
              <Text fontSize="sm" color="gray.600">
                {t('pages:landing.cards.demo.description')}
              </Text>
            </CardBody>
          </Card>

          <Card as={RouterLink} to="/architecture" _hover={{ shadow: 'lg', transform: 'translateY(-4px)' }} transition="all 0.2s">
            <CardBody>
              <Heading size="md" mb={2}>{t('pages:landing.cards.architecture.title')}</Heading>
              <Text fontSize="sm" color="gray.600">
                {t('pages:landing.cards.architecture.description')}
              </Text>
            </CardBody>
          </Card>

          <Card as={RouterLink} to="/results" _hover={{ shadow: 'lg', transform: 'translateY(-4px)' }} transition="all 0.2s">
            <CardBody>
              <Heading size="md" mb={2}>{t('pages:landing.cards.results.title')}</Heading>
              <Text fontSize="sm" color="gray.600">
                {t('pages:landing.cards.results.description')}
              </Text>
            </CardBody>
          </Card>
        </SimpleGrid>
      </Section>
    </Box>
  );
};
