/**
 * Landing page
 */
import {
  Box,
  Heading,
  Text,
  Button,
  VStack,
  SimpleGrid,
  Card,
  CardBody,
} from '@chakra-ui/react';
import { ArrowForwardIcon } from '@chakra-ui/icons';
import { Link as RouterLink } from 'react-router-dom';
import { useTranslation, Trans } from 'react-i18next';
import { Section } from '../components/layout/Section';
import { SolarChargingAnimation } from '../components/animations/SolarChargingAnimation';

export const Landing = () => {
  const { t } = useTranslation(['common', 'pages']);

  return (
    <Box>
      {/* Hero Section - 2-column layout */}
      <Box
        minH="100vh"
        display="flex"
        alignItems="center"
        bg="spacex.black"
        color="white"
        position="relative"
      >
        <Box w="100%">
        <Section maxW="7xl" py={20}>
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={12} alignItems="center">
            {/* Left: Text content */}
            <VStack spacing={6} align="flex-start">
              <Heading
                as="h1"
                fontSize={{ base: '5xl', md: '7xl', lg: '8xl' }}
                fontWeight="900"
                color="white"
                lineHeight="0.9"
                letterSpacing="tighter"
                textAlign={{ base: 'center', md: 'left' }}
              >
                {/* Line 1: nowrap ensures it never breaks mid-word */}
                <Box as="span" display="block" whiteSpace="nowrap">
                  {t('pages:landing.hero.titleLine1')}
                </Box>
                <Box as="span" display="block">
                  {t('pages:landing.hero.titleLine2')}
                </Box>
              </Heading>

              <Text
                fontSize={{ base: 'md', md: 'lg' }}
                maxW="500px"
                color="gray.400"
                textAlign={{ base: 'center', md: 'left' }}
              >
                <Trans
                  i18nKey="pages:landing.hero.subtitle"
                  components={{ br: <br /> }}
                />
              </Text>

              <Button
                as={RouterLink}
                to="/demo"
                size="lg"
                variant="spacexSolar"
                rightIcon={<ArrowForwardIcon />}
                mt={4}
                alignSelf={{ base: 'center', md: 'flex-start' }}
              >
                {t('common:buttons.tryDemo')}
              </Button>
            </VStack>

            {/* Right: Solar charging animation (desktop only) */}
            <Box
              display={{ base: 'none', md: 'flex' }}
              justifyContent="center"
              alignItems="center"
            >
              <SolarChargingAnimation />
            </Box>
          </SimpleGrid>
        </Section>
        </Box>
      </Box>

      {/* Metrics Section - Separated */}
      <Section py={20}>
        <SimpleGrid columns={{ base: 1, md: 3 }} spacing={16}>
          {[
            { value: '3', label: t('pages:landing.hero.metrics.vendors') },
            { value: '99.3%', label: t('pages:landing.hero.metrics.soh') },
            { value: '284%', label: t('pages:landing.hero.metrics.roi') },
          ].map((metric, index) => (
            <VStack key={index} spacing={3} align="center">
              <Heading size="3xl" color="solar.gold" fontWeight="800">
                {metric.value}
              </Heading>
              <Text
                color="gray.500"
                textTransform="uppercase"
                fontSize="xs"
                letterSpacing="widest"
              >
                {metric.label}
              </Text>
            </VStack>
          ))}
        </SimpleGrid>
      </Section>

      {/* Quick Navigation Cards - SpaceX Minimal */}
      <Section py={20}>
        <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={8}>
          {[
            { to: '/story', title: t('pages:landing.cards.story.title'), desc: t('pages:landing.cards.story.description') },
            { to: '/demo', title: t('pages:landing.cards.demo.title'), desc: t('pages:landing.cards.demo.description') },
            { to: '/architecture', title: t('pages:landing.cards.architecture.title'), desc: t('pages:landing.cards.architecture.description') },
            { to: '/results', title: t('pages:landing.cards.results.title'), desc: t('pages:landing.cards.results.description') },
          ].map((card) => (
            <Card
              key={card.to}
              as={RouterLink}
              to={card.to}
              bg="spacex.darkGray"
              borderWidth="1px"
              borderColor="spacex.borderGray"
              borderRadius="0"
              transition="all 0.3s ease"
              _hover={{
                transform: 'translateY(-4px)',
                borderColor: 'white',
              }}
            >
              <CardBody>
                <Heading size="md" mb={2} color="white">
                  {card.title}
                </Heading>
                <Text fontSize="sm" color="spacex.textGray">
                  {card.desc}
                </Text>
              </CardBody>
            </Card>
          ))}
        </SimpleGrid>
      </Section>
    </Box>
  );
};
