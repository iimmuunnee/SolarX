/**
 * Navigation bar component
 */
import {
  Box,
  Flex,
  HStack,
  Link,
  IconButton,
  useDisclosure,
  Stack,
  Text,
} from '@chakra-ui/react';
import { HamburgerIcon, CloseIcon } from '@chakra-ui/icons';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { LanguageSwitcher } from './LanguageSwitcher';

export const Navbar = () => {
  const { isOpen, onToggle } = useDisclosure();
  const location = useLocation();
  const { t } = useTranslation('common');

  const NAV_ITEMS = [
    { name: t('nav.home'), path: '/' },
    { name: t('nav.story'), path: '/story' },
    { name: t('nav.demo'), path: '/demo' },
    { name: t('nav.architecture'), path: '/architecture' },
    { name: t('nav.results'), path: '/results' },
  ];

  return (
    <Box
      as="nav"
      bg="spacex.black"
      borderBottom="1px"
      borderColor="spacex.borderGray"
      px={4}
      position="sticky"
      top={0}
      zIndex={1000}
    >
      <Flex h={16} alignItems="center" maxW="7xl" mx="auto" position="relative">
        {/* Logo - always left */}
        <Link as={RouterLink} to="/" _hover={{ textDecoration: 'none' }}>
          <Text
            fontWeight="800"
            fontSize="xl"
            color="white"
            fontFamily="heading"
            letterSpacing="wide"
          >
            SOLARX
          </Text>
        </Link>

        {/* Desktop Navigation - absolutely centered, unaffected by logo or lang switcher width */}
        <HStack
          as="nav"
          spacing={20}
          display={{ base: 'none', md: 'flex' }}
          position="absolute"
          left="50%"
          transform="translateX(-50%)"
        >
          {NAV_ITEMS.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                as={RouterLink}
                to={item.path}
                px={0}
                py={2}
                position="relative"
                fontWeight={isActive ? 'bold' : 'normal'}
                color={isActive ? 'white' : 'spacex.textGray'}
                textTransform="uppercase"
                fontSize="sm"
                letterSpacing="widest"
                transition="color 0.3s ease"
                borderBottom={isActive ? '2px solid' : 'none'}
                borderColor="solar.gold"
                _hover={{
                  textDecoration: 'none',
                  color: 'white',
                }}
              >
                {item.name}
              </Link>
            );
          })}
        </HStack>

        {/* Language switcher - always right */}
        <Box ml="auto" display={{ base: 'none', md: 'block' }}>
          <LanguageSwitcher />
        </Box>

        {/* Mobile menu toggle */}
        <IconButton
          size="md"
          icon={isOpen ? <CloseIcon /> : <HamburgerIcon />}
          aria-label="Toggle navigation"
          display={{ md: 'none' }}
          ml="auto"
          onClick={onToggle}
        />
      </Flex>

      {/* Mobile Navigation */}
      {isOpen && (
        <Box pb={4} display={{ md: 'none' }}>
          <Stack as="nav" spacing={4}>
            {NAV_ITEMS.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  as={RouterLink}
                  to={item.path}
                  px={3}
                  py={2}
                  fontWeight={isActive ? 'bold' : 'normal'}
                  color={isActive ? 'white' : 'spacex.textGray'}
                  textTransform="uppercase"
                  fontSize="xs"
                  letterSpacing="widest"
                  borderLeft={isActive ? '2px solid' : 'none'}
                  borderColor="white"
                  _hover={{
                    textDecoration: 'none',
                    color: 'white',
                  }}
                  onClick={onToggle}
                >
                  {item.name}
                </Link>
              );
            })}
          </Stack>
        </Box>
      )}
    </Box>
  );
};
