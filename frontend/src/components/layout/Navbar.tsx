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
  useColorModeValue,
} from '@chakra-ui/react';
import { HamburgerIcon, CloseIcon } from '@chakra-ui/icons';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { LanguageSwitcher } from './LanguageSwitcher';
import { DarkModeToggle } from './DarkModeToggle';

export const Navbar = () => {
  const { isOpen, onToggle } = useDisclosure();
  const location = useLocation();
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
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
      bg={bgColor}
      borderBottom="1px"
      borderColor={borderColor}
      px={4}
      position="sticky"
      top={0}
      zIndex={1000}
    >
      <Flex h={16} alignItems="center" justifyContent="space-between" maxW="7xl" mx="auto">
        {/* Logo */}
        <Box fontWeight="bold" fontSize="xl">
          <Link as={RouterLink} to="/" _hover={{ textDecoration: 'none' }}>
            SolarX
          </Link>
        </Box>

        {/* Desktop Navigation */}
        <HStack as="nav" spacing={8} display={{ base: 'none', md: 'flex' }}>
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.path}
              as={RouterLink}
              to={item.path}
              px={2}
              py={1}
              rounded="md"
              fontWeight={location.pathname === item.path ? 'bold' : 'normal'}
              color={location.pathname === item.path ? 'blue.500' : 'inherit'}
              _hover={{
                textDecoration: 'none',
                bg: useColorModeValue('gray.100', 'gray.700'),
              }}
            >
              {item.name}
            </Link>
          ))}
          <DarkModeToggle />
          <LanguageSwitcher />
        </HStack>

        {/* Mobile menu toggle */}
        <IconButton
          size="md"
          icon={isOpen ? <CloseIcon /> : <HamburgerIcon />}
          aria-label="Toggle navigation"
          display={{ md: 'none' }}
          onClick={onToggle}
        />
      </Flex>

      {/* Mobile Navigation */}
      {isOpen && (
        <Box pb={4} display={{ md: 'none' }}>
          <Stack as="nav" spacing={4}>
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.path}
                as={RouterLink}
                to={item.path}
                px={2}
                py={1}
                rounded="md"
                fontWeight={location.pathname === item.path ? 'bold' : 'normal'}
                color={location.pathname === item.path ? 'blue.500' : 'inherit'}
                _hover={{
                  textDecoration: 'none',
                  bg: useColorModeValue('gray.100', 'gray.700'),
                }}
                onClick={onToggle}
              >
                {item.name}
              </Link>
            ))}
          </Stack>
        </Box>
      )}
    </Box>
  );
};
