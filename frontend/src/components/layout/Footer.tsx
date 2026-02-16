/**
 * Footer component
 */
import { Box, Container, Flex, Link, Text, useColorModeValue } from '@chakra-ui/react';

export const Footer = () => {
  const bgColor = useColorModeValue('gray.50', 'gray.900');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box bg={bgColor} borderTop="1px" borderColor={borderColor} mt={8}>
      <Container maxW="7xl" py={8}>
        <Flex direction={{ base: 'column', md: 'row' }} justify="space-between" align="center" gap={4}>
          <Text fontSize="sm" color="gray.600">
            © 2026 SolarX v5.0. Battery optimization for humanoid robot charging stations.
          </Text>
          <Flex gap={4}>
            <Link
              href="https://github.com/yourusername/solarx"
              isExternal
              fontSize="sm"
              color="blue.500"
              _hover={{ textDecoration: 'underline' }}
            >
              GitHub
            </Link>
            <Link
              href="https://claude.com/claude-code"
              isExternal
              fontSize="sm"
              color="blue.500"
              _hover={{ textDecoration: 'underline' }}
            >
              Built with Claude Code
            </Link>
          </Flex>
        </Flex>
      </Container>
    </Box>
  );
};
