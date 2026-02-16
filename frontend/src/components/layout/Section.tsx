/**
 * Reusable section container component
 */
import { Container, Box } from '@chakra-ui/react';
import type { ReactNode } from 'react';

interface SectionProps {
  children: ReactNode;
  bg?: string;
  py?: number | string;
  maxW?: string;
}

export const Section = ({ children, bg, py = 16, maxW = '7xl' }: SectionProps) => {
  return (
    <Box bg={bg} py={py}>
      <Container maxW={maxW} px={{ base: 4, md: 6, lg: 8 }}>
        {children}
      </Container>
    </Box>
  );
};
