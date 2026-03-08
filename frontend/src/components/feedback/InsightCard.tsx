/**
 * InsightCard - Glassmorphism insight card with lightbulb icon
 */
import { Box, HStack, Text } from '@chakra-ui/react';

interface InsightCardProps {
  text: string;
  index?: number;
}

export const InsightCard = ({ text }: InsightCardProps) => {
  return (
    <Box
      bg="rgba(255, 255, 255, 0.04)"
      backdropFilter="blur(12px)"
      border="1px solid"
      borderColor="rgba(255, 215, 0, 0.2)"
      p={5}
      transition="all 0.3s ease"
      _hover={{
        borderColor: 'solar.gold',
        bg: 'rgba(255, 255, 255, 0.06)',
        transform: 'translateX(4px)',
      }}
    >
      <HStack align="flex-start" spacing={3}>
        <Text fontSize="lg" flexShrink={0} mt={-0.5}>
          {'\u{1F4A1}'}
        </Text>
        <Text fontSize="sm" color="spacex.textGray" lineHeight="tall">
          {text}
        </Text>
      </HStack>
    </Box>
  );
};
