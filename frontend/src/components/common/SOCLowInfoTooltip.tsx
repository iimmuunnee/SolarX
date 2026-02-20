import { InfoOutlineIcon } from '@chakra-ui/icons';
import { Box, HStack, Text, Tooltip } from '@chakra-ui/react';

interface SOCLowInfoTooltipProps {
  label: string;
  title?: string;
}

export const SOCLowInfoTooltip = ({
  label,
  title = '\uC65C SOC\uAC00 \uB0AE\uB098\uC694?',
}: SOCLowInfoTooltipProps) => {
  return (
    <Box display="inline-flex">
      <Tooltip
        label={label}
        placement="top-end"
        hasArrow
        bg="gray.800"
        color="white"
        fontSize="sm"
        lineHeight="1.5"
        maxW="340px"
        p={4}
        borderRadius="lg"
        borderWidth="1px"
        borderColor="gray.600"
        boxShadow="xl"
        textAlign="left"
        openDelay={120}
        closeDelay={120}
      >
        <Box
          as="button"
          type="button"
          aria-label="SOC explanation"
          px={2.5}
          py={1.5}
          borderWidth="1px"
          borderColor="gray.600"
          borderRadius="md"
          bg="rgba(17, 24, 39, 0.9)"
          color="gray.200"
          cursor="help"
          _hover={{ bg: 'gray.700', color: 'white', borderColor: 'gray.400' }}
          _focusVisible={{
            outline: 'none',
            boxShadow: '0 0 0 2px rgba(96, 165, 250, 0.8)',
            borderColor: 'blue.300',
          }}
        >
          <HStack spacing={1.5}>
            <InfoOutlineIcon boxSize={3.5} />
            <Text fontSize="xs" fontWeight="semibold" lineHeight="1.2" whiteSpace="nowrap">
              {title}
            </Text>
          </HStack>
        </Box>
      </Tooltip>
    </Box>
  );
};

export default SOCLowInfoTooltip;
