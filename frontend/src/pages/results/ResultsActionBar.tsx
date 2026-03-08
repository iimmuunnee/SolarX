/**
 * ResultsActionBar - Navigation + share actions
 */
import { Box, Container, HStack, Button, useToast } from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { ExternalLinkIcon, CopyIcon } from '@chakra-ui/icons';
import { useInViewAnimation } from '../../hooks/useInViewAnimation';

const MotionBox = motion(Box);

export const ResultsActionBar = () => {
  const { t } = useTranslation('pages');
  const navigate = useNavigate();
  const toast = useToast();
  const { variants, viewportOptions } = useInViewAnimation();

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href).then(() => {
      toast({
        title: t('results.actions.copied', { defaultValue: 'URL copied to clipboard!' }),
        status: 'success',
        duration: 2000,
        isClosable: true,
      });
    });
  };

  return (
    <Box py={12} borderTop="1px solid" borderColor="spacex.borderGray">
      <Container maxW="7xl" px={{ base: 4, md: 6, lg: 8 }}>
        <MotionBox
          variants={variants}
          initial="hidden"
          whileInView="visible"
          viewport={viewportOptions}
        >
          <HStack
            spacing={4}
            justify="center"
            flexWrap="wrap"
          >
            <Button
              variant="spacexSolar"
              onClick={() => navigate('/demo')}
              rightIcon={<ExternalLinkIcon />}
            >
              {t('results.actions.demo', { defaultValue: 'Try Demo Simulation' })}
            </Button>
            <Button
              variant="spacex"
              onClick={() => navigate('/architecture')}
              rightIcon={<ExternalLinkIcon />}
            >
              {t('results.actions.architecture', { defaultValue: 'View Architecture' })}
            </Button>
            <Button
              variant="spacex"
              onClick={handleShare}
              rightIcon={<CopyIcon />}
            >
              {t('results.actions.share', { defaultValue: 'Share Results' })}
            </Button>
          </HStack>
        </MotionBox>
      </Container>
    </Box>
  );
};
