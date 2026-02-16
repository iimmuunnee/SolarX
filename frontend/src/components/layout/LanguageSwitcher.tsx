/**
 * Language switcher component - Toggle between Korean and English
 */
import { ButtonGroup, IconButton, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

export const LanguageSwitcher = () => {
  const { i18n } = useTranslation();
  const currentLang = i18n.language;

  return (
    <ButtonGroup size="sm" isAttached variant="ghost">
      <IconButton
        aria-label="한국어"
        icon={<Text fontSize="lg">🇰🇷</Text>}
        onClick={() => i18n.changeLanguage('ko')}
        colorScheme={currentLang === 'ko' ? 'blue' : 'gray'}
        fontWeight={currentLang === 'ko' ? 'bold' : 'normal'}
      />
      <IconButton
        aria-label="English"
        icon={<Text fontSize="lg">🇺🇸</Text>}
        onClick={() => i18n.changeLanguage('en')}
        colorScheme={currentLang === 'en' ? 'blue' : 'gray'}
        fontWeight={currentLang === 'en' ? 'bold' : 'normal'}
      />
    </ButtonGroup>
  );
};
