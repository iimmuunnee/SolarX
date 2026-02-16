import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Import translation files
import koCommon from './locales/ko/common.json';
import koPages from './locales/ko/pages.json';
import koCharts from './locales/ko/charts.json';
import enCommon from './locales/en/common.json';
import enPages from './locales/en/pages.json';
import enCharts from './locales/en/charts.json';

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    fallbackLng: 'ko', // Korean as default
    defaultNS: 'common',
    ns: ['common', 'pages', 'charts'],
    interpolation: {
      escapeValue: false, // React already escapes values
    },
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage'],
    },
    resources: {
      ko: {
        common: koCommon,
        pages: koPages,
        charts: koCharts,
      },
      en: {
        common: enCommon,
        pages: enPages,
        charts: enCharts,
      },
    },
  });

export default i18n;
