/**
 * Main App component with routing
 */
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ChakraProvider, Box, Flex } from '@chakra-ui/react';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { Landing } from './pages/Landing';
import { Story } from './pages/Story';
import { Demo } from './pages/Demo';
import { Architecture } from './pages/Architecture';
import { Results } from './pages/Results';

function App() {
  return (
    <ChakraProvider>
      <Router>
        <Flex direction="column" minH="100vh">
          <Navbar />
          <Box flex="1">
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/story" element={<Story />} />
              <Route path="/demo" element={<Demo />} />
              <Route path="/architecture" element={<Architecture />} />
              <Route path="/results" element={<Results />} />
            </Routes>
          </Box>
          <Footer />
        </Flex>
      </Router>
    </ChakraProvider>
  );
}

export default App;
