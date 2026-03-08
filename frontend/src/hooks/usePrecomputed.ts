/**
 * Hook for fetching pre-computed results
 * Falls back to static data when the API is unavailable
 */
import { useState, useEffect } from 'react';
import { fetchPrecomputedBenchmark } from '../services/api';
import { fallbackBenchmarkData } from '../data/fallbackResults';
import type { BenchmarkResponse } from '../types/simulation';

export const usePrecomputed = () => {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isFallback, setIsFallback] = useState(false);

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetchPrecomputedBenchmark();
        setData(response);
        setError(null);
      } catch (err) {
        // API failed — use fallback static data
        setData(fallbackBenchmarkData);
        setIsFallback(true);
        setError(null);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  return { data, loading, error, isFallback };
};
