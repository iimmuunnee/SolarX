/**
 * Hook for fetching pre-computed results
 */
import { useState, useEffect } from 'react';
import { fetchPrecomputedBenchmark } from '../services/api';
import type { BenchmarkResponse } from '../types/simulation';

export const usePrecomputed = () => {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetchPrecomputedBenchmark();
        setData(response);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load results');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  return { data, loading, error };
};
