/**
 * Hook for running simulations
 */
import { useState, useCallback } from 'react';
import { runBenchmark, runCustom } from '../services/api';
import type { BenchmarkRequest, BenchmarkResponse, CustomRequest, CustomResponse } from '../types/simulation';

interface SimulationState {
  loading: boolean;
  error: string | null;
  result: BenchmarkResponse | CustomResponse | null;
}

export const useSimulation = () => {
  const [state, setState] = useState<SimulationState>({
    loading: false,
    error: null,
    result: null,
  });

  const simulateBenchmark = useCallback(async (params: BenchmarkRequest) => {
    setState({ loading: true, error: null, result: null });
    try {
      const result = await runBenchmark(params);
      setState({ loading: false, error: null, result });
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Simulation failed';
      setState({ loading: false, error: errorMessage, result: null });
      throw error;
    }
  }, []);

  const simulateCustom = useCallback(async (params: CustomRequest) => {
    setState({ loading: true, error: null, result: null });
    try {
      const result = await runCustom(params);
      setState({ loading: false, error: null, result });
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Simulation failed';
      setState({ loading: false, error: errorMessage, result: null });
      throw error;
    }
  }, []);

  const reset = useCallback(() => {
    setState({ loading: false, error: null, result: null });
  }, []);

  return {
    ...state,
    simulateBenchmark,
    simulateCustom,
    reset,
  };
};
