/**
 * Hook for running simulations
 */
import { useState, useCallback } from 'react';
import { runBenchmark, runCustom } from '../services/api';
import { fallbackBenchmarkData } from '../data/fallbackResults';
import type { BenchmarkRequest, BenchmarkResponse, CustomRequest, CustomResponse } from '../types/simulation';

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    if (error.message === 'Network Error') {
      return '서버가 시작 중입니다. 30초~1분 후 다시 시도해주세요. (The server is waking up. Please retry in 30s–1min.)';
    }
    if (error.message.includes('timeout')) {
      return '서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요. (Request timed out. Please retry shortly.)';
    }
    return error.message;
  }
  return 'Simulation failed';
}

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
      const errorMessage = getErrorMessage(error);
      // API 실패 시 fallback 데이터를 result로 설정
      setState({ loading: false, error: errorMessage, result: fallbackBenchmarkData });
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
      const errorMessage = getErrorMessage(error);
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
