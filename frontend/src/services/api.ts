/**
 * API client for SolarX backend
 */
import axios from 'axios';
import type {
  BenchmarkRequest,
  BenchmarkResponse,
  CustomRequest,
  CustomResponse,
  ScalabilityResponse,
  HealthResponse,
} from '../types/simulation';

// Create axios instance with base URL
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

/**
 * Run benchmark simulation comparing all vendors
 */
export const runBenchmark = async (params: BenchmarkRequest): Promise<BenchmarkResponse> => {
  const response = await apiClient.post<BenchmarkResponse>('/simulate/benchmark', params);
  return response.data;
};

/**
 * Run custom simulation with specific vendor
 */
export const runCustom = async (params: CustomRequest): Promise<CustomResponse> => {
  const response = await apiClient.post<CustomResponse>('/simulate/custom', params);
  return response.data;
};

/**
 * Fetch pre-computed benchmark results (instant response)
 */
export const fetchPrecomputedBenchmark = async (): Promise<BenchmarkResponse> => {
  const response = await apiClient.get<BenchmarkResponse>('/results/benchmark');
  return response.data;
};

/**
 * Fetch pre-computed scalability results
 */
export const fetchPrecomputedScalability = async (): Promise<ScalabilityResponse> => {
  const response = await apiClient.get<ScalabilityResponse>('/results/scalability');
  return response.data;
};

/**
 * Health check
 */
export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
};

export default apiClient;
