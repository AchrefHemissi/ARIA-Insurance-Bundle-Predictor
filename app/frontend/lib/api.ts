const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface HealthResponse {
  model_loaded: boolean;
  num_features: number;
  num_classes: number;
  uptime_seconds: number;
}

export interface MetricsResponse {
  total_predictions: number;
  rolling_avg_latency_ms: number;
  bundle_distribution: Record<string, number>;
  error_rate: number;
}

export interface ModelInfoResponse {
  model_version: string;
  num_features: number;
  num_classes: number;
  feature_columns: string[];
  hyperparameters: { num_leaves: number; learning_rate: number };
  encoding_strategy: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelFeaturesResponse {
  full: Record<string, number>;
  top_10: FeatureImportance[];
}

export interface PredictRequest {
  Adult_Dependents: number;
  Child_Dependents: number;
  Infant_Dependents: number;
  Annual_Income: number;
  Employment_Status: number;
  Previous_Claims_Filed: number;
  Years_Without_Claims: number;
  Existing_Policyholder: number;
  Grace_Period_Extensions: number;
  Days_Since_Quote: number;
  Acquisition_Channel: string;
  Broker_Agency_Type: string;
  Deductible_Tier: string;
  Policy_Start_Month: number;
  Custom_Riders: number;
  Vehicles: number;
}

export interface PredictResponse {
  predicted_bundle: number;
  probabilities: number[];
  confidence: number;
  confidence_label: "High" | "Medium" | "Low";
  inference_latency_ms: number;
}

export interface ShapFeature {
  feature: string;
  shap_value: number;
  direction: "positive" | "negative";
}

export interface ExplainResponse extends PredictResponse {
  top_5_shap_features: ShapFeature[];
}

export interface BatchResult {
  User_ID: number;
  predicted_bundle: number;
  confidence: number;
}

export interface BatchResponse {
  rows_processed: number;
  results: BatchResult[];
  bundle_distribution: Record<string, number>;
  average_confidence: number;
  total_latency_ms: number;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options?.headers || {}),
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `API Error: ${res.status}`);
  }
  return res.json();
}

export const api = {
  getHealth: () => apiFetch<HealthResponse>("/health"),
  getMetrics: () => apiFetch<MetricsResponse>("/metrics"),
  getModelInfo: () => apiFetch<ModelInfoResponse>("/model/info"),
  getModelFeatures: () => apiFetch<ModelFeaturesResponse>("/model/features"),
  predict: (data: PredictRequest) =>
    apiFetch<PredictResponse>("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }),
  explain: (data: PredictRequest) =>
    apiFetch<ExplainResponse>("/predict/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }),
  batchPredict: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return apiFetch<BatchResponse>("/predict/batch", {
      method: "POST",
      body: formData,
    });
  },
};

// Bundle color map
export const BUNDLE_COLORS = [
  "#00e5ff", "#ff2d78", "#39ff14", "#ffaa00", "#bf5fff",
  "#00ffaa", "#ff6b35", "#4dc9f6", "#f67019", "#f53794",
];

export const BUNDLE_NAMES = [
  "Essential", "Standard", "Premium", "Family", "Elite",
  "Starter", "Business", "Corporate", "Enterprise", "Ultimate",
];
