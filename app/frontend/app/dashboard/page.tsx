"use client";

import { Starfield } from "@/components/starfield";
import { AriaAvatar } from "@/components/aria-avatar";
import { api, BUNDLE_COLORS } from "@/lib/api";
import useSWR from "swr";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Activity, Cpu, Clock, Layers, Gauge, Boxes, Brain, Settings2 } from "lucide-react";
import { useState } from "react";

const ARIA_QUIPS = [
  "Hovering around, I see. Curious minds predict better.",
  "That data point is particularly interesting, isn't it?",
  "I crunched those numbers myself. You're welcome.",
  "The neural matrix is strong with this one.",
  "Keep hovering, I like the attention.",
  "These stats are real-time. I never sleep.",
];

export default function DashboardPage() {
  const [ariaQuip, setAriaQuip] = useState("Welcome to the observatory. Everything you see here is live data.");

  const { data: metrics } = useSWR("dash-metrics", () => api.getMetrics(), {
    refreshInterval: 5000,
    fallbackData: { total_predictions: 0, rolling_avg_latency_ms: 0, bundle_distribution: {}, error_rate: 0 },
  });

  const { data: features } = useSWR("dash-features", () => api.getModelFeatures(), {
    fallbackData: { full: {}, top_10: [] },
  });

  const { data: modelInfo } = useSWR("dash-model-info", () => api.getModelInfo(), {
    fallbackData: { model_version: "...", num_features: 0, num_classes: 0, feature_columns: [], hyperparameters: { num_leaves: 0, learning_rate: 0 }, encoding_strategy: "" },
  });

  const bundleData = Object.entries(metrics?.bundle_distribution ?? {}).map(([k, v]) => ({
    name: `B${k}`,
    value: v,
    fill: BUNDLE_COLORS[parseInt(k)] || "#666",
  }));

  const featureData = (features?.top_10 ?? []).map((f) => ({
    name: f.feature.length > 18 ? f.feature.slice(0, 18) + "..." : f.feature,
    fullName: f.feature,
    importance: f.importance,
  }));

  const maxImportance = Math.max(...featureData.map((f) => f.importance), 1);

  const handleCardHover = () => {
    setAriaQuip(ARIA_QUIPS[Math.floor(Math.random() * ARIA_QUIPS.length)]);
  };

  // Latency gauge percentage (cap at 200ms)
  const latency = metrics?.rolling_avg_latency_ms ?? 0;
  const latencyPct = Math.min((latency / 200) * 100, 100);
  const latencyColor = latency < 50 ? "#39ff14" : latency < 100 ? "#ffaa00" : "#ff4444";

  return (
    <div className="relative min-h-screen">
      <Starfield />

      {/* ARIA floating in background */}
      <div className="fixed bottom-8 right-8 z-20 flex items-end gap-3 hidden lg:flex">
        <div className="glass rounded-xl px-4 py-2 max-w-xs">
          <p className="text-foreground text-xs">{ariaQuip}</p>
        </div>
        <AriaAvatar size={80} state="idle" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-neon-cyan/10 flex items-center justify-center">
            <Activity className="w-5 h-5 text-neon-cyan" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground font-mono">
              Observatory <span className="text-neon-cyan">Dashboard</span>
            </h1>
            <p className="text-muted-foreground text-sm">Real-time model metrics and insights</p>
          </div>
        </div>

        {/* Top stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <MetricCard
            icon={<Boxes className="w-5 h-5 text-neon-cyan" />}
            label="Total Predictions"
            value={metrics?.total_predictions?.toLocaleString() ?? "0"}
            color="#00e5ff"
            onHover={handleCardHover}
          />
          <MetricCard
            icon={<Clock className="w-5 h-5 text-neon-amber" />}
            label="Avg Latency"
            value={`${(metrics?.rolling_avg_latency_ms ?? 0).toFixed(1)}ms`}
            color="#ffaa00"
            onHover={handleCardHover}
          />
          <MetricCard
            icon={<Layers className="w-5 h-5 text-neon-green" />}
            label="Features"
            value={String(modelInfo?.num_features ?? 0)}
            color="#39ff14"
            onHover={handleCardHover}
          />
          <MetricCard
            icon={<Gauge className="w-5 h-5 text-neon-pink" />}
            label="Error Rate"
            value={`${((metrics?.error_rate ?? 0) * 100).toFixed(2)}%`}
            color="#ff2d78"
            onHover={handleCardHover}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Bundle distribution chart */}
          <div className="glass rounded-xl p-6" onMouseEnter={handleCardHover}>
            <h3 className="text-foreground font-semibold mb-4 font-mono flex items-center gap-2">
              <Boxes className="w-4 h-4 text-neon-cyan" />
              Bundle Distribution
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={bundleData}>
                  <XAxis
                    dataKey="name"
                    stroke="#7a7aaa"
                    fontSize={12}
                    fontFamily="monospace"
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    stroke="#7a7aaa"
                    fontSize={12}
                    fontFamily="monospace"
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0a0a20",
                      border: "1px solid rgba(0,229,255,0.2)",
                      borderRadius: "8px",
                      color: "#e8eaf6",
                      fontSize: "12px",
                      fontFamily: "monospace",
                    }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {bundleData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Latency gauge */}
          <div className="glass rounded-xl p-6" onMouseEnter={handleCardHover}>
            <h3 className="text-foreground font-semibold mb-4 font-mono flex items-center gap-2">
              <Gauge className="w-4 h-4 text-neon-amber" />
              Latency Gauge
            </h3>
            <div className="flex flex-col items-center justify-center h-64 gap-6">
              <div className="relative w-48 h-48">
                <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                  {/* Background arc */}
                  <circle
                    cx="50" cy="50" r="42"
                    fill="none"
                    stroke="#1a1a3e"
                    strokeWidth="8"
                    strokeDasharray={`${Math.PI * 84}`}
                    strokeLinecap="round"
                  />
                  {/* Value arc */}
                  <circle
                    cx="50" cy="50" r="42"
                    fill="none"
                    stroke={latencyColor}
                    strokeWidth="8"
                    strokeDasharray={`${(latencyPct / 100) * Math.PI * 84} ${Math.PI * 84}`}
                    strokeLinecap="round"
                    style={{
                      filter: `drop-shadow(0 0 6px ${latencyColor}60)`,
                      transition: "stroke-dasharray 0.5s ease",
                    }}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold font-mono" style={{ color: latencyColor }}>
                    {latency.toFixed(0)}
                  </span>
                  <span className="text-muted-foreground text-xs">ms avg</span>
                </div>
              </div>
              <div className="flex items-center gap-4 text-xs">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-neon-green" /> {"< 50ms"}</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-neon-amber" /> {"50-100ms"}</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ backgroundColor: "#ff4444" }} /> {"> 100ms"}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Feature importance */}
          <div className="glass rounded-xl p-6" onMouseEnter={handleCardHover}>
            <h3 className="text-foreground font-semibold mb-4 font-mono flex items-center gap-2">
              <Brain className="w-4 h-4 text-neon-green" />
              Top 10 Feature Importances
            </h3>
            <div className="space-y-3">
              {featureData.map((f, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="text-xs text-muted-foreground w-40 truncate font-mono" title={f.fullName}>
                    {f.name}
                  </span>
                  <div className="flex-1 h-4 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-1000"
                      style={{
                        width: `${(f.importance / maxImportance) * 100}%`,
                        backgroundColor: BUNDLE_COLORS[i % BUNDLE_COLORS.length],
                        boxShadow: `0 0 8px ${BUNDLE_COLORS[i % BUNDLE_COLORS.length]}40`,
                      }}
                    />
                  </div>
                  <span className="text-xs font-mono text-foreground w-20 text-right">
                    {f.importance > 1000 ? `${(f.importance / 1000).toFixed(1)}k` : f.importance.toFixed(1)}
                  </span>
                </div>
              ))}
              {featureData.length === 0 && (
                <p className="text-muted-foreground text-sm text-center py-8">No feature data available</p>
              )}
            </div>
          </div>

          {/* Model info */}
          <div className="glass rounded-xl p-6" onMouseEnter={handleCardHover}>
            <h3 className="text-foreground font-semibold mb-4 font-mono flex items-center gap-2">
              <Settings2 className="w-4 h-4 text-neon-purple" />
              Model Information
            </h3>
            <div className="space-y-4">
              <InfoRow label="Version" value={modelInfo?.model_version ?? "..."} />
              <InfoRow label="Features" value={String(modelInfo?.num_features ?? 0)} />
              <InfoRow label="Classes" value={String(modelInfo?.num_classes ?? 0)} />
              <InfoRow
                label="Num Leaves"
                value={String(modelInfo?.hyperparameters?.num_leaves ?? 0)}
              />
              <InfoRow
                label="Learning Rate"
                value={String(modelInfo?.hyperparameters?.learning_rate ?? 0)}
              />
              <div className="pt-2 border-t border-border">
                <p className="text-muted-foreground text-xs font-mono uppercase tracking-wider mb-2">Encoding Strategy</p>
                <p className="text-foreground text-sm leading-relaxed">
                  {modelInfo?.encoding_strategy || "Not available"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  color,
  onHover,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: string;
  onHover: () => void;
}) {
  return (
    <div
      className="glass glass-hover rounded-xl p-5 flex flex-col items-center gap-2 transition-all hover:scale-105 cursor-default"
      onMouseEnter={onHover}
      style={{ boxShadow: `0 0 20px ${color}15` }}
    >
      {icon}
      <span className="font-mono text-xl font-bold" style={{ color }}>{value}</span>
      <span className="text-muted-foreground text-xs">{label}</span>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground text-sm">{label}</span>
      <span className="text-foreground font-mono text-sm font-bold">{value}</span>
    </div>
  );
}
