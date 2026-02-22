"use client";

import { Starfield } from "@/components/starfield";
import { AriaAvatar } from "@/components/aria-avatar";
import { useTypewriter } from "@/hooks/use-typewriter";
import { api } from "@/lib/api";
import Link from "next/link";
import { ArrowRight, Upload, Zap, Clock, Layers } from "lucide-react";
import useSWR from "swr";

const GREETING =
  "Hey there! I'm ARIA, your AI insurance oracle. Drop a client profile on me and I'll tell you exactly which bundle they'll pick — I'm basically never wrong. (Okay, almost never.)";

export default function LandingPage() {
  const { displayed, isDone } = useTypewriter(GREETING, 25, 800);

  const { data: metrics } = useSWR("metrics", () => api.getMetrics(), {
    refreshInterval: 10000,
    fallbackData: { total_predictions: 0, rolling_avg_latency_ms: 0, bundle_distribution: {}, error_rate: 0 },
  });

  const { data: modelInfo } = useSWR("modelInfo", () => api.getModelInfo(), {
    fallbackData: { model_version: "...", num_features: 0, num_classes: 10, feature_columns: [], hyperparameters: { num_leaves: 0, learning_rate: 0 }, encoding_strategy: "" },
  });

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
      <Starfield />

      <div className="relative z-10 flex flex-col items-center gap-8 px-6 max-w-3xl text-center">
        {/* ARIA Avatar */}
        <div className="animate-float">
          <AriaAvatar size={220} state="idle" />
        </div>

        {/* Greeting text */}
        <div className="glass rounded-2xl p-6 max-w-2xl">
          <p className="text-foreground text-lg leading-relaxed font-sans">
            {displayed}
            {!isDone && (
              <span className="inline-block w-0.5 h-5 bg-neon-cyan ml-1 animate-pulse" />
            )}
          </p>
        </div>

        {/* CTA Buttons */}
        <div
          className="flex flex-col sm:flex-row items-center gap-4 transition-all"
          style={{ opacity: isDone ? 1 : 0, transform: isDone ? "translateY(0)" : "translateY(20px)", transition: "all 0.6s ease-out" }}
        >
          <Link
            href="/predict"
            className="glass glass-hover glow-cyan rounded-xl px-8 py-4 flex items-center gap-3 text-neon-cyan font-semibold text-lg transition-all hover:scale-105"
          >
            {"Let's Go"}
            <ArrowRight className="w-5 h-5" />
          </Link>
          <Link
            href="/batch"
            className="glass glass-hover glow-pink rounded-xl px-8 py-4 flex items-center gap-3 text-neon-pink font-semibold text-lg transition-all hover:scale-105"
          >
            Upload Batch
            <Upload className="w-5 h-5" />
          </Link>
        </div>

        {/* Stats */}
        <div
          className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full mt-8 transition-all"
          style={{ opacity: isDone ? 1 : 0, transform: isDone ? "translateY(0)" : "translateY(30px)", transition: "all 0.8s ease-out 0.2s" }}
        >
          <StatCard
            icon={<Zap className="w-5 h-5 text-neon-cyan" />}
            label="Predictions Served"
            value={metrics?.total_predictions?.toLocaleString() ?? "0"}
            color="cyan"
          />
          <StatCard
            icon={<Clock className="w-5 h-5 text-neon-pink" />}
            label="Avg Latency"
            value={`${(metrics?.rolling_avg_latency_ms ?? 0).toFixed(1)}ms`}
            color="pink"
          />
          <StatCard
            icon={<Layers className="w-5 h-5 text-neon-green" />}
            label="Features"
            value={String(modelInfo?.num_features ?? 0)}
            color="green"
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: "cyan" | "pink" | "green";
}) {
  const glowClass = color === "cyan" ? "glow-cyan" : color === "pink" ? "glow-pink" : "glow-green";
  return (
    <div className={`glass glass-hover rounded-xl p-5 flex flex-col items-center gap-2 ${glowClass} transition-all hover:scale-105`}>
      {icon}
      <span className="font-mono text-2xl font-bold text-foreground">{value}</span>
      <span className="text-muted-foreground text-sm">{label}</span>
    </div>
  );
}
