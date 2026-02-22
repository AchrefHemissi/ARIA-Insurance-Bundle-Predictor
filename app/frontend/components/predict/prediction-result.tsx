"use client";

import { BUNDLE_COLORS, BUNDLE_NAMES, type ExplainResponse } from "@/lib/api";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

export function PredictionResult({ result }: { result: ExplainResponse }) {
  const { predicted_bundle, confidence, confidence_label, probabilities, inference_latency_ms, top_5_shap_features } = result;
  const bundleColor = BUNDLE_COLORS[predicted_bundle] || "#00e5ff";
  const bundleName = BUNDLE_NAMES[predicted_bundle] || `Bundle ${predicted_bundle}`;

  const confColor =
    confidence_label === "High" ? "#39ff14" :
    confidence_label === "Medium" ? "#ffaa00" : "#ff4444";

  const pieData = probabilities.map((p, i) => ({
    name: `Bundle ${i}`,
    value: parseFloat((p * 100).toFixed(1)),
    fill: BUNDLE_COLORS[i],
  }));

  const shapFeatures = top_5_shap_features?.slice(0, 5) ?? [];
  const maxShap = Math.max(...shapFeatures.map((f) => Math.abs(f.shap_value)), 0.01);

  return (
    <div className="glass rounded-xl p-6 space-y-6">
      {/* Bundle reveal */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl font-bold font-mono"
            style={{
              backgroundColor: `${bundleColor}20`,
              color: bundleColor,
              boxShadow: `0 0 30px ${bundleColor}40`,
            }}
          >
            {predicted_bundle}
          </div>
          <div>
            <p className="text-foreground font-bold text-lg">{bundleName}</p>
            <p className="text-muted-foreground text-sm">
              Bundle {predicted_bundle}
            </p>
          </div>
        </div>
        <div className="text-right">
          <div
            className="inline-block px-3 py-1 rounded-full text-sm font-bold font-mono"
            style={{ backgroundColor: `${confColor}20`, color: confColor }}
          >
            {(confidence * 100).toFixed(0)}%
          </div>
          <p className="text-muted-foreground text-xs mt-1">{confidence_label} Confidence</p>
        </div>
      </div>

      {/* Probability donut */}
      <div className="flex items-center gap-6">
        <div className="w-32 h-32">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={28}
                outerRadius={55}
                paddingAngle={2}
                dataKey="value"
                stroke="none"
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} opacity={i === predicted_bundle ? 1 : 0.3} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0a0a20",
                  border: "1px solid rgba(0,229,255,0.2)",
                  borderRadius: "8px",
                  color: "#e8eaf6",
                  fontSize: "12px",
                }}
                formatter={(value: number) => [`${value}%`]}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="flex-1 grid grid-cols-2 gap-x-4 gap-y-1">
          {probabilities.map((p, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: BUNDLE_COLORS[i] }} />
              <span className="text-muted-foreground">B{i}</span>
              <span className="font-mono text-foreground">{(p * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* SHAP features */}
      {shapFeatures.length > 0 && (
        <div className="space-y-3">
          <p className="text-muted-foreground text-xs font-mono uppercase tracking-wider">Top Influencing Features</p>
          {shapFeatures.map((f, i) => {
            const width = (Math.abs(f.shap_value) / maxShap) * 100;
            const isPositive = f.direction === "positive";
            return (
              <div key={i} className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground w-40 truncate font-mono">
                  {f.feature.replace(/_/g, " ")}
                </span>
                <div className="flex-1 h-3 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-1000"
                    style={{
                      width: `${width}%`,
                      backgroundColor: isPositive ? "#39ff14" : "#ff4444",
                      boxShadow: `0 0 10px ${isPositive ? "#39ff1440" : "#ff444440"}`,
                      animationDelay: `${i * 150}ms`,
                    }}
                  />
                </div>
                <span className={`text-xs font-mono w-14 text-right ${isPositive ? "text-neon-green" : "text-destructive"}`}>
                  {isPositive ? "+" : ""}{f.shap_value.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Latency */}
      <div className="flex justify-end">
        <span className="text-muted-foreground text-xs font-mono">
          Inference: {inference_latency_ms.toFixed(1)}ms
        </span>
      </div>
    </div>
  );
}
