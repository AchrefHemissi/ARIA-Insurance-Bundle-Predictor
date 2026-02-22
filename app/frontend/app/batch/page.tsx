"use client";

import { useState, useCallback } from "react";
import { Starfield } from "@/components/starfield";
import { AriaAvatar } from "@/components/aria-avatar";
import { api, BUNDLE_COLORS, type BatchResponse } from "@/lib/api";
import { Upload, Download, FileSpreadsheet, AlertCircle } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

type AriaState = "idle" | "thinking" | "happy" | "nervous";

export default function BatchPage() {
  const [ariaState, setAriaState] = useState<AriaState>("idle");
  const [ariaMessage, setAriaMessage] = useState("Drop a CSV on me. Let's see what you've got.");
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(0);

  const ROWS_PER_PAGE = 20;

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.endsWith(".csv")) {
      setError("Please upload a CSV file.");
      setAriaMessage("That's not a CSV... I need a CSV file to work my magic.");
      setAriaState("nervous");
      return;
    }

    setError(null);
    setResult(null);
    setIsProcessing(true);
    setAriaState("thinking");
    setAriaMessage(`Processing ${file.name}...`);

    try {
      const res = await api.batchPredict(file);
      setResult(res);
      setAriaState("happy");

      const distribution = res.bundle_distribution;
      const mostCommon = Object.entries(distribution).sort((a, b) => b[1] - a[1])[0];
      setAriaMessage(
        `${res.rows_processed} clients done. Bundle ${mostCommon[0]} was the crowd favorite with ${mostCommon[1]} picks. Average confidence: ${(res.average_confidence * 100).toFixed(0)}%.`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch prediction failed.");
      setAriaMessage("Something went wrong with that batch. Check the file format and try again.");
      setAriaState("nervous");
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const downloadResults = useCallback(() => {
    if (!result) return;
    const header = "User_ID,Predicted_Bundle,Confidence\n";
    const rows = result.results
      .map((r) => `${r.User_ID},${r.predicted_bundle},${r.confidence.toFixed(4)}`)
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  const pieData = result
    ? Object.entries(result.bundle_distribution).map(([k, v]) => ({
        name: `Bundle ${k}`,
        value: v,
        fill: BUNDLE_COLORS[parseInt(k)] || "#666",
      }))
    : [];

  const totalPages = result ? Math.ceil(result.results.length / ROWS_PER_PAGE) : 0;
  const pagedResults = result?.results.slice(currentPage * ROWS_PER_PAGE, (currentPage + 1) * ROWS_PER_PAGE) ?? [];

  return (
    <div className="relative min-h-screen">
      <Starfield />

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-8">
        <div className="flex items-start gap-6 mb-8">
          <div className="flex-1">
            <h1 className="text-3xl font-bold text-foreground font-mono">
              Mission <span className="text-neon-cyan">Control</span>
            </h1>
            <p className="text-muted-foreground mt-1">Batch prediction processing</p>
          </div>

          {/* Mini ARIA */}
          <div className="flex items-center gap-3">
            <AriaAvatar size={60} state={ariaState} />
            <div className="glass rounded-xl px-4 py-2 max-w-xs">
              <p className="text-foreground text-sm">{ariaMessage}</p>
            </div>
          </div>
        </div>

        {/* Drop zone */}
        {!result && (
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`glass rounded-2xl p-12 flex flex-col items-center justify-center gap-4 transition-all cursor-pointer border-2 border-dashed ${
              isDragging
                ? "border-neon-cyan bg-neon-cyan/5 glow-cyan"
                : "border-border hover:border-neon-cyan/40"
            } ${isProcessing ? "pointer-events-none opacity-60" : ""}`}
          >
            {isProcessing ? (
              <>
                <div className="w-12 h-12 border-2 border-neon-cyan border-t-transparent rounded-full animate-spin" />
                <p className="text-neon-cyan font-mono">Processing batch...</p>
              </>
            ) : (
              <>
                <div className="w-16 h-16 rounded-2xl bg-neon-cyan/10 flex items-center justify-center">
                  <Upload className="w-8 h-8 text-neon-cyan" />
                </div>
                <p className="text-foreground text-lg font-semibold">Drop your CSV file here</p>
                <p className="text-muted-foreground text-sm">or click to browse</p>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleInputChange}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                  aria-label="Upload CSV file"
                />
              </>
            )}
          </div>
        )}

        {error && (
          <div className="glass rounded-xl p-4 flex items-center gap-3 mt-4 border border-destructive/30">
            <AlertCircle className="w-5 h-5 text-destructive shrink-0" />
            <p className="text-destructive text-sm">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6 mt-6">
            {/* Stats row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Rows Processed" value={result.rows_processed.toLocaleString()} color="#00e5ff" />
              <StatCard label="Avg Confidence" value={`${(result.average_confidence * 100).toFixed(0)}%`} color="#39ff14" />
              <StatCard label="Total Latency" value={`${result.total_latency_ms.toFixed(0)}ms`} color="#ffaa00" />
              <StatCard label="Bundles Found" value={String(Object.values(result.bundle_distribution).filter((v) => v > 0).length)} color="#ff2d78" />
            </div>

            {/* Distribution chart + download */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="glass rounded-xl p-6">
                <h3 className="text-foreground font-semibold mb-4 font-mono">Bundle Distribution</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData.filter((d) => d.value > 0)}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={80}
                        paddingAngle={2}
                        dataKey="value"
                        stroke="none"
                      >
                        {pieData.filter((d) => d.value > 0).map((entry, i) => (
                          <Cell key={i} fill={entry.fill} />
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
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex flex-wrap gap-3 mt-4 justify-center">
                  {pieData.filter((d) => d.value > 0).map((d, i) => (
                    <div key={i} className="flex items-center gap-1.5 text-xs">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: d.fill }} />
                      <span className="text-muted-foreground">{d.name}: {d.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass rounded-xl p-6 flex flex-col items-center justify-center gap-4">
                <FileSpreadsheet className="w-12 h-12 text-neon-green" />
                <p className="text-foreground font-semibold">Results Ready</p>
                <p className="text-muted-foreground text-sm text-center">
                  {result.rows_processed} predictions generated
                </p>
                <button
                  onClick={downloadResults}
                  className="glass glass-hover glow-green rounded-xl px-6 py-3 flex items-center gap-2 text-neon-green font-semibold transition-all hover:scale-105"
                >
                  <Download className="w-4 h-4" />
                  Download CSV
                </button>
                <button
                  onClick={() => { setResult(null); setAriaMessage("Ready for another batch. Drop it on me."); setAriaState("idle"); }}
                  className="text-muted-foreground text-sm hover:text-foreground transition-colors"
                >
                  Upload another file
                </button>
              </div>
            </div>

            {/* Results table */}
            <div className="glass rounded-xl overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left px-4 py-3 font-mono text-muted-foreground text-xs uppercase tracking-wider">User ID</th>
                      <th className="text-left px-4 py-3 font-mono text-muted-foreground text-xs uppercase tracking-wider">Bundle</th>
                      <th className="text-left px-4 py-3 font-mono text-muted-foreground text-xs uppercase tracking-wider">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pagedResults.map((row, i) => {
                      const color = BUNDLE_COLORS[row.predicted_bundle] || "#666";
                      return (
                        <tr key={i} className="border-b border-border/50 hover:bg-secondary/30 transition-colors">
                          <td className="px-4 py-3 font-mono text-foreground">{row.User_ID}</td>
                          <td className="px-4 py-3">
                            <span
                              className="inline-block px-2.5 py-1 rounded-md text-xs font-bold font-mono"
                              style={{ backgroundColor: `${color}20`, color }}
                            >
                              Bundle {row.predicted_bundle}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <div className="w-20 h-2 bg-secondary rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full"
                                  style={{
                                    width: `${row.confidence * 100}%`,
                                    backgroundColor: row.confidence >= 0.6 ? "#39ff14" : row.confidence >= 0.35 ? "#ffaa00" : "#ff4444",
                                  }}
                                />
                              </div>
                              <span className="font-mono text-foreground text-xs">{(row.confidence * 100).toFixed(0)}%</span>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between px-4 py-3 border-t border-border">
                  <span className="text-muted-foreground text-xs">
                    Page {currentPage + 1} of {totalPages}
                  </span>
                  <div className="flex gap-2">
                    <button
                      disabled={currentPage === 0}
                      onClick={() => setCurrentPage((p) => p - 1)}
                      className="px-3 py-1 rounded-md text-xs glass text-muted-foreground hover:text-foreground disabled:opacity-30 transition-all"
                    >
                      Prev
                    </button>
                    <button
                      disabled={currentPage >= totalPages - 1}
                      onClick={() => setCurrentPage((p) => p + 1)}
                      className="px-3 py-1 rounded-md text-xs glass text-muted-foreground hover:text-foreground disabled:opacity-30 transition-all"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="glass rounded-xl p-4 flex flex-col items-center gap-1">
      <span className="font-mono text-xl font-bold" style={{ color }}>{value}</span>
      <span className="text-muted-foreground text-xs">{label}</span>
    </div>
  );
}
