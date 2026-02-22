"use client";

import { useState } from "react";
import { Slider } from "@/components/ui/slider";
import { ArrowRight } from "lucide-react";

interface RiskData {
  Previous_Claims_Filed: number;
  Years_Without_Claims: number;
  Existing_Policyholder: number;
  Grace_Period_Extensions: number;
  Days_Since_Quote: number;
}

export function RiskInput({ onSubmit }: { onSubmit: (data: RiskData) => void }) {
  const [claims, setClaims] = useState(0);
  const [yearsClean, setYearsClean] = useState(3);
  const [existing, setExisting] = useState(0);
  const [grace, setGrace] = useState(0);
  const [daysSinceQuote, setDaysSinceQuote] = useState(30);

  return (
    <div className="glass rounded-xl p-5 space-y-5">
      <div className="space-y-3">
        <SliderField label="Previous Claims Filed" value={claims} min={0} max={15} onChange={setClaims} />
        <SliderField label="Years Without Claims" value={yearsClean} min={0} max={20} onChange={setYearsClean} />
        <SliderField label="Grace Period Extensions" value={grace} min={0} max={10} onChange={setGrace} />
        <SliderField label="Days Since Quote" value={daysSinceQuote} min={0} max={365} step={5} onChange={setDaysSinceQuote} />
      </div>

      <div className="space-y-2">
        <label className="text-muted-foreground text-xs font-mono uppercase tracking-wider">Existing Policyholder</label>
        <div className="flex gap-2">
          {[{ label: "Yes", value: 1 }, { label: "No", value: 0 }].map((opt) => (
            <button
              key={opt.value}
              onClick={() => setExisting(opt.value)}
              suppressHydrationWarning
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${existing === opt.value
                  ? "bg-neon-cyan/20 text-neon-cyan border border-neon-cyan/40"
                  : "glass text-muted-foreground hover:text-foreground"
                }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={() =>
          onSubmit({
            Previous_Claims_Filed: claims,
            Years_Without_Claims: yearsClean,
            Existing_Policyholder: existing,
            Grace_Period_Extensions: grace,
            Days_Since_Quote: daysSinceQuote,
          })
        }
        suppressHydrationWarning
        className="w-full glass glass-hover glow-cyan rounded-xl px-4 py-3 flex items-center justify-center gap-2 text-neon-cyan font-semibold transition-all hover:scale-[1.02]"
      >
        {"Looks good"}
        <ArrowRight className="w-4 h-4" />
      </button>
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-muted-foreground text-xs font-mono uppercase tracking-wider">{label}</label>
        <span className="text-foreground text-sm font-mono font-bold">{value}</span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([v]) => onChange(v)}
        className="[&_[role=slider]]:bg-neon-cyan [&_[role=slider]]:border-neon-cyan/50 [&_.range]:bg-neon-cyan/40"
      />
    </div>
  );
}
