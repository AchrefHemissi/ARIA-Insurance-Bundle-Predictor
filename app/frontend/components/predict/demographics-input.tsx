"use client";

import { useState } from "react";
import { Slider } from "@/components/ui/slider";
import { ArrowRight } from "lucide-react";

interface DemographicsData {
  Adult_Dependents: number;
  Child_Dependents: number;
  Infant_Dependents: number;
  Annual_Income: number;
  Employment_Status: number;
}

export function DemographicsInput({ onSubmit }: { onSubmit: (data: DemographicsData) => void }) {
  const [adults, setAdults] = useState(0);
  const [children, setChildren] = useState(0);
  const [infants, setInfants] = useState(0);
  const [income, setIncome] = useState(50000);
  const [employed, setEmployed] = useState(1);

  return (
    <div className="glass rounded-xl p-5 space-y-5">
      <div className="space-y-3">
        <SliderField label="Adult Dependents" value={adults} min={0} max={10} onChange={setAdults} />
        <SliderField label="Child Dependents" value={children} min={0} max={10} onChange={setChildren} />
        <SliderField label="Infant Dependents" value={infants} min={0} max={5} onChange={setInfants} />
        <SliderField label="Annual Income" value={income} min={10000} max={200000} step={5000} onChange={setIncome} format={(v) => `$${v.toLocaleString()}`} />
      </div>

      <div className="space-y-2">
        <label className="text-muted-foreground text-xs font-mono uppercase tracking-wider">Employment Status</label>
        <div className="flex gap-2">
          {[{ label: "Employed", value: 1 }, { label: "Unemployed", value: 0 }].map((opt) => (
            <button
              key={opt.value}
              onClick={() => setEmployed(opt.value)}
              suppressHydrationWarning
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${employed === opt.value
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
            Adult_Dependents: adults,
            Child_Dependents: children,
            Infant_Dependents: infants,
            Annual_Income: income,
            Employment_Status: employed,
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
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-muted-foreground text-xs font-mono uppercase tracking-wider">{label}</label>
        <span className="text-foreground text-sm font-mono font-bold">{format ? format(value) : value}</span>
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
