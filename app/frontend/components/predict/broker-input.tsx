"use client";

import { useState } from "react";
import { Slider } from "@/components/ui/slider";
import { ArrowRight, Building2, Globe, Users, Handshake } from "lucide-react";

interface BrokerData {
  Acquisition_Channel: string;
  Broker_Agency_Type: string;
  Deductible_Tier: string;
  Policy_Start_Month: number;
  Custom_Riders: number;
  Vehicles: number;
}

const channels = [
  { value: "Online", label: "Online", icon: Globe },
  { value: "Corporate_Partner", label: "Corporate", icon: Building2 },
  { value: "Referral", label: "Referral", icon: Users },
  { value: "Broker", label: "Broker", icon: Handshake },
];

const brokerTypes = ["Independent", "Tied", "Corporate", "Direct"];
const deductibleTiers = ["Low", "Medium", "High"];
const months = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

export function BrokerInput({ onSubmit }: { onSubmit: (data: BrokerData) => void }) {
  const [channel, setChannel] = useState("Online");
  const [brokerType, setBrokerType] = useState("Independent");
  const [deductible, setDeductible] = useState("Medium");
  const [month, setMonth] = useState(1);
  const [riders, setRiders] = useState(0);
  const [vehicles, setVehicles] = useState(1);

  return (
    <div className="glass rounded-xl p-5 space-y-5">
      {/* Acquisition Channel */}
      <div className="space-y-2">
        <label className="text-muted-foreground text-xs font-mono uppercase tracking-wider">Acquisition Channel</label>
        <div className="grid grid-cols-2 gap-2">
          {channels.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              onClick={() => setChannel(value)}
              suppressHydrationWarning
              className={`flex items-center gap-2 px-3 py-3 rounded-lg text-sm font-medium transition-all ${channel === value
                  ? "bg-neon-cyan/20 text-neon-cyan border border-neon-cyan/40"
                  : "glass text-muted-foreground hover:text-foreground"
                }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Broker Agency Type */}
      <div className="space-y-2">
        <label className="text-muted-foreground text-xs font-mono uppercase tracking-wider">Broker Agency Type</label>
        <div className="flex flex-wrap gap-2">
          {brokerTypes.map((t) => (
            <button
              key={t}
              onClick={() => setBrokerType(t)}
              suppressHydrationWarning
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${brokerType === t
                  ? "bg-neon-pink/20 text-neon-pink border border-neon-pink/40"
                  : "glass text-muted-foreground hover:text-foreground"
                }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Deductible Tier */}
      <div className="space-y-2">
        <label className="text-muted-foreground text-xs font-mono uppercase tracking-wider">Deductible Tier</label>
        <div className="flex gap-2">
          {deductibleTiers.map((d) => (
            <button
              key={d}
              onClick={() => setDeductible(d)}
              suppressHydrationWarning
              className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all ${deductible === d
                  ? "bg-neon-green/20 text-neon-green border border-neon-green/40"
                  : "glass text-muted-foreground hover:text-foreground"
                }`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      {/* Policy Start Month */}
      <div className="space-y-2">
        <label className="text-muted-foreground text-xs font-mono uppercase tracking-wider">Policy Start Month</label>
        <div className="flex flex-wrap gap-1.5">
          {months.map((m, i) => (
            <button
              key={m}
              onClick={() => setMonth(i + 1)}
              suppressHydrationWarning
              className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-all ${month === i + 1
                  ? "bg-neon-amber/20 text-neon-amber border border-neon-amber/40"
                  : "glass text-muted-foreground hover:text-foreground"
                }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        <SliderField label="Custom Riders" value={riders} min={0} max={10} onChange={setRiders} />
        <SliderField label="Vehicles" value={vehicles} min={0} max={10} onChange={setVehicles} />
      </div>

      <button
        onClick={() =>
          onSubmit({
            Acquisition_Channel: channel,
            Broker_Agency_Type: brokerType,
            Deductible_Tier: deductible,
            Policy_Start_Month: month,
            Custom_Riders: riders,
            Vehicles: vehicles,
          })
        }
        suppressHydrationWarning
        className="w-full glass glass-hover glow-cyan rounded-xl px-4 py-3 flex items-center justify-center gap-2 text-neon-cyan font-semibold transition-all hover:scale-[1.02]"
      >
        Predict Bundle
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
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
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
        step={1}
        onValueChange={([v]) => onChange(v)}
        className="[&_[role=slider]]:bg-neon-cyan [&_[role=slider]]:border-neon-cyan/50 [&_.range]:bg-neon-cyan/40"
      />
    </div>
  );
}
