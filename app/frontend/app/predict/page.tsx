"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { AriaAvatar } from "@/components/aria-avatar";
import { Starfield } from "@/components/starfield";
import { DemographicsInput } from "@/components/predict/demographics-input";
import { RiskInput } from "@/components/predict/risk-input";
import { BrokerInput } from "@/components/predict/broker-input";
import { PredictionResult } from "@/components/predict/prediction-result";
import { ChatMessage } from "@/components/predict/chat-message";
import { api, type PredictRequest, type ExplainResponse } from "@/lib/api";
import { RotateCcw } from "lucide-react";

type AriaState = "idle" | "thinking" | "happy" | "nervous" | "squinting";

interface Message {
  id: string;
  sender: "aria" | "user";
  text: string;
  widget?: "demographics" | "risk" | "broker" | "result" | "thinking";
}

const INTRO_MESSAGE =
  "Alright, let's figure out this client. I'll ask you a few questions and then I'll work my magic. First up...";
const DEMO_MESSAGE =
  "Who's the client? Give me the family situation and income -- I need to know if we're dealing with a solo adventurer or a full squad.";

function getReactionToDemographics(income: number, deps: number): string {
  if (income > 80000 && deps > 3) return "Rich AND a big family? This is going to be interesting...";
  if (income > 80000) return "Oof, big earner. They'll expect the premium treatment. Now tell me about their risk history...";
  if (deps > 3) return "A whole crew! Okay interesting... Now let me know about their risk profile.";
  if (deps === 0) return "Solo adventurer, got it. Let's check their risk history next...";
  return "Noted! Let's look at their risk profile next...";
}

function getReactionToRisk(claims: number, yearsClean: number): string {
  if (claims > 3) return "Yikes, they like filing claims don't they... Almost there! Last few details:";
  if (yearsClean > 5) return "Squeaky clean record! Nice. Last round of questions coming up:";
  if (claims === 0) return "Zero claims? A model citizen. Alright, one more set of questions:";
  return "Okay, I've got the picture. Just a few more details:";
}

const BROKER_MESSAGE = "Tell me about the acquisition channel, broker, and policy details.";

export default function PredictPage() {
  const [messages, setMessages] = useState<Message[]>([
    { id: "intro", sender: "aria", text: INTRO_MESSAGE },
    { id: "demo-ask", sender: "aria", text: DEMO_MESSAGE, widget: "demographics" },
  ]);
  const [ariaState, setAriaState] = useState<AriaState>("idle");
  const [step, setStep] = useState<"demographics" | "risk" | "broker" | "thinking" | "result">("demographics");
  const [formData, setFormData] = useState<Partial<PredictRequest>>({});
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleDemographicsSubmit = (data: {
    Adult_Dependents: number;
    Child_Dependents: number;
    Infant_Dependents: number;
    Annual_Income: number;
    Employment_Status: number;
  }) => {
    const totalDeps = data.Adult_Dependents + data.Child_Dependents + data.Infant_Dependents;
    const reaction = getReactionToDemographics(data.Annual_Income, totalDeps);
    setFormData((prev) => ({ ...prev, ...data }));
    setMessages((prev) => [
      ...prev,
      { id: `user-demo`, sender: "user", text: `${totalDeps} dependents, $${data.Annual_Income.toLocaleString()} income, ${data.Employment_Status ? "Employed" : "Unemployed"}` },
      { id: "risk-reaction", sender: "aria", text: reaction },
      { id: "risk-ask", sender: "aria", text: "Give me the claims history, policyholder status, and timing info.", widget: "risk" },
    ]);
    setStep("risk");
  };

  const handleRiskSubmit = (data: {
    Previous_Claims_Filed: number;
    Years_Without_Claims: number;
    Existing_Policyholder: number;
    Grace_Period_Extensions: number;
    Days_Since_Quote: number;
  }) => {
    const reaction = getReactionToRisk(data.Previous_Claims_Filed, data.Years_Without_Claims);
    setFormData((prev) => ({ ...prev, ...data }));
    setMessages((prev) => [
      ...prev,
      { id: `user-risk`, sender: "user", text: `${data.Previous_Claims_Filed} claims, ${data.Years_Without_Claims}yr clean, ${data.Existing_Policyholder ? "existing" : "new"} policyholder` },
      { id: "broker-reaction", sender: "aria", text: reaction },
      { id: "broker-ask", sender: "aria", text: BROKER_MESSAGE, widget: "broker" },
    ]);
    setStep("broker");
  };

  const handleBrokerSubmit = async (data: {
    Acquisition_Channel: string;
    Broker_Agency_Type: string;
    Deductible_Tier: string;
    Policy_Start_Month: number;
    Custom_Riders: number;
    Vehicles: number;
  }) => {
    const fullData = { ...formData, ...data } as PredictRequest;
    setFormData(fullData);
    setStep("thinking");
    setAriaState("thinking");
    setMessages((prev) => [
      ...prev,
      { id: `user-broker`, sender: "user", text: `${data.Acquisition_Channel} channel, ${data.Broker_Agency_Type} broker, ${data.Deductible_Tier} deductible` },
      { id: "thinking", sender: "aria", text: "Okay okay okay... processing... accessing the neural matrix... don't rush me...", widget: "thinking" },
    ]);

    try {
      const res = await api.explain(fullData);
      setResult(res);
      setStep("result");

      const conf = res.confidence;
      let reaction: string;
      let newState: AriaState;
      if (conf >= 0.6) {
        reaction = `BUNDLE ${res.predicted_bundle}! Called it. I am genuinely built different. Confidence: ${(conf * 100).toFixed(0)}% -- that's basically certainty.`;
        newState = "happy";
      } else if (conf >= 0.35) {
        reaction = `I'm gonna say Bundle ${res.predicted_bundle}... I mean, I'm like ${(conf * 100).toFixed(0)}% sure. Which is a lot! Probably. Trust me.`;
        newState = "squinting";
      } else {
        reaction = `Okay so... Bundle ${res.predicted_bundle}? Don't quote me on that. ${(conf * 100).toFixed(0)}% confidence which is... fine. It's fine.`;
        newState = "nervous";
      }

      setAriaState(newState);

      // Build explanation
      const topFeatures = res.top_5_shap_features?.slice(0, 3) ?? [];
      let explanation = "";
      if (topFeatures.length > 0) {
        explanation = `The biggest reason? ${topFeatures[0].feature.replace(/_/g, " ")}. `;
        if (topFeatures[1]) explanation += `Also ${topFeatures[1].feature.replace(/_/g, " ")} pulled ${topFeatures[1].direction === "positive" ? "toward" : "against"} this result. `;
        if (topFeatures[2]) explanation += `${topFeatures[2].feature.replace(/_/g, " ")} was ${topFeatures[2].direction === "negative" ? "actually working against them a bit but not enough to change my mind." : "supporting this prediction too."}`;
      }

      setMessages((prev) => {
        const filtered = prev.filter((m) => m.id !== "thinking");
        return [
          ...filtered,
          { id: "result-msg", sender: "aria", text: reaction, widget: "result" },
          ...(explanation ? [{ id: "explain-msg", sender: "aria", text: explanation }] : []),
        ];
      });
    } catch (err) {
      setAriaState("nervous");
      setMessages((prev) => {
        const filtered = prev.filter((m) => m.id !== "thinking");
        return [
          ...filtered,
          { id: "error", sender: "aria", text: `Hmm, something went wrong. ${err instanceof Error ? err.message : "The neural matrix hiccupped."} Try again?` },
        ];
      });
      setStep("demographics");
    }
  };

  const handleReset = () => {
    setMessages([
      { id: "intro-2", sender: "aria", text: "Fresh start! Let's predict another bundle. Who's the new client?" },
      { id: "demo-ask-2", sender: "aria", text: DEMO_MESSAGE, widget: "demographics" },
    ]);
    setStep("demographics");
    setAriaState("idle");
    setFormData({});
    setResult(null);
  };

  return (
    <div className="relative min-h-screen flex">
      <Starfield />

      {/* Chat panel */}
      <div className="relative z-10 flex-1 flex flex-col max-w-2xl mx-auto px-4 py-6">
        <div className="flex-1 overflow-y-auto space-y-4 pb-4">
          {messages.map((msg) => (
            <div key={msg.id}>
              <ChatMessage sender={msg.sender} text={msg.text} />
              {msg.widget === "demographics" && step === "demographics" && (
                <div className="ml-12 mt-3 animate-slide-up">
                  <DemographicsInput onSubmit={handleDemographicsSubmit} />
                </div>
              )}
              {msg.widget === "risk" && step === "risk" && (
                <div className="ml-12 mt-3 animate-slide-up">
                  <RiskInput onSubmit={handleRiskSubmit} />
                </div>
              )}
              {msg.widget === "broker" && step === "broker" && (
                <div className="ml-12 mt-3 animate-slide-up">
                  <BrokerInput onSubmit={handleBrokerSubmit} />
                </div>
              )}
              {msg.widget === "thinking" && step === "thinking" && (
                <div className="ml-12 mt-3 flex items-center gap-3">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-neon-cyan rounded-full animate-bounce" style={{ animationDelay: "0s" }} />
                    <span className="w-2 h-2 bg-neon-cyan rounded-full animate-bounce" style={{ animationDelay: "0.15s" }} />
                    <span className="w-2 h-2 bg-neon-cyan rounded-full animate-bounce" style={{ animationDelay: "0.3s" }} />
                  </div>
                </div>
              )}
              {msg.widget === "result" && result && (
                <div className="ml-12 mt-3 animate-slide-up">
                  <PredictionResult result={result} />
                </div>
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        {step === "result" && (
          <div className="relative z-10 py-4 flex justify-center">
            <button
              onClick={handleReset}
              className="glass glass-hover glow-cyan rounded-xl px-6 py-3 flex items-center gap-2 text-neon-cyan font-semibold transition-all hover:scale-105"
            >
              <RotateCcw className="w-4 h-4" />
              Predict Another Client
            </button>
          </div>
        )}
      </div>

      {/* ARIA panel (desktop only) */}
      <div className="relative z-10 hidden lg:flex flex-col items-center justify-center w-80 pr-8">
        <div className="sticky top-32">
          <AriaAvatar size={280} state={ariaState} />
          <p className="text-center text-muted-foreground text-sm mt-4 font-mono">
            ARIA v2.0
          </p>
        </div>
      </div>
    </div>
  );
}
