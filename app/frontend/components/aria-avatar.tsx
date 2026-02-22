"use client";

import { useEffect, useRef } from "react";

type AriaState = "idle" | "thinking" | "happy" | "nervous" | "squinting";

interface AriaAvatarProps {
  size?: number;
  state?: AriaState;
  className?: string;
}

export function AriaAvatar({ size = 200, state = "idle", className = "" }: AriaAvatarProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const particles: { x: number; y: number; vx: number; vy: number; size: number; alpha: number; hue: number }[] = [];
    for (let i = 0; i < 60; i++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = 30 + Math.random() * 40;
      particles.push({
        x: Math.cos(angle) * dist,
        y: Math.sin(angle) * dist,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: 1 + Math.random() * 2,
        alpha: 0.3 + Math.random() * 0.7,
        hue: Math.random() > 0.5 ? 187 : 340,
      });
    }

    let frame = 0;

    function draw() {
      frame++;
      const cx = size / 2;
      const cy = size / 2;
      const currentState = stateRef.current;
      ctx.clearRect(0, 0, size, size);

      // Sphere glow
      const isThinking = currentState === "thinking";
      const pulseScale = isThinking
        ? 1 + Math.sin(frame * 0.15) * 0.15
        : 1 + Math.sin(frame * 0.03) * 0.05;

      const baseRadius = size * 0.18 * pulseScale;

      // Outer glow
      const glowColor =
        currentState === "happy" ? "0, 255, 100" :
        currentState === "nervous" ? "255, 80, 80" :
        currentState === "squinting" ? "255, 170, 0" :
        "0, 229, 255";

      const outerGlow = ctx.createRadialGradient(cx, cy, baseRadius * 0.5, cx, cy, baseRadius * 2.5);
      outerGlow.addColorStop(0, `rgba(${glowColor}, 0.3)`);
      outerGlow.addColorStop(0.5, `rgba(${glowColor}, 0.1)`);
      outerGlow.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = outerGlow;
      ctx.beginPath();
      ctx.arc(cx, cy, baseRadius * 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Core sphere
      const coreGrad = ctx.createRadialGradient(cx - baseRadius * 0.3, cy - baseRadius * 0.3, 0, cx, cy, baseRadius);
      coreGrad.addColorStop(0, `rgba(${glowColor}, 0.9)`);
      coreGrad.addColorStop(0.6, `rgba(${glowColor}, 0.4)`);
      coreGrad.addColorStop(1, `rgba(${glowColor}, 0.1)`);
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, baseRadius, 0, Math.PI * 2);
      ctx.fill();

      // Particles
      const speed = isThinking ? 3 : 1;
      particles.forEach((p) => {
        p.x += p.vx * speed;
        p.y += p.vy * speed;
        const dist = Math.sqrt(p.x * p.x + p.y * p.y);
        if (dist > size * 0.35) {
          const angle = Math.random() * Math.PI * 2;
          const d = 20 + Math.random() * 30;
          p.x = Math.cos(angle) * d;
          p.y = Math.sin(angle) * d;
        }
        const flickerAlpha = p.alpha * (0.5 + Math.sin(frame * 0.05 + p.x) * 0.5);
        const particleHue = p.hue;
        ctx.fillStyle = `hsla(${particleHue}, 100%, 70%, ${flickerAlpha})`;
        ctx.beginPath();
        ctx.arc(cx + p.x, cy + p.y, p.size * pulseScale, 0, Math.PI * 2);
        ctx.fill();
      });

      // Eyes
      const eyeY = cy - baseRadius * 0.1;
      const eyeSpacing = baseRadius * 0.4;
      const eyeSize = baseRadius * 0.12;

      if (currentState === "squinting") {
        // Squinting eyes (horizontal lines)
        ctx.strokeStyle = "#050510";
        ctx.lineWidth = 2;
        [-1, 1].forEach((dir) => {
          ctx.beginPath();
          ctx.moveTo(cx + dir * eyeSpacing - eyeSize * 1.2, eyeY);
          ctx.lineTo(cx + dir * eyeSpacing + eyeSize * 1.2, eyeY);
          ctx.stroke();
        });
      } else if (currentState === "happy") {
        // Wide happy eyes
        const wideSize = eyeSize * 1.6;
        ctx.fillStyle = "#050510";
        [-1, 1].forEach((dir) => {
          ctx.beginPath();
          ctx.arc(cx + dir * eyeSpacing, eyeY, wideSize, 0, Math.PI * 2);
          ctx.fill();
        });
        // Sparkle in eyes
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        [-1, 1].forEach((dir) => {
          ctx.beginPath();
          ctx.arc(cx + dir * eyeSpacing - 2, eyeY - 2, wideSize * 0.35, 0, Math.PI * 2);
          ctx.fill();
        });
      } else if (currentState === "nervous") {
        // Nervous eyes (slightly offset, uneven)
        ctx.fillStyle = "#050510";
        ctx.beginPath();
        ctx.arc(cx - eyeSpacing, eyeY - 2, eyeSize * 1.1, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cx + eyeSpacing, eyeY + 2, eyeSize * 0.9, 0, Math.PI * 2);
        ctx.fill();
      } else {
        // Normal eyes with blink
        const blinkCycle = frame % 180;
        const isBlinking = blinkCycle > 175;
        if (isBlinking) {
          ctx.strokeStyle = "#050510";
          ctx.lineWidth = 2;
          [-1, 1].forEach((dir) => {
            ctx.beginPath();
            ctx.moveTo(cx + dir * eyeSpacing - eyeSize, eyeY);
            ctx.lineTo(cx + dir * eyeSpacing + eyeSize, eyeY);
            ctx.stroke();
          });
        } else {
          ctx.fillStyle = "#050510";
          [-1, 1].forEach((dir) => {
            ctx.beginPath();
            ctx.arc(cx + dir * eyeSpacing, eyeY, eyeSize, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }

      // Mouth
      const mouthY = cy + baseRadius * 0.25;
      if (currentState === "happy") {
        ctx.strokeStyle = "#050510";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, mouthY - 4, baseRadius * 0.2, 0.1 * Math.PI, 0.9 * Math.PI);
        ctx.stroke();
      } else if (currentState === "nervous") {
        ctx.strokeStyle = "#050510";
        ctx.lineWidth = 2;
        // Wavy nervous mouth
        ctx.beginPath();
        for (let i = 0; i < 20; i++) {
          const mx = cx - baseRadius * 0.2 + (i / 20) * baseRadius * 0.4;
          const my = mouthY + Math.sin(i * 0.8 + frame * 0.1) * 2;
          if (i === 0) ctx.moveTo(mx, my);
          else ctx.lineTo(mx, my);
        }
        ctx.stroke();
      } else if (currentState === "thinking") {
        // Small "o" mouth
        ctx.strokeStyle = "#050510";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, mouthY, baseRadius * 0.08, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Ring around sphere
      if (isThinking) {
        const ringAngle = frame * 0.05;
        ctx.strokeStyle = `rgba(${glowColor}, ${0.3 + Math.sin(frame * 0.1) * 0.2})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.ellipse(cx, cy, baseRadius * 1.6, baseRadius * 0.5, ringAngle, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.ellipse(cx, cy, baseRadius * 1.8, baseRadius * 0.6, -ringAngle * 0.7, 0, Math.PI * 2);
        ctx.stroke();
      }

      animRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [size]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size }}
      className={className}
      aria-label="ARIA AI Avatar"
      role="img"
    />
  );
}
