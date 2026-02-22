"use client";

import { useEffect, useRef } from "react";

export function Starfield() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    function resize() {
      canvas!.width = window.innerWidth;
      canvas!.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    const stars: { x: number; y: number; size: number; speed: number; alpha: number }[] = [];
    for (let i = 0; i < 200; i++) {
      stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 2,
        speed: 0.02 + Math.random() * 0.08,
        alpha: Math.random(),
      });
    }

    let frame = 0;
    let animId: number;

    function draw() {
      frame++;
      ctx.clearRect(0, 0, canvas!.width, canvas!.height);

      // Stars
      stars.forEach((s) => {
        s.alpha = 0.3 + Math.sin(frame * s.speed) * 0.5;
        ctx.fillStyle = `rgba(200, 220, 255, ${Math.max(0, s.alpha)})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
        ctx.fill();
      });

      // Neon grid on the bottom
      const gridY = canvas!.height * 0.7;
      const gridHeight = canvas!.height * 0.3;
      const gridLines = 20;
      const perspective = 0.8;

      ctx.strokeStyle = "rgba(0, 229, 255, 0.08)";
      ctx.lineWidth = 1;

      // Horizontal grid lines
      for (let i = 0; i <= gridLines; i++) {
        const t = i / gridLines;
        const y = gridY + t * t * gridHeight;
        const alpha = 0.03 + t * 0.08;
        ctx.strokeStyle = `rgba(0, 229, 255, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas!.width, y);
        ctx.stroke();
      }

      // Vertical grid lines (perspective)
      const vertLines = 30;
      for (let i = 0; i <= vertLines; i++) {
        const x = (i / vertLines) * canvas!.width;
        const topX = canvas!.width / 2 + (x - canvas!.width / 2) * (1 - perspective);
        ctx.strokeStyle = "rgba(0, 229, 255, 0.06)";
        ctx.beginPath();
        ctx.moveTo(topX, gridY);
        ctx.lineTo(x, canvas!.height);
        ctx.stroke();
      }

      animId = requestAnimationFrame(draw);
    }

    draw();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    />
  );
}
