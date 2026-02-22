"use client";

import { cn } from "@/lib/utils";
import { Bot, User } from "lucide-react";

interface ChatMessageProps {
  sender: "aria" | "user";
  text: string;
}

export function ChatMessage({ sender, text }: ChatMessageProps) {
  const isAria = sender === "aria";

  return (
    <div className={cn("flex gap-3 items-start", !isAria && "flex-row-reverse")}>
      <div
        className={cn(
          "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
          isAria ? "bg-neon-cyan/20" : "bg-neon-pink/20"
        )}
      >
        {isAria ? (
          <Bot className="w-4 h-4 text-neon-cyan" />
        ) : (
          <User className="w-4 h-4 text-neon-pink" />
        )}
      </div>
      <div
        className={cn(
          "glass rounded-xl px-4 py-3 max-w-md",
          isAria ? "rounded-tl-sm" : "rounded-tr-sm"
        )}
      >
        <p className="text-foreground text-sm leading-relaxed">{text}</p>
      </div>
    </div>
  );
}
