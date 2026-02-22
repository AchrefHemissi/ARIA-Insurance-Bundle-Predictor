"use client";

import { useState, useEffect, useCallback } from "react";

export function useTypewriter(text: string, speed = 30, startDelay = 0) {
  const [displayed, setDisplayed] = useState("");
  const [isDone, setIsDone] = useState(false);

  useEffect(() => {
    setDisplayed("");
    setIsDone(false);
    let i = 0;
    let timeout: NodeJS.Timeout;

    const delayTimeout = setTimeout(() => {
      function type() {
        if (i < text.length) {
          setDisplayed(text.slice(0, i + 1));
          i++;
          timeout = setTimeout(type, speed);
        } else {
          setIsDone(true);
        }
      }
      type();
    }, startDelay);

    return () => {
      clearTimeout(delayTimeout);
      clearTimeout(timeout);
    };
  }, [text, speed, startDelay]);

  const skip = useCallback(() => {
    setDisplayed(text);
    setIsDone(true);
  }, [text]);

  return { displayed, isDone, skip };
}
