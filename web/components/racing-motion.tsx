"use client";

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

const MotionContext = createContext({ paused: false, toggle: () => {} });
export function RacingMotion({ children }: { children: ReactNode }) {
  const [paused, setPaused] = useState(false);
  const halo = useRef<HTMLDivElement>(null);
  const trail = useRef<HTMLDivElement>(null);
  const progress = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const startup = window.setTimeout(() => {
      try {
        setPaused(
          window.localStorage.getItem("apex-pulse-motion") === "paused",
        );
      } catch {
        /* Persistence is optional. */
      }
    }, 0);
    return () => window.clearTimeout(startup);
  }, []);
  useEffect(() => {
    document.documentElement.classList.toggle("motion-paused", paused);
    return () => document.documentElement.classList.remove("motion-paused");
  }, [paused]);
  useEffect(() => {
    const media = window.matchMedia?.(
      "(prefers-reduced-motion: reduce), (pointer: coarse)",
    );
    if (!media || paused) return;
    const move = (event: PointerEvent) => {
      if (media.matches || event.pointerType === "touch") return;
      if (halo.current) {
        halo.current.style.opacity = "1";
        halo.current.style.transform = `translate(${event.clientX - 12}px,${event.clientY - 12}px)`;
        halo.current.classList.toggle(
          "over-control",
          Boolean(
            (event.target as Element)?.closest("a,button,input,select,summary"),
          ),
        );
      }
      if (trail.current) {
        trail.current.style.transform = `translate(${event.clientX}px,${event.clientY}px)`;
        trail.current.style.opacity = ".55";
      }
    };
    const hide = () => {
      if (halo.current) halo.current.style.opacity = "0";
      if (trail.current) trail.current.style.opacity = "0";
    };
    const scroll = () => {
      const total = document.documentElement.scrollHeight - window.innerHeight;
      if (progress.current)
        progress.current.style.width = `${total > 0 ? (window.scrollY / total) * 100 : 0}%`;
    };
    window.addEventListener("pointermove", move, { passive: true });
    document.addEventListener("pointerleave", hide);
    window.addEventListener("scroll", scroll, { passive: true });
    return () => {
      window.removeEventListener("pointermove", move);
      document.removeEventListener("pointerleave", hide);
      window.removeEventListener("scroll", scroll);
      hide();
    };
  }, [paused]);
  return (
    <MotionContext.Provider
      value={{
        paused,
        toggle: () => {
          const next = !paused;
          setPaused(next);
          try {
            window.localStorage.setItem(
              "apex-pulse-motion",
              next ? "paused" : "playing",
            );
          } catch {
            /* Persistence is optional. */
          }
        },
      }}
    >
      <div ref={progress} className="scroll-progress" aria-hidden="true" />
      <div ref={halo} className="cursor-halo" aria-hidden="true" />
      <div ref={trail} className="cursor-spark" aria-hidden="true" />
      {children}
    </MotionContext.Provider>
  );
}

export function MotionToggle() {
  const { paused, toggle } = useContext(MotionContext);
  return (
    <button
      className="icon-button"
      type="button"
      onClick={toggle}
      aria-pressed={paused}
      aria-label={paused ? "Resume animations" : "Pause animations"}
      title={paused ? "Resume animations" : "Pause animations"}
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 16 16"
        aria-hidden="true"
        fill="currentColor"
      >
        {paused ? (
          <path d="m5 3 8 5-8 5Z" />
        ) : (
          <path d="M4 3h3v10H4zm5 0h3v10H9z" />
        )}
      </svg>
    </button>
  );
}
