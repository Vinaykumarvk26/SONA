/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg0: "#06090f",
        bg1: "#0d1322",
        bg2: "#121a2b",
        text0: "#e2e8f0",
        text1: "#94a3b8",
        accent: "#38bdf8",
        accent2: "#22d3ee",
        success: "#34d399",
        warn: "#fb7185",
      },
      fontFamily: {
        heading: ["Sora", "sans-serif"],
        body: ["Manrope", "sans-serif"],
      },
      keyframes: {
        floatIn: {
          "0%": { opacity: "0", transform: "translateY(12px) scale(0.98)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" },
        },
        pulseSoft: {
          "0%, 100%": { transform: "scale(1)", opacity: "1" },
          "50%": { transform: "scale(1.08)", opacity: "0.9" },
        },
      },
      animation: {
        floatIn: "floatIn 420ms ease-out",
        pulseSoft: "pulseSoft 1.8s ease-in-out infinite",
      },
      boxShadow: {
        glass: "0 12px 30px rgba(2, 10, 24, 0.38)",
      },
    },
  },
  plugins: [],
};
