import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark Mode Colors - Modern and appealing
        'dark-bg-primary': '#0f0f23',
        'dark-bg-secondary': '#1a1a2e',
        'dark-card': 'rgba(255, 255, 255, 0.12)',
        'dark-text-primary': '#ffffff',
        'dark-text-secondary': '#e2e8f0',
        'dark-border': '#2d3748',
        
        // Light Mode Colors
        'light-bg-primary': '#f8fafc',
        'light-bg-secondary': '#e2e8f0',
        'light-card': '#ffffff',
        'light-text-primary': '#0f172a',
        'light-text-secondary': '#475569',
        'light-border': '#cbd5e1',
        
        // Accent Colors - Enhanced for better visibility
        'accent-blue': '#60a5fa',
        'accent-blue-light': '#0284c7',
        'accent-violet': '#c084fc',
        'accent-violet-light': '#7c3aed',
        'accent-green': '#4ade80',
        'accent-green-light': '#65a30d',
        'accent-red': '#f87171',
        'accent-red-light': '#dc2626',
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'display': ['Poppins', 'system-ui', 'sans-serif'],
      },
      backdropBlur: {
        'xs': '2px',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'glow-blue': 'glowBlue 2s ease-in-out infinite alternate',
        'glow-violet': 'glowViolet 2s ease-in-out infinite alternate',
        'glow-green': 'glowGreen 2s ease-in-out infinite alternate',
        'ripple': 'ripple 0.6s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
        'wave': 'wave 1.5s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(56, 189, 248, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(56, 189, 248, 0.8)' },
        },
        glowBlue: {
          '0%': { boxShadow: '0 0 5px rgba(56, 189, 248, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(56, 189, 248, 0.8), 0 0 30px rgba(56, 189, 248, 0.6)' },
        },
        glowViolet: {
          '0%': { boxShadow: '0 0 5px rgba(167, 139, 250, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(167, 139, 250, 0.8), 0 0 30px rgba(167, 139, 250, 0.6)' },
        },
        glowGreen: {
          '0%': { boxShadow: '0 0 5px rgba(132, 204, 22, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(132, 204, 22, 0.8), 0 0 30px rgba(132, 204, 22, 0.6)' },
        },
        ripple: {
          '0%': { transform: 'scale(0)', opacity: '1' },
          '100%': { transform: 'scale(4)', opacity: '0' },
        },
        wave: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      boxShadow: {
        'glow-blue': '0 0 20px rgba(56, 189, 248, 0.3)',
        'glow-violet': '0 0 20px rgba(167, 139, 250, 0.3)',
        'glow-green': '0 0 20px rgba(132, 204, 22, 0.3)',
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        'glass-light': '0 8px 32px 0 rgba(0, 0, 0, 0.1)',
      },
    },
  },
  plugins: [],
};

export default config;
