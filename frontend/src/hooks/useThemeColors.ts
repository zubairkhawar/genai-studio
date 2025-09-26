import { useTheme } from '@/contexts/ThemeContext';

export function useThemeColors() {
  const { theme } = useTheme();

  const colors = {
    // Background colors
    bg: {
      primary: theme === 'dark' ? 'bg-dark-bg-primary' : 'bg-light-bg-primary',
      secondary: theme === 'dark' ? 'bg-dark-bg-secondary' : 'bg-light-bg-secondary',
      card: theme === 'dark' ? 'bg-dark-card' : 'bg-light-card',
    },
    
    // Text colors
    text: {
      primary: theme === 'dark' ? 'text-dark-text-primary' : 'text-light-text-primary',
      secondary: theme === 'dark' ? 'text-dark-text-secondary' : 'text-light-text-secondary',
    },
    
    // Border colors
    border: theme === 'dark' ? 'border-dark-border' : 'border-light-border',
    
    // Accent colors
    accent: {
      blue: theme === 'dark' ? 'accent-blue' : 'accent-blue-light',
      violet: theme === 'dark' ? 'accent-violet' : 'accent-violet-light',
      green: theme === 'dark' ? 'accent-green' : 'accent-green-light',
      red: theme === 'dark' ? 'accent-red' : 'accent-red-light',
    },
    
    // Glass effect
    glass: {
      bg: theme === 'dark' 
        ? 'bg-dark-card/80 backdrop-blur-md' 
        : 'bg-white/80 backdrop-blur-md',
      border: theme === 'dark' 
        ? 'border-white/10' 
        : 'border-black/10',
      shadow: theme === 'dark' 
        ? 'shadow-glass' 
        : 'shadow-glass-light',
    },
    
    // Button styles
    button: {
      primary: theme === 'dark'
        ? 'bg-accent-blue hover:bg-accent-blue/90 text-white shadow-glow-blue'
        : 'bg-accent-blue-light hover:bg-accent-blue-light/90 text-white shadow-lg',
      secondary: theme === 'dark'
        ? 'bg-dark-bg-secondary hover:bg-dark-bg-secondary/80 text-dark-text-primary border border-dark-border'
        : 'bg-light-bg-secondary hover:bg-light-bg-secondary/80 text-light-text-primary border border-light-border',
      ghost: theme === 'dark'
        ? 'bg-transparent hover:bg-white/5 text-dark-text-primary'
        : 'bg-transparent hover:bg-black/5 text-light-text-primary',
    },
    
    // Card styles
    card: theme === 'dark'
      ? 'bg-dark-card/50 backdrop-blur-md border border-white/10 shadow-glass'
      : 'bg-white/80 backdrop-blur-md border border-black/10 shadow-glass-light',
  };

  return colors;
}
