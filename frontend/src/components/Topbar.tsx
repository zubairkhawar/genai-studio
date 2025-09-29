'use client';

import { Sun, Moon } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { useThemeColors } from '@/hooks/useThemeColors';
import Image from 'next/image';

interface TopbarProps {
  title?: string;
  sidebarExpanded?: boolean;
}

export function Topbar({ title, sidebarExpanded = false }: TopbarProps) {
  const { theme, toggleTheme } = useTheme();
  const colors = useThemeColors();

  return (
    <header 
      className={`fixed top-0 right-0 z-20 bg-white dark:bg-slate-800/50 text-gray-900 dark:text-slate-100 border-b border-gray-200 dark:border-slate-700 transition-all duration-300 ease-in-out`} 
      style={{ left: sidebarExpanded ? '16rem' : '4rem' }}
    >
      <div className="h-14 px-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Image
            src="/logo.png"
            alt="GenStudio"
            width={32}
            height={32}
            className="rounded-lg"
          />
          <div className="text-sm font-medium text-center truncate px-2">
            {title ?? ''}
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <button aria-label="Toggle theme" onClick={toggleTheme} className="p-2 rounded-lg hover:bg-white/5 hover:dark:bg-black/10 transition-colors">
            {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </button>
        </div>
      </div>
    </header>
  );
}


