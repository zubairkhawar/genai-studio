'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Film, Volume2, Image as ImageIcon, Settings, FolderOpen, ChevronLeft, ChevronRight, ListChecks } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';

interface SidebarProps {
  current: string;
  expanded: boolean;
  onToggle: () => void;
}

export function Sidebar({ current, expanded, onToggle }: SidebarProps) {
  const colors = useThemeColors();

  const navGroups = [
    {
      label: 'Generators',
      items: [
        { href: '/video', label: 'Video Generator', icon: Film },
        { href: '/image', label: 'Image Generator', icon: ImageIcon },
        { href: '/audio', label: 'Audio Generator', icon: Volume2 },
        { href: '/jobs', label: 'Jobs / Queue', icon: ListChecks },
      ],
    },
    {
      label: 'System',
      items: [
        { href: '/outputs', label: 'Outputs', icon: FolderOpen },
        { href: '/settings', label: 'Settings', icon: Settings },
      ],
    },
  ];

  return (
    <aside
      className={`group fixed left-0 top-0 h-full bg-white dark:bg-slate-800/50 text-gray-900 dark:text-slate-100 border-r border-gray-200 dark:border-slate-700 transition-all duration-300 ease-in-out ${expanded ? 'w-64' : 'w-16'} shadow-md z-30`}
    >
      <div className="flex items-center justify-between h-14 px-2 md:px-4 border-b">
        <div className="flex items-center">
          {expanded ? (
            <>
              <div className="w-8 h-8 rounded-md bg-accent-blue/20 flex items-center justify-center text-accent-blue font-bold">G</div>
              <span className="ml-3 font-semibold transition-all opacity-100 whitespace-nowrap">GenStudio</span>
            </>
          ) : (
            <button
              aria-label="Expand sidebar"
              onClick={onToggle}
              className="w-8 h-8 rounded-md bg-accent-blue/20 flex items-center justify-center text-accent-blue hover:bg-accent-blue/30 transition-colors"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          )}
        </div>
        {expanded && (
          <button
            aria-label="Collapse sidebar"
            onClick={onToggle}
            className="p-2 rounded-md hover:bg-white/5 hover:dark:bg-black/10"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>
        )}
      </div>
      <nav className="py-2 space-y-2">
        {navGroups.map(({ label, items }) => (
          <div key={label}>
            {expanded && (
              <div className={`px-3 py-2 text-xs uppercase tracking-wide text-gray-600 dark:text-slate-300`}>{label}</div>
            )}
            <div className="space-y-1">
              {items.map(({ href, label, icon: Icon }) => {
                const active = (current === href) || (href !== '/' && current.startsWith(href));
                return (
                  <Link
                    key={href}
                    href={href}
                    title={label}
                    className={`relative flex items-center mx-1 md:mx-2 rounded-lg ${expanded ? 'px-3' : 'px-2'} py-2 transition-all ${active ? 'text-accent-blue' : 'hover:bg-white/5 hover:dark:bg-black/10'}`}
                  >
                    {active && <span className="absolute left-0 inset-y-0 w-1 rounded-r bg-accent-blue shadow-glow-blue" />}
                    <Icon className={`h-5 w-5 flex-shrink-0 ${active ? 'text-accent-blue' : ''}`} />
                    <span className={`ml-3 text-sm transition-all ${expanded ? 'opacity-100' : 'opacity-0 w-0'} whitespace-nowrap`}>{label}</span>
                  </Link>
                );
              })}
            </div>
            <div className={`my-2 ${expanded ? 'mx-3' : 'mx-2'} h-px ${expanded ? 'border-gray-200 dark:border-slate-700' : 'bg-transparent'}`} />
          </div>
        ))}
      </nav>

      {/* Bottom spacer */}
      <div className="px-3 mt-auto hidden md:block" />
      
    </aside>
  );
}


