'use client';

import { useMemo } from 'react';
import { usePathname } from 'next/navigation';
import { Sidebar } from '@/components/Sidebar';
import { Topbar } from '@/components/Topbar';
import { useState, useEffect } from 'react';

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname() || '/';

  const title = useMemo(() => {
    if (pathname.startsWith('/video')) return 'Text-to-Video Generator';
    if (pathname.startsWith('/audio')) return 'Text-to-Audio Generator';
    if (pathname.startsWith('/outputs')) return 'Outputs';
    if (pathname.startsWith('/settings')) return 'Settings';
    if (pathname.startsWith('/jobs')) return 'Active Jobs';
    return 'Dashboard';
  }, [pathname]);

  const [expanded, setExpanded] = useState(true);

  useEffect(() => {
    const saved = localStorage.getItem('sidebar-expanded');
    if (saved !== null) setExpanded(saved === 'true');
  }, []);

  useEffect(() => {
    localStorage.setItem('sidebar-expanded', String(expanded));
  }, [expanded]);

  return (
    <>
      <Sidebar current={pathname} expanded={expanded} onToggle={() => setExpanded(!expanded)} />
      <div 
        className="h-screen flex flex-col transition-all duration-300 ease-in-out"
        style={{ marginLeft: expanded ? '16rem' : '4rem' }}
      >
        <Topbar title={title} sidebarExpanded={expanded} />
        <main className="flex-1 overflow-y-auto p-4 md:p-6 mt-14">
          {children}
        </main>
      </div>
    </>
  );
}


