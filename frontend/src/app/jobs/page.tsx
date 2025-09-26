"use client";

import { useEffect, useState } from 'react';
import { JobQueue } from '@/components/JobQueue';
import { useThemeColors } from '@/hooks/useThemeColors';
import { getApiUrl } from '@/config';
import { RefreshCw, Activity, TrendingUp, Clock, AlertCircle, Trash2 } from 'lucide-react';

interface Job {
  job_id: string;
  status: string;
  progress: number;
  output_file?: string;
  error?: string;
  message?: string;
  created_at: string;
  updated_at: string;
  prompt?: string;
  model_type?: string;
  model_name?: string;
}

export default function Page() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const colors = useThemeColors();

  const fetchJobs = async () => {
    setIsRefreshing(true);
    try {
      const res = await fetch(getApiUrl('/jobs'));
      if (res.ok) {
        const data = await res.json();
        setJobs(data);
      }
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  const clearAllJobs = async () => {
    if (!confirm('Are you sure you want to clear all jobs? This action cannot be undone.')) {
      return;
    }
    
    setIsClearing(true);
    try {
      const res = await fetch(getApiUrl('/jobs'), {
        method: 'DELETE',
      });
      if (res.ok) {
        const data = await res.json();
        console.log('Cleared jobs:', data);
        // Refresh the jobs list
        await fetchJobs();
      } else {
        console.error('Failed to clear jobs');
      }
    } catch (error) {
      console.error('Failed to clear jobs:', error);
    } finally {
      setIsClearing(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const t = setInterval(fetchJobs, 2000);
    return () => clearInterval(t);
  }, []);

  // Calculate stats (only show completed, failed)
  const stats = {
    completed: jobs.filter(j => j.status === 'completed').length,
    failed: jobs.filter(j => j.status === 'failed').length,
  };

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent">
            Job Queue
          </h1>
          <p className={`text-${colors.text.secondary} mt-1`}>
            Monitor and manage your AI generation jobs
          </p>
        </div>
        <button
          onClick={clearAllJobs}
          disabled={isClearing || jobs.length === 0}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl border border-red-200 dark:border-red-600 hover:bg-red-50 dark:hover:bg-red-700/50 transition-all duration-200 ${
            isClearing || jobs.length === 0 ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
          }`}
        >
          <Trash2 className={`h-4 w-4 ${isClearing ? 'animate-pulse' : ''}`} />
          <span className="text-sm font-medium text-red-600 dark:text-red-400">Clear All</span>
        </button>
      </div>

      {/* Stats Cards (only show completed, failed) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          { label: 'Completed', value: stats.completed, icon: TrendingUp, color: 'accent-green' },
          { label: 'Failed', value: stats.failed, icon: AlertCircle, color: 'accent-red' },
        ].map(({ label, value, icon: Icon, color }) => (
          <div
            key={label}
            className={`p-6 rounded-2xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 hover:scale-105 transition-all duration-300 group shadow-lg`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-sm font-medium text-gray-600 dark:text-slate-300 mb-1`}>
                  {label}
                </p>
                <p className={`text-2xl font-bold text-gray-900 dark:text-slate-100`}>
                  {value}
                </p>
              </div>
              <div className={`p-3 rounded-xl bg-${color}/10 group-hover:bg-${color}/20 transition-colors`}>
                <Icon className={`h-6 w-6 text-${color}`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Job Queue */}
      <div className={`rounded-2xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 overflow-hidden shadow-lg`}>
        <JobQueue jobs={jobs} onJobSelect={(j)=>setSelectedJob(j)} selectedJob={selectedJob} />
      </div>
    </div>
  );
}


