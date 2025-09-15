"use client";

import { useEffect, useState } from 'react';
import { JobQueue } from '@/components/JobQueue';
import { useThemeColors } from '@/hooks/useThemeColors';
import { RefreshCw, Activity, TrendingUp, Clock, AlertCircle } from 'lucide-react';

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
  const colors = useThemeColors();

  const fetchJobs = async () => {
    setIsRefreshing(true);
    try {
      const res = await fetch('http://localhost:8000/jobs');
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

  useEffect(() => {
    fetchJobs();
    const t = setInterval(fetchJobs, 2000);
    return () => clearInterval(t);
  }, []);

  // Calculate stats
  const stats = {
    total: jobs.length,
    pending: jobs.filter(j => j.status === 'pending' || j.status === 'queued').length,
    running: jobs.filter(j => j.status === 'running' || j.status === 'processing').length,
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
          onClick={fetchJobs}
          disabled={isRefreshing}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl border border-gray-200 dark:border-slate-600 hover:bg-gray-50 dark:hover:bg-slate-700/50 transition-all duration-200 ${
            isRefreshing ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
          }`}
        >
          <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span className="text-sm font-medium">Refresh</span>
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        {[
          { label: 'Total Jobs', value: stats.total, icon: Activity, color: 'accent-blue' },
          { label: 'Pending', value: stats.pending, icon: Clock, color: 'accent-violet' },
          { label: 'Running', value: stats.running, icon: TrendingUp, color: 'accent-green' },
          { label: 'Completed', value: stats.completed, icon: Activity, color: 'accent-green' },
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


