'use client';

import { useState } from 'react';
import { 
  Pause, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Sparkles,
  Zap
} from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
import { getApiUrl } from '@/config';

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

interface JobQueueProps {
  jobs: Job[];
  onJobSelect: (job: Job) => void;
  selectedJob: Job | null;
}

export function JobQueue({ jobs, onJobSelect, selectedJob }: JobQueueProps) {
  const [filter, setFilter] = useState<'all' | 'completed' | 'failed'>('all');
  const colors = useThemeColors();

  const filteredJobs = jobs.filter(job => {
    // Map backend statuses to frontend filters
    switch (filter) {
      case 'all':
        return true;
      case 'completed':
        return job.status === 'completed';
      case 'failed':
        return job.status === 'failed';
      default:
        return job.status === filter;
    }
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'queued':
        return <Clock className="h-4 w-4 text-yellow-500 animate-pulse" />;
      case 'processing':
        return <Zap className="h-4 w-4 text-accent-blue animate-pulse" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-accent-green" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-accent-red" />;
      case 'cancelled':
        return <Pause className="h-4 w-4 text-gray-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'queued':
        return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
      case 'processing':
        return 'bg-accent-blue/10 text-accent-blue border-accent-blue/20';
      case 'completed':
        return 'bg-accent-green/10 text-accent-green border-accent-green/20';
      case 'failed':
        return 'bg-accent-red/10 text-accent-red border-accent-red/20';
      case 'cancelled':
        return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
      default:
        return 'bg-gray-500/10 text-gray-600 border-gray-500/20';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };


  return (
    <div className="h-[600px] overflow-hidden flex flex-col">
      {/* Filter Tabs */}
      <div className={`flex border-b border-gray-200 dark:border-slate-700 px-6 bg-gradient-to-r from-transparent via-gray-50/50 dark:via-slate-800/50 to-transparent`}>
        {[
          { key: 'all', label: 'All Jobs', count: jobs.length, icon: Sparkles, color: 'accent-blue' },
          { key: 'completed', label: 'Completed', count: jobs.filter(j => j.status === 'completed').length, icon: CheckCircle, color: 'accent-green' },
          { key: 'failed', label: 'Failed', count: jobs.filter(j => j.status === 'failed').length, icon: XCircle, color: 'accent-red' },
        ].map(({ key, label, count, icon: Icon, color }) => (
          <button
            key={key}
            onClick={() => setFilter(key as 'all' | 'completed' | 'failed')}
            className={`relative flex items-center space-x-3 px-6 py-4 text-sm font-medium transition-all duration-300 group ${
              filter === key
                ? `text-${color} bg-${color}/10`
                : `text-gray-600 dark:text-slate-300 hover:text-gray-900 dark:hover:text-slate-100 hover:bg-gray-50 dark:hover:bg-slate-700/50`
            }`}
          >
            {filter === key && (
              <div className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-${color} to-${color}/50 rounded-full`} />
            )}
            <Icon className={`h-5 w-5 transition-transform duration-200 ${filter === key ? 'scale-110' : 'group-hover:scale-105'}`} />
            <span className="font-semibold">{label}</span>
            <span className={`px-3 py-1 rounded-full text-xs font-bold transition-all duration-200 ${
              filter === key 
                ? `bg-${color}/20 text-${color} shadow-lg` 
                : count > 0 
                  ? `bg-${color}/10 text-${color} group-hover:bg-${color}/20` 
                  : `bg-gray-50 dark:bg-slate-700/50 text-gray-600 dark:text-slate-300`
            }`}>
              {count}
            </span>
          </button>
        ))}
      </div>

      {/* Job List */}
      <div className="flex-1 overflow-y-auto">
        {filteredJobs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-center p-12">
            <div className="max-w-md">
              <div className="relative mb-8">
                <div className="w-24 h-24 mx-auto bg-gradient-to-br from-accent-blue/20 via-accent-violet/20 to-accent-green/20 rounded-3xl flex items-center justify-center shadow-2xl">
                  <Sparkles className="h-12 w-12 text-accent-blue animate-pulse" />
                </div>
              </div>
              <h3 className={`text-2xl font-bold text-gray-900 dark:text-slate-100 mb-4 bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent`}>
                {filter === 'all' ? 'No jobs yet' : `No ${filter} jobs yet`}
              </h3>
              <p className={`text-gray-600 dark:text-slate-300 mb-8 text-lg leading-relaxed`}>
                {filter === 'all' 
                  ? 'No jobs found. Create a new job to get started.'
                  : `No ${filter} jobs found. Try switching to another filter or create a new job.`
                }
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-2 p-4">
            {filteredJobs.map((job, index) => (
              <div
                key={job.job_id}
                className={`p-6 rounded-2xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 hover:scale-[1.01] cursor-pointer transition-all duration-300 group shadow-sm hover:shadow-xl ${
                  selectedJob?.job_id === job.job_id 
                    ? `bg-accent-blue/10 border-accent-blue/50 shadow-lg shadow-accent-blue/20` 
                    : 'hover:shadow-lg hover:border-accent-blue/30 hover:bg-gradient-to-br hover:from-white hover:to-gray-50 dark:hover:from-slate-800 dark:hover:to-slate-700/50'
                }`}
                onClick={() => onJobSelect(job)}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="space-y-5">
                  {/* Header with Status and Progress */}
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="relative">
                        {getStatusIcon(job.status)}
                        {job.status === 'processing' && (
                          <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent-blue rounded-full animate-ping"></div>
                        )}
                      </div>
                      <div className="flex flex-col space-y-2">
                        <span className={`inline-flex items-center px-4 py-2 rounded-xl text-sm font-bold border-2 ${getStatusColor(job.status)} shadow-lg`}>
                          {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                        </span>
                        {job.progress > 0 && (job.status === 'processing' || job.status === 'running') && (
                          <div className="flex flex-col space-y-2">
                            <div className="flex items-center space-x-3">
                              <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                                <div 
                                  className="bg-gradient-to-r from-accent-blue to-accent-violet h-2 rounded-full transition-all duration-500 shadow-lg"
                                  style={{ width: `${job.progress}%` }}
                                ></div>
                              </div>
                              <span className={`text-sm font-bold text-gray-900 dark:text-slate-100`}>{job.progress}%</span>
                            </div>
                            {job.message && (
                              <p className="text-xs text-gray-600 dark:text-gray-400 italic">{job.message}</p>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={`text-sm font-medium text-gray-600 dark:text-slate-300 bg-gray-100 dark:bg-slate-700 px-3 py-1 rounded-lg`}>
                        {new Date(job.created_at).toLocaleTimeString()}
                      </span>
                      {job.status === 'processing' && (
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-accent-blue rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Prompt Preview */}
                  {job.prompt && (
                    <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-slate-800 dark:to-slate-700 rounded-xl p-4 border border-gray-200/50 dark:border-slate-600/50">
                      <p className={`text-sm text-gray-900 dark:text-slate-100 line-clamp-2 font-medium leading-relaxed`}>
                        "{job.prompt}"
                      </p>
                    </div>
                  )}
                  
                  {/* Job Details */}
                  <div className="flex items-center justify-between">
                    <div className={`flex items-center space-x-4 text-xs text-gray-600 dark:text-slate-300`}>
                      <span className="flex items-center space-x-2 bg-gray-100 dark:bg-slate-700 px-3 py-1.5 rounded-lg">
                        <Sparkles className="h-3 w-3 text-accent-blue" />
                        <span className="font-medium">{job.model_type} • {job.model_name}</span>
                      </span>
                      <span className="bg-gray-100 dark:bg-slate-700 px-3 py-1.5 rounded-lg font-medium">
                        {formatDate(job.created_at)}
                      </span>
                    </div>
                    
                    {/* Status Indicator */}
                    <div className="flex items-center space-x-2">
                      {job.status === 'processing' && (
                        <div className="flex items-center space-x-2 bg-accent-blue/10 px-3 py-1.5 rounded-lg border border-accent-blue/20">
                          <div className="w-2 h-2 bg-accent-blue rounded-full animate-pulse"></div>
                          <span className="text-xs text-accent-blue font-semibold">Processing...</span>
                        </div>
                      )}
                      {job.status === 'completed' && (
                        <div className="flex items-center space-x-2 bg-accent-green/10 px-3 py-1.5 rounded-lg border border-accent-green/20">
                          <div className="w-2 h-2 bg-accent-green rounded-full"></div>
                          <span className="text-xs text-accent-green font-semibold">Ready</span>
                        </div>
                      )}
                      {job.status === 'failed' && (
                        <div className="flex items-center space-x-2 bg-accent-red/10 px-3 py-1.5 rounded-lg border border-accent-red/20">
                          <div className="w-2 h-2 bg-accent-red rounded-full"></div>
                          <span className="text-xs text-accent-red font-semibold">Error</span>
                        </div>
                      )}
                      {job.status === 'queued' && (
                        <div className="flex items-center space-x-2 bg-yellow-500/10 px-3 py-1.5 rounded-lg border border-yellow-500/20">
                          <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
                          <span className="text-xs text-yellow-600 font-semibold">Queued</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Error Message */}
                  {job.error && (
                    <div className={`flex items-start space-x-3 p-4 bg-gradient-to-r from-accent-red/10 to-red-100/50 dark:from-accent-red/10 dark:to-red-900/20 border border-accent-red/30 rounded-xl shadow-sm`}>
                      <XCircle className="h-5 w-5 text-accent-red flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="text-sm text-accent-red font-semibold mb-1">Generation Failed</p>
                        <p className="text-xs text-red-600 dark:text-red-400 leading-relaxed">{job.error}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
