'use client';

import { useState } from 'react';
import { 
  Pause, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Download,
  Trash2,
  Eye,
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
  const [filter, setFilter] = useState<'all' | 'pending' | 'running' | 'completed' | 'failed'>('all');
  const colors = useThemeColors();

  const filteredJobs = jobs.filter(job => {
    if (filter === 'all') return true;
    
    // Map backend statuses to frontend filters
    switch (filter) {
      case 'pending':
        return job.status === 'queued' || job.status === 'pending';
      case 'running':
        return job.status === 'processing' || job.status === 'running';
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

  const handleDownload = async (job: Job) => {
    if (job.output_file) {
      const link = document.createElement('a');
      link.href = `${getApiUrl('')}/outputs/${job.output_file.split('/').pop()}`;
      link.download = job.output_file.split('/').pop() || 'output';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const handleCancel = async (jobId: string) => {
    try {
      const response = await fetch(getApiUrl(`/job/${jobId}`), {
        method: 'DELETE',
      });
      
      if (response.ok) {
        // Job cancelled successfully
        console.log('Job cancelled');
      }
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  return (
    <div className="h-[600px] overflow-hidden flex flex-col">
      {/* Filter Tabs */}
      <div className={`flex border-b border-gray-200 dark:border-slate-700 px-6 bg-gradient-to-r from-transparent via-gray-50/50 dark:via-slate-800/50 to-transparent`}>
        {[
          { key: 'all', label: 'All Jobs', count: jobs.length, icon: Sparkles, color: 'accent-blue' },
          { key: 'pending', label: 'Pending', count: jobs.filter(j => j.status === 'pending' || j.status === 'queued').length, icon: Clock, color: 'accent-violet' },
          { key: 'running', label: 'Running', count: jobs.filter(j => j.status === 'running' || j.status === 'processing').length, icon: Zap, color: 'accent-green' },
          { key: 'completed', label: 'Completed', count: jobs.filter(j => j.status === 'completed').length, icon: CheckCircle, color: 'accent-green' },
          { key: 'failed', label: 'Failed', count: jobs.filter(j => j.status === 'failed').length, icon: XCircle, color: 'accent-red' },
        ].map(({ key, label, count, icon: Icon, color }) => (
          <button
            key={key}
            onClick={() => setFilter(key as 'all' | 'pending' | 'running' | 'completed' | 'failed')}
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
                <div className="absolute -top-3 -right-3 w-8 h-8 bg-gradient-to-r from-accent-green to-accent-blue rounded-full flex items-center justify-center shadow-lg">
                  <span className="text-white text-sm">✨</span>
                </div>
                <div className="absolute -bottom-2 -left-2 w-6 h-6 bg-gradient-to-r from-accent-violet to-accent-blue rounded-full flex items-center justify-center">
                  <span className="text-white text-xs">⚡</span>
                </div>
              </div>
              <h3 className={`text-2xl font-bold text-gray-900 dark:text-slate-100 mb-4 bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent`}>
                {filter === 'all' ? 'Ready to Create Magic?' : `No ${filter} jobs yet`}
              </h3>
              <p className={`text-gray-600 dark:text-slate-300 mb-8 text-lg leading-relaxed`}>
                {filter === 'all' 
                  ? 'Start by entering a creative prompt to generate your first video or audio masterpiece!'
                  : `No ${filter} jobs found. Try switching to another filter or create a new job.`
                }
              </p>
              {filter === 'all' && (
                <div className="flex items-center justify-center space-x-3 text-accent-blue">
                  <div className="w-2 h-2 bg-accent-blue rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium">Enter a prompt to get started</span>
                  <div className="w-2 h-2 bg-accent-blue rounded-full animate-pulse"></div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-2 p-4">
            {filteredJobs.map((job, index) => (
              <div
                key={job.job_id}
                className={`p-6 rounded-2xl border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 hover:scale-[1.02] cursor-pointer transition-all duration-300 group ${
                  selectedJob?.job_id === job.job_id 
                    ? `bg-accent-blue/10 border-accent-blue/50 shadow-glow-blue` 
                    : 'hover:shadow-lg hover:border-accent-blue/30'
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
                    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
                      <p className={`text-sm text-gray-900 dark:text-slate-100 line-clamp-2 font-medium`}>
                        {job.prompt}
                      </p>
                    </div>
                  )}
                  
                  {/* Job Details */}
                  <div className="flex items-center justify-between">
                    <div className={`flex items-center space-x-4 text-xs text-gray-600 dark:text-slate-300`}>
                      <span className="flex items-center space-x-1">
                        <Sparkles className="h-3 w-3" />
                        <span>{job.model_type} • {job.model_name}</span>
                      </span>
                      <span>{formatDate(job.created_at)}</span>
                    </div>
                    
                    {/* Action Buttons */}
                    <div className="flex items-center space-x-2">
                      {job.status === 'completed' && job.output_file && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            // Handle download
                          }}
                          className="p-1.5 text-accent-green hover:bg-accent-green/10 rounded-lg transition-colors"
                          title="Download"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                      )}
                      {job.status === 'processing' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            // Handle cancel
                          }}
                          className="p-1.5 text-accent-red hover:bg-accent-red/10 rounded-lg transition-colors"
                          title="Cancel"
                        >
                          <XCircle className="h-4 w-4" />
                        </button>
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onJobSelect(job);
                        }}
                        className="p-1.5 text-accent-blue hover:bg-accent-blue/10 rounded-lg transition-colors"
                        title="View Details"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  {/* Error Message */}
                  {job.error && (
                    <div className={`flex items-center space-x-2 p-3 bg-accent-red/10 border border-accent-red/20 rounded-lg`}>
                      <XCircle className="h-4 w-4 text-accent-red flex-shrink-0" />
                      <p className="text-sm text-accent-red font-medium">{job.error}</p>
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
