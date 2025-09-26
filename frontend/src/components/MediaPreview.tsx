'use client';

// import { useState } from 'react';
import { Download, Volume2, Sparkles, AlertCircle, Loader2 } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';

interface Job {
  job_id: string;
  status: string;
  progress: number;
  output_file?: string;
  error?: string;
  created_at: string;
  updated_at: string;
  prompt?: string;
  model_type?: string;
  model_name?: string;
}

interface MediaPreviewProps {
  job: Job;
}

export function MediaPreview({ job }: MediaPreviewProps) {
  // const [isPlaying, setIsPlaying] = useState(false);
  // const [isMuted, setIsMuted] = useState(false);
  const colors = useThemeColors();

  const handleDownload = () => {
    if (job.output_file) {
      const link = document.createElement('a');
      link.href = `http://localhost:8000/outputs/${job.output_file.split('/').pop()}`;
      link.download = job.output_file.split('/').pop() || 'output';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // Media controls would be implemented here
  // const togglePlayPause = () => {
  //   setIsPlaying(!isPlaying);
  // };

  // const toggleMute = () => {
  //   setIsMuted(!isMuted);
  // };

  if (job.status === 'failed') {
    return (
      <div className={`p-8 text-center`}>
        <div className="relative mb-6">
          <AlertCircle className="h-16 w-16 mx-auto text-accent-red animate-pulse" />
          <div className="absolute inset-0 h-16 w-16 mx-auto rounded-full border-2 border-accent-red/20 animate-ping"></div>
        </div>
        <h3 className={`text-xl font-bold text-gray-900 dark:text-slate-100 mb-3`}>Generation Failed</h3>
        <div className={`p-4 bg-accent-red/10 border border-accent-red/20 rounded-xl mb-6`}>
          <p className={`text-gray-600 dark:text-slate-300 font-medium`}>{job.error}</p>
        </div>
        <button
          onClick={() => window.location.reload()}
          className={`px-6 py-3 bg-accent-blue text-white hover:bg-accent-blue/90 rounded-xl font-semibold hover:scale-105 transition-all duration-200`}
        >
          Try Again
        </button>
      </div>
    );
  }

  if (job.status === 'processing') {
    return (
      <div className={`p-8 text-center`}>
        <div className="relative mb-6">
          <Loader2 className="h-16 w-16 mx-auto text-accent-blue animate-spin" />
          <div className="absolute inset-0 h-16 w-16 mx-auto rounded-full border-2 border-accent-blue/20 animate-pulse"></div>
        </div>
        <h3 className={`text-xl font-bold text-gray-900 dark:text-slate-100 mb-3`}>
          Creating {job.model_type === 'video' ? 'Video' : 'Audio'} Magic...
        </h3>
        <p className={`text-gray-600 dark:text-slate-300 mb-6 font-medium`}>{job.prompt}</p>
        <div className={`w-full bg-${colors.bg.secondary} rounded-full h-3 mb-4 overflow-hidden`}>
          <div
            className="bg-gradient-to-r from-accent-blue to-accent-violet h-3 rounded-full transition-all duration-500 relative"
            style={{ width: `${job.progress}%` }}
          >
            <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
          </div>
        </div>
        <p className={`text-sm font-semibold text-gray-600 dark:text-slate-300`}>{job.progress}% complete</p>
      </div>
    );
  }

  if (job.status === 'queued') {
    return (
      <div className={`p-8 text-center`}>
        <div className="relative mb-6">
          <div className="h-16 w-16 mx-auto text-yellow-500 animate-pulse">
            <svg className="h-16 w-16 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="absolute inset-0 h-16 w-16 mx-auto rounded-full border-2 border-yellow-500/20 animate-ping"></div>
        </div>
        <h3 className={`text-xl font-bold text-gray-900 dark:text-slate-100 mb-3`}>Job Queued</h3>
        <p className={`text-gray-600 dark:text-slate-300 mb-4 font-medium`}>{job.prompt}</p>
        <p className={`text-sm text-gray-600 dark:text-slate-300`}>Waiting to start generation...</p>
      </div>
    );
  }

  if (job.status !== 'completed' || !job.output_file) {
    return (
      <div className={`p-8 text-center text-gray-600 dark:text-slate-300`}>
        <div className="relative mb-4">
          <Sparkles className="h-12 w-12 mx-auto text-accent-blue/50" />
          <div className="absolute inset-0 h-12 w-12 mx-auto rounded-full border-2 border-accent-blue/20 animate-pulse"></div>
        </div>
        <p className="text-lg font-medium">No preview available</p>
      </div>
    );
  }

  const isVideo = job.model_type === 'video';
  const fileName = job.output_file.split('/').pop() || 'output';
  const fileUrl = `http://localhost:8000/outputs/${fileName}`;

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-2 bg-accent-green/10 rounded-lg">
            <Sparkles className="h-5 w-5 text-accent-green" />
          </div>
          <h3 className={`text-xl font-bold text-gray-900 dark:text-slate-100`}>Preview</h3>
        </div>
        <p className={`text-gray-600 dark:text-slate-300 font-medium`}>{job.prompt}</p>
      </div>

      <div className={`bg-${colors.bg.secondary} rounded-2xl overflow-hidden shadow-lg`}>
        {isVideo ? (
          <div className="relative group">
            <video
              className="w-full h-auto"
              controls
              onPlay={() => {}}
              onPause={() => {}}
            >
              <source src={fileUrl} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
            <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          </div>
        ) : (
          <div className="p-8 text-center">
            <div className="mb-6">
              <div className="relative">
                <Volume2 className="h-20 w-20 mx-auto text-accent-violet animate-pulse" />
                <div className="absolute inset-0 h-20 w-20 mx-auto rounded-full border-2 border-accent-violet/20 animate-ping"></div>
              </div>
            </div>
            <audio
              className="w-full"
              controls
              onPlay={() => {}}
              onPause={() => {}}
            >
              <source src={fileUrl} type="audio/mpeg" />
              <source src={fileUrl} type="audio/wav" />
              Your browser does not support the audio element.
            </audio>
          </div>
        )}
      </div>

      <div className="mt-6 flex items-center justify-between">
        <div className={`flex items-center space-x-3 text-sm text-gray-600 dark:text-slate-300`}>
          <div className="flex items-center space-x-1">
            <Sparkles className="h-4 w-4" />
            <span>{job.model_name}</span>
          </div>
          <span>•</span>
          <span className="font-mono">{fileName}</span>
        </div>
        
        <button
          onClick={handleDownload}
          className={`flex items-center space-x-2 px-6 py-3 bg-accent-blue text-white hover:bg-accent-blue/90 rounded-xl font-semibold hover:scale-105 transition-all duration-200 group`}
        >
          <Download className="h-4 w-4 group-hover:animate-bounce" />
          <span>Download</span>
        </button>
      </div>
    </div>
  );
}
