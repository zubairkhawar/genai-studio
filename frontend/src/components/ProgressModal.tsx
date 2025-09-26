'use client';

import { useEffect, useState } from 'react';
import { X, CheckCircle, AlertCircle, Trash2, CloudDownload } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';

interface ModelProgress {
  name: string;
  size: string;
  progress: number;
  status: 'pending' | 'downloading' | 'completed' | 'error';
  speed?: string;
  eta?: string;
  downloaded_mb?: number;
  size_gb?: number;
  speed_mbps?: number;
  eta_seconds?: number;
}

interface ProgressModalProps {
  isOpen: boolean;
  onClose: () => void;
  onRetry?: () => void;
  title: string;
  description: string;
  progress: number;
  status: 'idle' | 'in_progress' | 'completed' | 'error';
  currentStep?: string;
  details?: string[];
  error?: string;
  type: 'download' | 'delete';
  modelProgress?: ModelProgress[];
}

export function ProgressModal({
  isOpen,
  onClose,
  onRetry,
  title,
  description,
  progress,
  status,
  currentStep,
  details = [],
  error,
  type,
  modelProgress = []
}: ProgressModalProps) {
  const colors = useThemeColors();
  const [showSuccess, setShowSuccess] = useState(false);

  useEffect(() => {
    if (status === 'completed') {
      setShowSuccess(true);
      const timer = setTimeout(() => {
        setShowSuccess(false);
        onClose();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [status, onClose]);

  if (!isOpen) return null;

  const getIcon = () => {
    if (showSuccess) return <CheckCircle className="h-8 w-8 text-green-500" />;
    if (status === 'error') return <AlertCircle className="h-8 w-8 text-red-500" />;
    if (type === 'download') return <CloudDownload className="h-8 w-8 text-accent-blue" />;
    return <Trash2 className="h-8 w-8 text-red-500" />;
  };

  const getStatusColor = () => {
    if (showSuccess) return 'text-green-500';
    if (status === 'error') return 'text-red-500';
    if (type === 'download') return 'text-accent-blue';
    return 'text-red-500';
  };

  return (
    <div className="fixed top-0 left-0 right-0 bottom-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4" style={{ width: '100vw', height: '100vh' }}>
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl max-w-md w-full p-6 animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-xl ${type === 'download' ? 'bg-accent-blue/10' : 'bg-red-500/10'}`}>
              {getIcon()}
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-slate-100">{title}</h3>
              <p className={`text-sm ${colors.text.secondary}`}>{description}</p>
            </div>
          </div>
          {status !== 'in_progress' && (
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
            >
              <X className="h-5 w-5 text-gray-500" />
            </button>
          )}
        </div>

        {/* Progress Section */}
        <div className="mb-6">
          {status === 'in_progress' && (
            <div className="flex items-center justify-center mb-4">
              <div className="relative w-24 h-24">
                <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                  {/* Background circle */}
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    className="text-gray-200 dark:text-slate-600"
                  />
                  {/* Progress circle */}
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    strokeLinecap="round"
                    strokeDasharray={`${2 * Math.PI * 40}`}
                    strokeDashoffset={`${2 * Math.PI * 40 * (1 - progress / 100)}`}
                    className={`transition-all duration-500 ease-out ${
                      type === 'download' ? 'text-accent-blue' : 'text-red-500'
                    }`}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className={`text-lg font-bold ${getStatusColor()}`}>
                    {Math.round(progress)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {status === 'completed' && (
            <div className="text-center mb-4">
              <div className="w-16 h-16 mx-auto mb-3 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center">
                <CheckCircle className="h-8 w-8 text-green-500" />
              </div>
              <p className="text-green-600 dark:text-green-400 font-medium">
                {type === 'download' ? 'Download Complete!' : 'Deletion Complete!'}
              </p>
            </div>
          )}

          {status === 'error' && (
            <div className="text-center mb-4">
              <div className="w-16 h-16 mx-auto mb-3 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
                <AlertCircle className="h-8 w-8 text-red-500" />
              </div>
              <p className="text-red-600 dark:text-red-400 font-medium">Operation Failed</p>
            </div>
          )}

          {/* Current Step */}
          {currentStep && status === 'in_progress' && (
            <div className="text-center mb-4">
              <p className={`text-sm font-medium ${getStatusColor()}`}>{currentStep}</p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            </div>
          )}

          {/* Model Progress Details */}
          {modelProgress.length > 0 && type === 'download' && (
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-gray-900 dark:text-slate-100">Model Download Progress:</h4>
              <div className="space-y-3">
                {modelProgress.map((model, index) => (
                  <div key={index} className="p-3 rounded-lg bg-gray-50 dark:bg-slate-700/50 border border-gray-200 dark:border-slate-600">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${
                          model.status === 'completed' ? 'bg-green-500' :
                          model.status === 'downloading' ? 'bg-accent-blue animate-pulse' :
                          model.status === 'error' ? 'bg-red-500' :
                          'bg-gray-400'
                        }`}></div>
                        <span className="text-sm font-medium text-gray-900 dark:text-slate-100">{model.name}</span>
                      </div>
                      <div className="text-xs text-gray-500 text-right">
                        {model.downloaded_mb && model.size_gb ? 
                          `${(model.downloaded_mb / 1024).toFixed(1)}GB / ${model.size_gb}GB` : 
                          model.size
                        }
                      </div>
                    </div>
                    
                    {model.status === 'downloading' && (
                      <div className="space-y-2">
                        <div className="w-full bg-gray-200 dark:bg-slate-600 rounded-full h-1.5">
                          <div 
                            className="bg-accent-blue h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${model.progress}%` }}
                          ></div>
                        </div>
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <span>{model.progress.toFixed(1)}%</span>
                          {model.speed_mbps && (
                            <span>{model.speed_mbps.toFixed(1)} MB/s</span>
                          )}
                          {model.eta_seconds && model.eta_seconds > 0 && (
                            <span>ETA: {Math.floor(model.eta_seconds / 60)}:{(model.eta_seconds % 60).toFixed(0).padStart(2, '0')}</span>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {model.status === 'completed' && (
                      <div className="flex items-center space-x-2 text-xs text-green-600 dark:text-green-400">
                        <CheckCircle className="h-3 w-3" />
                        <span>Download completed</span>
                      </div>
                    )}
                    
                    {model.status === 'error' && (
                      <div className="flex items-center space-x-2 text-xs text-red-600 dark:text-red-400">
                        <AlertCircle className="h-3 w-3" />
                        <span>Download failed</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Details */}
          {details.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-semibold text-gray-900 dark:text-slate-100">Details:</h4>
              <div className="space-y-1">
                {details.map((detail, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      type === 'download' ? 'bg-accent-blue' : 'bg-red-500'
                    }`}></div>
                    <span className="text-sm text-gray-600 dark:text-slate-300">{detail}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        {status === 'error' && (
          <div className="flex space-x-3">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-gray-100 dark:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg hover:bg-gray-200 dark:hover:bg-slate-600 transition-colors"
            >
              Close
            </button>
            <button
              onClick={onRetry || onClose}
              className="flex-1 px-4 py-2 bg-accent-blue text-white rounded-lg hover:bg-accent-blue-light transition-colors"
            >
              Retry
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
