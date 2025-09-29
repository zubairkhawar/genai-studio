'use client';

import { useState, useEffect } from 'react';
import { X, Download, AlertTriangle, CheckCircle, Loader2, CloudDownload, HardDrive, Clock } from 'lucide-react';
import { getApiUrl } from '@/config';

interface Model {
  id: string;
  name: string;
  description: string;
  max_duration: number;
  resolution?: string;
  sample_rate?: number;
  size_gb?: number;
  loaded: boolean;
}

interface ModelDownloadModalProps {
  isOpen: boolean;
  onClose: () => void;
  missingModels: Model[];
  modelType: 'video' | 'audio' | 'image';
  onModelsDownloaded?: () => void;
}

interface DownloadStatus {
  is_downloading: boolean;
  overall_progress: number;
  current_model: string;
  status: string;
  message: string;
  error: string | null;
  models?: {
    [key: string]: {
      name: string;
      repo_id?: string;
      local_dir?: string;
      size_gb?: number;
      status: string;
      progress: number;
      downloaded_mb: number;
      eta_seconds?: number;
      files_verified?: boolean;
    };
  };
}

export default function ModelDownloadModal({
  isOpen,
  onClose,
  missingModels,
  modelType,
  onModelsDownloaded
}: ModelDownloadModalProps) {
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus>({
    is_downloading: false,
    overall_progress: 0,
    current_model: "",
    status: "idle",
    message: "",
    error: null
  });
  const [downloadingModels, setDownloadingModels] = useState<string[]>([]);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortController) {
        abortController.abort();
      }
    };
  }, [abortController]);

  const getModelTypeInfo = (type: 'video' | 'audio' | 'image') => {
    switch (type) {
      case 'video':
        return {
          title: 'Video Models',
          description: 'Download AI models for video generation',
          icon: '🎬',
          color: 'from-accent-blue to-accent-violet'
        };
      case 'audio':
        return {
          title: 'Audio Models',
          description: 'Download AI models for audio generation',
          icon: '🎵',
          color: 'from-accent-violet to-accent-blue'
        };
      case 'image':
        return {
          title: 'Image Models',
          description: 'Download AI models for image generation',
          icon: '🖼️',
          color: 'from-accent-green to-accent-blue'
        };
    }
  };

  const startDownload = async (modelId: string) => {
    if (downloadStatus.is_downloading) return;

    // Create new AbortController for this download
    const controller = new AbortController();
    setAbortController(controller);

    setDownloadingModels(prev => [...prev, modelId]);
    setDownloadStatus({
      is_downloading: true,
      overall_progress: 0,
      current_model: modelId,
      status: 'downloading',
      message: `Starting download for ${modelId}...`,
      error: null
    });

    try {
      const response = await fetch(getApiUrl(`/download-model-stream/${modelId}?force=false`), {
        headers: {
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache'
        },
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body reader available');
      }

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'log') {
                setDownloadStatus(prev => ({
                  ...prev,
                  message: data.message
                }));
              } else if (data.type === 'success') {
                setDownloadStatus(prev => ({
                  ...prev,
                  status: 'completed',
                  message: data.message,
                  is_downloading: false
                }));
                
                setDownloadingModels(prev => prev.filter(id => id !== modelId));
                
                // Check if all models are downloaded
                if (downloadingModels.length === 1) {
                  setTimeout(() => {
                    onModelsDownloaded?.();
                    onClose();
                  }, 2000);
                }
                break;
              } else if (data.type === 'error') {
                setDownloadStatus(prev => ({
                  ...prev,
                  status: 'error',
                  error: data.message,
                  is_downloading: false
                }));
                setDownloadingModels(prev => prev.filter(id => id !== modelId));
                break;
              }
            } catch (parseError) {
              console.error('Error parsing streaming data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error starting download:', error);
      
      // Handle cancellation
      if (error instanceof Error && error.name === 'AbortError') {
        setDownloadStatus(prev => ({
          ...prev,
          status: 'cancelled',
          message: 'Download cancelled by user',
          is_downloading: false
        }));
      } else {
        setDownloadStatus(prev => ({
          ...prev,
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error',
          is_downloading: false
        }));
      }
      setDownloadingModels(prev => prev.filter(id => id !== modelId));
    }
  };

  const downloadAllModels = async () => {
    if (downloadStatus.is_downloading) return;

    // Create new AbortController for this download
    const controller = new AbortController();
    setAbortController(controller);

    setDownloadingModels(missingModels.map(model => model.id));
    setDownloadStatus({
      is_downloading: true,
      overall_progress: 0,
      current_model: '',
      status: 'downloading',
      message: 'Starting download of all models...',
      error: null
    });

    try {
      const response = await fetch(getApiUrl('/download-models-stream?force=false'), {
        signal: controller.signal
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      if (!reader) {
        throw new Error('No response body reader available');
      }
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'log') {
                setDownloadStatus(prev => ({
                  ...prev,
                  message: data.message
                }));
              } else if (data.type === 'success') {
                setDownloadStatus(prev => ({
                  ...prev,
                  status: 'completed',
                  message: data.message,
                  is_downloading: false
                }));
                
                setDownloadingModels([]);
                
                setTimeout(() => {
                  onModelsDownloaded?.();
                  onClose();
                }, 2000);
                break;
              } else if (data.type === 'error') {
                setDownloadStatus(prev => ({
                  ...prev,
                  status: 'error',
                  error: data.message,
                  is_downloading: false
                }));
                setDownloadingModels([]);
                break;
              }
            } catch (parseError) {
              console.error('Error parsing streaming data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error starting download:', error);
      
      // Handle cancellation
      if (error instanceof Error && error.name === 'AbortError') {
        setDownloadStatus(prev => ({
          ...prev,
          status: 'cancelled',
          message: 'Download cancelled by user',
          is_downloading: false
        }));
      } else {
        setDownloadStatus(prev => ({
          ...prev,
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error',
          is_downloading: false
        }));
      }
      setDownloadingModels([]);
    }
  };

  const getTotalSize = () => {
    return missingModels.reduce((total, model) => {
      // Estimate size if not provided
      const estimatedSize = model.size_gb || getEstimatedSize(model.id);
      return total + estimatedSize;
    }, 0);
  };

  const getEstimatedSize = (modelId: string): number => {
    // Estimated sizes for models
    const sizeMap: { [key: string]: number } = {
      'stable-diffusion': 4.0,
      'animatediff': 3.4,
      'bark': 4.0
    };
    return sizeMap[modelId] || 2.0;
  };

  const cancelDownload = async () => {
    if (abortController) {
      // Cancel the fetch request
      abortController.abort();
      setAbortController(null);
    }

    // Also call backend cancel endpoint for any currently downloading model
    if (downloadStatus.current_model) {
      try {
        await fetch(getApiUrl(`/cancel-download/${downloadStatus.current_model}`), {
          method: 'POST'
        });
      } catch (error) {
        console.error('Error calling backend cancel endpoint:', error);
      }
    }

    // Reset download status
    setDownloadStatus({
      is_downloading: false,
      overall_progress: 0,
      current_model: "",
      status: "cancelled",
      message: "Download cancelled by user",
      error: null
    });
    setDownloadingModels([]);
  };

  const modelTypeInfo = getModelTypeInfo(modelType);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className={`p-6 bg-gradient-to-r ${modelTypeInfo.color} text-white`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="text-2xl">{modelTypeInfo.icon}</div>
              <div>
                <h2 className="text-xl font-bold">{modelTypeInfo.title} Required</h2>
                <p className="text-white/80 text-sm">{modelTypeInfo.description}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              disabled={downloadStatus.is_downloading}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors disabled:opacity-50"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 max-h-[60vh] overflow-y-auto">
          {/* Warning Message */}
          <div className="p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl">
            <div className="flex items-start space-x-3">
              <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-amber-900 dark:text-amber-100 mb-1">
                  Models Not Found
                </h3>
                <p className="text-sm text-amber-800 dark:text-amber-200">
                  The following {modelType} models need to be downloaded before you can generate content:
                </p>
              </div>
            </div>
          </div>

          {/* Missing Models List */}
          <div className="space-y-3">
            {missingModels.map((model) => (
              <div key={model.id} className="p-4 border border-gray-200 dark:border-slate-600 rounded-xl bg-gray-50 dark:bg-slate-700/50">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-900 dark:text-white">{model.name}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">{model.description}</p>
                    <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                      {model.resolution && (
                        <div className="flex items-center space-x-1">
                          <HardDrive className="h-3 w-3" />
                          <span>{model.resolution}</span>
                        </div>
                      )}
                      <div className="flex items-center space-x-1">
                        <CloudDownload className="h-3 w-3" />
                        <span>~{getEstimatedSize(model.id)}GB</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {downloadingModels.includes(model.id) ? (
                      <div className="flex items-center space-x-2 px-3 py-2 bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span className="text-sm">Downloading...</span>
                      </div>
                    ) : downloadStatus.status === 'completed' ? (
                      <div className="flex items-center space-x-2 px-3 py-2 bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg">
                        <CheckCircle className="h-4 w-4" />
                        <span className="text-sm">Downloaded</span>
                      </div>
                    ) : (
                      <button
                        onClick={() => startDownload(model.id)}
                        disabled={downloadStatus.is_downloading}
                        className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
                      >
                        <Download className="h-4 w-4" />
                        <span>Download</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Download Progress */}
                {downloadingModels.includes(model.id) && (
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                      <span>Downloading...</span>
                      <span>{downloadStatus.overall_progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${downloadStatus.overall_progress}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-1 truncate">{downloadStatus.message}</p>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Download All Button */}
          {missingModels.length > 1 && (
            <div className="pt-4 border-t border-gray-200 dark:border-slate-600">
              <button
                onClick={downloadAllModels}
                disabled={downloadStatus.is_downloading}
                className="w-full flex items-center justify-center space-x-3 px-6 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-500 text-white rounded-xl font-semibold transition-all duration-200 hover:scale-105 disabled:hover:scale-100"
              >
                <CloudDownload className="h-5 w-5" />
                <span>Download All Models (~{getTotalSize().toFixed(1)}GB)</span>
              </button>
            </div>
          )}

          {/* Error Message */}
          {downloadStatus.status === 'error' && downloadStatus.error && (
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
              <div className="flex items-start space-x-3">
                <AlertTriangle className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-red-900 dark:text-red-100 mb-1">
                    Download Failed
                  </h3>
                  <p className="text-sm text-red-800 dark:text-red-200">
                    {downloadStatus.error}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Success Message */}
          {downloadStatus.status === 'completed' && (
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl">
              <div className="flex items-start space-x-3">
                <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-1">
                    Download Complete!
                  </h3>
                  <p className="text-sm text-green-800 dark:text-green-200">
                    {downloadStatus.message}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Cancelled Message */}
          {downloadStatus.status === 'cancelled' && (
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl">
              <div className="flex items-start space-x-3">
                <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-yellow-900 dark:text-yellow-100 mb-1">
                    Download Cancelled
                  </h3>
                  <p className="text-sm text-yellow-800 dark:text-yellow-200">
                    {downloadStatus.message}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 bg-gray-50 dark:bg-slate-700/50 border-t border-gray-200 dark:border-slate-600">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <div className="flex items-center space-x-1">
                <Clock className="h-4 w-4" />
                <span>Estimated time: {missingModels.length * 2}-{missingModels.length * 5} minutes</span>
              </div>
            </div>
            <button
              onClick={downloadStatus.is_downloading ? cancelDownload : onClose}
              className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
            >
              {downloadStatus.is_downloading ? 'Cancel Download' : 'Close'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
