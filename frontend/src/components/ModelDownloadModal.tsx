'use client';

import { useMemo } from 'react';
import { X, AlertTriangle, HardDrive, Clock } from 'lucide-react';

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

// Downloads are disabled in this build. The modal is informational only.

export default function ModelDownloadModal({
  isOpen,
  onClose,
  missingModels,
  modelType,
  onModelsDownloaded
}: ModelDownloadModalProps) {
  // Memoized totals for display only
  const totalEstimatedGb = useMemo(() => {
    return missingModels.reduce((sum, m) => sum + (m.size_gb ?? getEstimatedSize(m.id)), 0);
  }, [missingModels]);

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

  // Download functions intentionally removed (downloads disabled)

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

  const cancelDownload = () => onClose();

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

          {/* Missing Models List (informational only) */}
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
                  
                  <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-300">
                    <span>Downloads are disabled in this build.</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Info note */}
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl">
            <div className="flex items-start space-x-3">
              <AlertTriangle className="h-5 w-5 text-blue-600 dark:text-blue-300 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-800 dark:text-blue-200">
                Model downloads are disabled in this build. Please place models manually in the configured models directory and click Close.
              </div>
            </div>
          </div>
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
