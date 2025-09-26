'use client';

import { useState } from 'react';
import { Settings, Loader2, CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';

interface Model {
  id: string;
  name: string;
  description: string;
  max_duration: number;
  resolution?: string;
  sample_rate?: number;
  loaded: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
interface ModelManagerProps {
  // No props needed for this component
}

export function ModelManager({}: ModelManagerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [models, setModels] = useState<{
    video_models: Model[];
    audio_models: Model[];
  }>({ video_models: [], audio_models: [] });
  const [loading, setLoading] = useState(false);
  const colors = useThemeColors();

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/models');
      const data = await response.json();
      setModels(data);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleModel = async (modelId: string, modelType: 'video' | 'audio') => {
    try {
      const model = models[`${modelType}_models`].find(m => m.id === modelId);
      if (!model) return;

      const endpoint = model.loaded ? 'unload' : 'load';
      const response = await fetch(`http://localhost:8000/models/${modelType}/${modelId}/${endpoint}`, {
        method: 'POST',
      });

      if (response.ok) {
        // Refresh models list
        await fetchModels();
      } else {
        console.error(`Failed to ${endpoint} model:`, modelId);
      }
    } catch (err) {
      console.error(`Error toggling model ${modelId}:`, err);
    }
  };

  // const allModels = [...models.video_models, ...models.audio_models];

  return (
    <>
      <button
        onClick={() => {
          setIsOpen(true);
          fetchModels();
        }}
        className={`p-2 rounded-lg transition-all duration-200 bg-gray-50 dark:bg-slate-700/50 text-gray-700 dark:text-slate-200 hover:bg-gray-100 dark:hover:bg-slate-600/50 hover:scale-105 group relative`}
        title="Manage AI Models - Load/Unload models to optimize memory usage"
      >
        <Settings className="h-5 w-5" />
        <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 bg-gray-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap">
          Model Manager
        </div>
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div 
            className="fixed inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          />
          
          <div className={`relative w-full max-w-4xl max-h-[80vh] overflow-hidden rounded-2xl bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700 animate-slide-up`}>
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Model Manager
                </h2>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setIsOpen(false)}
                    className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
                  >
                    <XCircle className="h-5 w-5" />
                  </button>
                </div>
              </div>
            </div>

            <div className="p-6 overflow-y-auto max-h-[60vh]">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Video Models */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Video Models
                    </h3>
                    <div className="space-y-3">
                      {models.video_models.map((model) => (
                        <div
                          key={model.id}
                          className={`p-4 rounded-lg border transition-all duration-200 ${
                            model.loaded
                              ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
                              : 'border-gray-200 dark:border-gray-700'
                          }`}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <h4 className="font-medium text-gray-900 dark:text-white">
                                  {model.name}
                                </h4>
                                {model.loaded ? (
                                  <CheckCircle className="h-4 w-4 text-green-500" />
                                ) : (
                                  <XCircle className="h-4 w-4 text-gray-400" />
                                )}
                              </div>
                              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                                {model.description}
                              </p>
                              <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                                <span>Max: {model.max_duration}s</span>
                                {model.resolution && <span>{model.resolution}</span>}
                              </div>
                            </div>
                            <button
                              onClick={() => handleToggleModel(model.id, 'video')}
                              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                                model.loaded
                                  ? 'bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/20 dark:text-red-400 dark:hover:bg-red-900/30'
                                  : 'bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/20 dark:text-blue-400 dark:hover:bg-blue-900/30'
                              }`}
                            >
                              {model.loaded ? 'Unload' : 'Load'}
                            </button>
                            {!model.loaded && (
                              <button
                                onClick={() => {
                                  localStorage.setItem('default_video_model', model.id);
                                  fetchModels();
                                }}
                                className="ml-2 px-3 py-1 rounded-md text-sm border hover:bg-black/5 dark:hover:bg-white/5"
                              >
                                Set Default
                              </button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Audio Models */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Audio Models
                    </h3>
                    <div className="space-y-3">
                      {models.audio_models.map((model) => (
                        <div
                          key={model.id}
                          className={`p-4 rounded-lg border transition-all duration-200 ${
                            model.loaded
                              ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
                              : 'border-gray-200 dark:border-gray-700'
                          }`}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <h4 className="font-medium text-gray-900 dark:text-white">
                                  {model.name}
                                </h4>
                                {model.loaded ? (
                                  <CheckCircle className="h-4 w-4 text-green-500" />
                                ) : (
                                  <XCircle className="h-4 w-4 text-gray-400" />
                                )}
                              </div>
                              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                                {model.description}
                              </p>
                              <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                                <span>Max: {model.max_duration}s</span>
                                {model.sample_rate && <span>{model.sample_rate}Hz</span>}
                              </div>
                            </div>
                            <button
                              onClick={() => handleToggleModel(model.id, 'audio')}
                              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                                model.loaded
                                  ? 'bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/20 dark:text-red-400 dark:hover:bg-red-900/30'
                                  : 'bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/20 dark:text-blue-400 dark:hover:bg-blue-900/30'
                              }`}
                            >
                              {model.loaded ? 'Unload' : 'Load'}
                            </button>
                            {!model.loaded && (
                              <button
                                onClick={() => {
                                  localStorage.setItem('default_audio_model', model.id);
                                  fetchModels();
                                }}
                                className="ml-2 px-3 py-1 rounded-md text-sm border hover:bg.black/5 dark:hover:bg-white/5"
                              >
                                Set Default
                              </button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
