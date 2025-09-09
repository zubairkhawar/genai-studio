'use client';

import { useEffect, useState } from 'react';
import { Trash2, Cpu, RefreshCw, Download, Volume2, Play, Pause, CheckCircle, XCircle, AlertTriangle, Zap, Settings as SettingsIcon } from 'lucide-react';
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

export default function Page() {
  const colors = useThemeColors();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<any>(null);
  const [clearing, setClearing] = useState(false);
  const [models, setModels] = useState<{
    video_models: Model[];
    audio_models: Model[];
  }>({ video_models: [], audio_models: [] });
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelLoadingStates, setModelLoadingStates] = useState<Record<string, boolean>>({});

  const fetchSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/settings');
      if (!res.ok) throw new Error('Failed to fetch settings');
      const data = await res.json();
      setSettings(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const response = await fetch('http://localhost:8000/models');
      const data = await response.json();
      setModels(data);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    } finally {
      setLoadingModels(false);
    }
  };

  const handleToggleModel = async (modelId: string, modelType: 'video' | 'audio') => {
    setModelLoadingStates(prev => ({ ...prev, [modelId]: true }));
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
    } finally {
      setModelLoadingStates(prev => ({ ...prev, [modelId]: false }));
    }
  };

  useEffect(() => {
    fetchSettings();
    fetchModels();
  }, []);

  const clearOutputs = async () => {
    setClearing(true);
    try {
      await fetch('http://localhost:8000/outputs/clear', { method: 'POST' });
      await fetchSettings();
    } finally {
      setClearing(false);
    }
  };

  // Get the primary models (SVD for video, Bark for audio)
  const primaryVideoModel = models.video_models.find(m => m.id === 'stable-video-diffusion') || models.video_models[0];
  const primaryAudioModel = models.audio_models.find(m => m.id === 'bark') || models.audio_models[0];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent">
            Settings
          </h1>
          <p className={`text-sm ${colors.text.secondary} mt-1`}>
            Manage AI models and system configuration
          </p>
        </div>
        <button 
          onClick={() => { fetchSettings(); fetchModels(); }} 
          disabled={loadingModels}
          className="flex items-center space-x-2 px-4 py-2 rounded-xl border border-gray-200 dark:border-slate-600 text-accent-blue hover:bg-accent-blue/10 transition-all duration-200 hover:scale-105"
        >
          <RefreshCw className={`h-4 w-4 ${loadingModels ? 'animate-spin' : ''}`} />
          <span className="text-sm font-medium">Refresh</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-600">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5" />
            <span className="font-medium">{error}</span>
          </div>
        </div>
      )}

      {/* Primary Models Section */}
      <div className={`p-6 rounded-2xl border border-gray-200 dark:border-slate-700 shadow-xl bg-white dark:bg-gradient-to-br dark:from-slate-900/90 dark:to-slate-800/50 backdrop-blur-md`}>
        <div className="flex items-center space-x-3 mb-6">
          <div className="p-2 rounded-xl bg-accent-green/10">
            <Zap className="h-6 w-6 text-accent-green" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-accent-green">Primary AI Models</h2>
            <p className={`text-sm ${colors.text.secondary}`}>Load the main models for video and audio generation</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Video Model */}
          {primaryVideoModel && (
            <div className={`p-6 rounded-xl border-2 transition-all duration-300 ${
              primaryVideoModel.loaded
                ? 'border-accent-green/50 bg-accent-green/5'
                : 'border-gray-200 dark:border-slate-600 hover:border-accent-blue/50'
            }`}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="p-2 rounded-lg bg-accent-blue/10">
                      <Play className="h-5 w-5 text-accent-blue" />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg text-accent-blue">Stable Video Diffusion</h3>
                      <p className="text-sm text-gray-500">Primary video generation model</p>
                    </div>
                    {primaryVideoModel.loaded ? (
                      <CheckCircle className="h-6 w-6 text-accent-green" />
                    ) : (
                      <XCircle className="h-6 w-6 text-gray-400" />
                    )}
                  </div>
                  <p className={`text-sm ${colors.text.secondary} mb-4`}>
                    High-quality text-to-video generation with excellent motion and detail
                  </p>
                  <div className="flex items-center gap-2 text-xs mb-4">
                    <span className={`px-3 py-1 rounded-full ${
                      primaryVideoModel.loaded 
                        ? 'bg-accent-green/10 text-accent-green' 
                        : 'bg-yellow-500/10 text-yellow-600 dark:text-yellow-400'
                    }`}>
                      {primaryVideoModel.loaded ? 'Ready to use' : 'Not loaded'}
                    </span>
                    <span className="px-3 py-1 rounded-full bg-accent-blue/10 text-accent-blue">
                      {primaryVideoModel.resolution || '1024x576'}
                    </span>
                    <span className="px-3 py-1 rounded-full bg-accent-violet/10 text-accent-violet">
                      Max: {primaryVideoModel.max_duration}s
                    </span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => handleToggleModel(primaryVideoModel.id, 'video')}
                disabled={modelLoadingStates[primaryVideoModel.id]}
                className={`w-full px-4 py-3 rounded-xl font-semibold transition-all duration-200 hover:scale-105 ${
                  primaryVideoModel.loaded
                    ? 'bg-red-500/10 text-red-600 hover:bg-red-500/20 border border-red-500/30'
                    : 'bg-accent-blue/10 text-accent-blue hover:bg-accent-blue/20 border border-accent-blue/30'
                } ${modelLoadingStates[primaryVideoModel.id] ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {modelLoadingStates[primaryVideoModel.id] ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                    <span>Loading...</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center space-x-2">
                    {primaryVideoModel.loaded ? (
                      <>
                        <XCircle className="h-4 w-4" />
                        <span>Unload Model</span>
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4" />
                        <span>Load Model</span>
                      </>
                    )}
                  </div>
                )}
              </button>
            </div>
          )}

          {/* Audio Model */}
          {primaryAudioModel && (
            <div className={`p-6 rounded-xl border-2 transition-all duration-300 ${
              primaryAudioModel.loaded
                ? 'border-accent-green/50 bg-accent-green/5'
                : 'border-gray-200 dark:border-slate-600 hover:border-accent-violet/50'
            }`}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="p-2 rounded-lg bg-accent-violet/10">
                      <Volume2 className="h-5 w-5 text-accent-violet" />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg text-accent-violet">Bark</h3>
                      <p className="text-sm text-gray-500">Primary audio generation model</p>
                    </div>
                    {primaryAudioModel.loaded ? (
                      <CheckCircle className="h-6 w-6 text-accent-green" />
                    ) : (
                      <XCircle className="h-6 w-6 text-gray-400" />
                    )}
                  </div>
                  <p className={`text-sm ${colors.text.secondary} mb-4`}>
                    High-quality text-to-speech with natural voice synthesis
                  </p>
                  <div className="flex items-center gap-2 text-xs mb-4">
                    <span className={`px-3 py-1 rounded-full ${
                      primaryAudioModel.loaded 
                        ? 'bg-accent-green/10 text-accent-green' 
                        : 'bg-yellow-500/10 text-yellow-600 dark:text-yellow-400'
                    }`}>
                      {primaryAudioModel.loaded ? 'Ready to use' : 'Not loaded'}
                    </span>
                    <span className="px-3 py-1 rounded-full bg-accent-violet/10 text-accent-violet">
                      {primaryAudioModel.sample_rate || 22050}Hz
                    </span>
                    <span className="px-3 py-1 rounded-full bg-accent-blue/10 text-accent-blue">
                      Max: {primaryAudioModel.max_duration}s
                    </span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => handleToggleModel(primaryAudioModel.id, 'audio')}
                disabled={modelLoadingStates[primaryAudioModel.id]}
                className={`w-full px-4 py-3 rounded-xl font-semibold transition-all duration-200 hover:scale-105 ${
                  primaryAudioModel.loaded
                    ? 'bg-red-500/10 text-red-600 hover:bg-red-500/20 border border-red-500/30'
                    : 'bg-accent-violet/10 text-accent-violet hover:bg-accent-violet/20 border border-accent-violet/30'
                } ${modelLoadingStates[primaryAudioModel.id] ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {modelLoadingStates[primaryAudioModel.id] ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                    <span>Loading...</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center space-x-2">
                    {primaryAudioModel.loaded ? (
                      <>
                        <XCircle className="h-4 w-4" />
                        <span>Unload Model</span>
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4" />
                        <span>Load Model</span>
                      </>
                    )}
                  </div>
                )}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* System Information */}
      <div className={`p-6 rounded-2xl border border-gray-200 dark:border-slate-700 shadow-xl bg-white dark:bg-gradient-to-br dark:from-slate-900/90 dark:to-slate-800/50 backdrop-blur-md`}>
        <div className="flex items-center space-x-3 mb-6">
          <div className="p-2 rounded-xl bg-accent-green/10">
            <SettingsIcon className="h-6 w-6 text-accent-green" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-accent-green">System Information</h2>
            <p className={`text-sm ${colors.text.secondary}`}>Hardware and storage details</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Hardware Info */}
          <div className="space-y-4">
            <h3 className="font-semibold text-lg flex items-center space-x-2">
              <Cpu className="h-5 w-5 text-accent-blue" />
              <span>Hardware</span>
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">GPU</span>
                <span className="text-sm">{settings?.gpu_info?.type || '—'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Device</span>
                <span className="text-sm">{settings?.gpu_info?.name || '—'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Memory</span>
                <span className="text-sm">{settings?.gpu_info?.memory_gb ? `${settings.gpu_info.memory_gb} GB` : '—'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">FFmpeg</span>
                <span className={`text-sm ${settings?.ffmpeg_available ? 'text-green-600' : 'text-red-600'}`}>
                  {settings?.ffmpeg_available ? 'Available' : 'Not detected'}
                </span>
              </div>
            </div>
          </div>
          
          {/* Storage Info */}
          <div className="space-y-4">
            <h3 className="font-semibold text-lg flex items-center space-x-2">
              <Trash2 className="h-5 w-5 text-accent-violet" />
              <span>Storage</span>
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Outputs path</span>
                <code className="text-xs px-2 py-1 rounded bg-black/5 dark:bg-white/5">
                  {settings?.outputs_path || '—'}
                </code>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Usage</span>
                <span className="text-sm">
                  {settings ? (settings.storage_usage_bytes / (1024*1024)).toFixed(2) : '—'} MB
                </span>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="space-y-4">
            <h3 className="font-semibold text-lg flex items-center space-x-2">
              <Zap className="h-5 w-5 text-accent-green" />
              <span>Actions</span>
            </h3>
            <div className="space-y-3">
              <button 
                disabled={clearing} 
                onClick={clearOutputs} 
                className="w-full inline-flex items-center justify-center space-x-2 px-4 py-3 rounded-xl text-red-600 hover:bg-red-500/10 border border-red-500/30 disabled:opacity-50 transition-all duration-200 hover:scale-105"
              >
                <Trash2 className="h-4 w-4" />
                <span>Clear All Outputs</span>
              </button>
              <button 
                onClick={() => { fetchSettings(); fetchModels(); }} 
                disabled={loadingModels}
                className="w-full inline-flex items-center justify-center space-x-2 px-4 py-3 rounded-xl text-accent-blue hover:bg-accent-blue/10 border border-accent-blue/30 disabled:opacity-50 transition-all duration-200 hover:scale-105"
              >
                <RefreshCw className={`h-4 w-4 ${loadingModels ? 'animate-spin' : ''}`} />
                <span>Refresh All</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


