'use client';

import { useEffect, useState } from 'react';
import { Trash2, Cpu, RefreshCw, Download, Volume2, Play, Pause, CheckCircle, XCircle, AlertTriangle, Zap, Settings as SettingsIcon, CloudDownload } from 'lucide-react';
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
  
  // Download state
  const [downloadStatus, setDownloadStatus] = useState({
    is_downloading: false,
    progress: 0,
    current_model: "",
    status: "idle",
    message: "",
    error: null
  });
  
  // Delete state
  const [deleting, setDeleting] = useState(false);

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

  const fetchDownloadStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/download-status');
      const data = await response.json();
      setDownloadStatus(data);
    } catch (err) {
      console.error('Failed to fetch download status:', err);
    }
  };

  const startDownload = async () => {
    try {
      const response = await fetch('http://localhost:8000/download-models', {
        method: 'POST',
      });
      
      if (response.ok) {
        // Start polling for download status
        const pollInterval = setInterval(async () => {
          await fetchDownloadStatus();
          
          // Stop polling if download is complete or error
          if (downloadStatus.status === 'completed' || downloadStatus.status === 'error') {
            clearInterval(pollInterval);
            // Refresh models after download
            await fetchModels();
          }
        }, 1000);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to start download');
      }
    } catch (err: any) {
      setError(err.message);
    }
  };

  const deleteModels = async () => {
    if (!confirm('Are you sure you want to delete all downloaded models? This action cannot be undone and will free up significant disk space.')) {
      return;
    }
    
    setDeleting(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/delete-models', {
        method: 'POST',
      });
      
      if (response.ok) {
        const result = await response.json();
        // Refresh models after deletion
        await fetchModels();
        // Show success message
        setError(null);
        // You could add a success state here if needed
        console.log('Models deleted successfully:', result);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to delete models');
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setDeleting(false);
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
    fetchDownloadStatus();
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


      {/* Model Download Section */}
      <div className={`p-6 rounded-2xl border border-gray-200 dark:border-slate-700 shadow-xl bg-white dark:bg-gradient-to-br dark:from-slate-900/90 dark:to-slate-800/50 backdrop-blur-md`}>
        <div className="flex items-center space-x-3 mb-6">
          <div className="p-2 rounded-xl bg-accent-blue/10">
            <CloudDownload className="h-6 w-6 text-accent-blue" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-accent-blue">Download AI Models</h2>
            <p className={`text-sm ${colors.text.secondary}`}>Download the latest AI models for optimal performance</p>
          </div>
        </div>
        
        <div className="space-y-6">
          {/* Download Progress */}
          {downloadStatus.is_downloading && (
            <div className="p-4 rounded-xl bg-accent-blue/5 border border-accent-blue/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-accent-blue border-t-transparent rounded-full animate-spin"></div>
                  <span className="font-medium text-accent-blue">Downloading Models...</span>
                </div>
                <span className="text-sm font-mono text-accent-blue">{downloadStatus.progress}%</span>
              </div>
              
              <div className="w-full bg-gray-200 dark:bg-slate-700 rounded-full h-2 mb-3">
                <div 
                  className="bg-accent-blue h-2 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${downloadStatus.progress}%` }}
                ></div>
              </div>
              
              <div className="text-sm text-gray-600 dark:text-gray-300">
                <p className="font-medium">{downloadStatus.message}</p>
                {downloadStatus.current_model && (
                  <p className="text-xs mt-1">Current: {downloadStatus.current_model}</p>
                )}
              </div>
            </div>
          )}

          {/* Download Status Messages */}
          {downloadStatus.status === 'completed' && (
            <div className="p-4 rounded-xl bg-accent-green/5 border border-accent-green/20">
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-5 w-5 text-accent-green" />
                <span className="font-medium text-accent-green">Download Complete!</span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                All models have been downloaded successfully. You can now use them for generation.
              </p>
            </div>
          )}

          {downloadStatus.status === 'error' && (
            <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/20">
              <div className="flex items-center space-x-2">
                <XCircle className="h-5 w-5 text-red-500" />
                <span className="font-medium text-red-500">Download Failed</span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                {downloadStatus.error || 'An error occurred during download. Please try again.'}
              </p>
            </div>
          )}

          {/* Download/Delete Button */}
          <div className="flex items-center justify-between p-4 rounded-xl bg-gray-50 dark:bg-slate-800/50">
            <div>
              <h3 className="font-semibold text-lg">
                {models.video_models.length > 0 || models.audio_models.length > 0 
                  ? "Manage Downloaded Models" 
                  : "Download All Models"
                }
              </h3>
              <p className={`text-sm ${colors.text.secondary} mt-1`}>
                {models.video_models.length > 0 || models.audio_models.length > 0 
                  ? "Models are downloaded and ready to use. Delete them to free up space."
                  : "Download Stable Video Diffusion, Stable Diffusion, and Bark models (~13GB total)"
                }
              </p>
            </div>
            
            {/* Show delete button if models are downloaded */}
            {models.video_models.length > 0 || models.audio_models.length > 0 ? (
              <button
                onClick={deleteModels}
                disabled={deleting || downloadStatus.is_downloading}
                className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 hover:scale-105 ${
                  deleting || downloadStatus.is_downloading
                    ? 'bg-gray-300 dark:bg-slate-600 text-gray-500 cursor-not-allowed'
                    : 'bg-red-500/10 text-red-600 hover:bg-red-500/20 border border-red-500/30'
                }`}
              >
                {deleting ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                    <span>Deleting...</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <Trash2 className="h-4 w-4" />
                    <span>Delete Models</span>
                  </div>
                )}
              </button>
            ) : (
              /* Show download button if no models are downloaded */
              <button
                onClick={startDownload}
                disabled={downloadStatus.is_downloading}
                className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 hover:scale-105 ${
                  downloadStatus.is_downloading
                    ? 'bg-gray-300 dark:bg-slate-600 text-gray-500 cursor-not-allowed'
                    : 'bg-accent-blue/10 text-accent-blue hover:bg-accent-blue/20 border border-accent-blue/30'
                }`}
              >
                {downloadStatus.is_downloading ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                    <span>Downloading...</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2">
                    <CloudDownload className="h-4 w-4" />
                    <span>Download Models</span>
                  </div>
                )}
              </button>
            )}
          </div>
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


