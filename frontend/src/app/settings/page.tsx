'use client';

import { useEffect, useState } from 'react';
import { Trash2, Cpu, RefreshCw, Volume2, Play, CheckCircle, XCircle, AlertTriangle, Settings as SettingsIcon, CloudDownload, HardDrive, Clock, FileText, Image as ImageIcon } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
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
  grouped_models?: string[];
}

export default function Page() {
  const colors = useThemeColors();
  const [, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<{
    outputs_path?: string;
    storage_usage_bytes?: number;
    gpu_info?: {
      type?: string;
      name?: string;
      memory_gb?: number;
    };
    ffmpeg_available?: boolean;
    system_info?: {
      platform?: string;
      platform_release?: string;
      architecture?: string;
      processor?: string;
    };
    memory_info?: {
      total_gb?: number;
      available_gb?: number;
      percent?: number;
    };
  } | null>(null);
  const [clearing, setClearing] = useState(false);
  const [models, setModels] = useState<{
    video_models: Model[];
    audio_models: Model[];
    image_models: Model[];
  }>({ video_models: [], audio_models: [], image_models: [] });
  const [loadingModels, setLoadingModels] = useState(false);
  
  // Download state
  const [downloadStatus, setDownloadStatus] = useState<{
    is_downloading: boolean;
    overall_progress: number;
    current_model: string;
    status: string;
    message: string;
    error: string | null;
    download_queue: string[];
    currently_downloading: string | null;
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
  }>({
    is_downloading: false,
    overall_progress: 0,
    current_model: "",
    status: "idle",
    message: "",
    error: null,
    download_queue: [],
    currently_downloading: null
  });
  
  const [clearingData, setClearingData] = useState(false);

  // Helper: determine if any model is actually downloaded (has size > 0)
  const hasAnyDownloaded = (
    (models.video_models || []).some(m => (m.size_gb || 0) > 0) ||
    (models.image_models || []).some(m => (m.size_gb || 0) > 0) ||
    (models.audio_models || []).some(m => (m.size_gb || 0) > 0)
  );

  // Normalize frontend model ids to backend ids for downloads
  const normalizeModelId = (modelId: string) => {
    if (modelId === 'stable-diffusion') return 'stable_diffusion';
    return modelId;
  };

  const fetchSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(getApiUrl('/settings'));
      if (!res.ok) throw new Error('Failed to fetch settings');
      const data = await res.json();
      setSettings(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to fetch settings');
    } finally {
      setLoading(false);
    }
  };

  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const response = await fetch(getApiUrl('/models'));
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
      const response = await fetch(getApiUrl('/download-status'));
      const data = await response.json();
      setDownloadStatus(data);
    } catch (err) {
      console.error('Failed to fetch download status:', err);
    }
  };


  const startDownload = async (force = false) => {
    // First, cleanup any existing download state
    try {
      await fetch(getApiUrl('/download-cleanup'), {
        method: 'POST',
      });
    } catch (cleanupError) {
      console.warn('Cleanup failed, continuing with download:', cleanupError);
    }
    
    // Clear any previous download status and show initial state
    setDownloadStatus({
      is_downloading: true,
      overall_progress: 0,
      current_model: '',
      status: 'downloading',
      message: force ? 'Force re-downloading all models...' : 'Starting download...',
      error: null,
      download_queue: [],
      currently_downloading: null,
      models: {}
    });
    
    try {
      // Use streaming endpoint for real-time progress
      const response = await fetch(getApiUrl(`/download-models-stream?force=${force}`));
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      if (!reader) {
        throw new Error('No response body reader available');
      }
      
      // Process streaming data
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
                // Update message with log content
                setDownloadStatus(prev => ({
                  ...prev,
                  message: data.message
                }));
                console.log('Download log:', data.message);
              } else if (data.type === 'success') {
                // Download completed successfully
                setDownloadStatus(prev => ({
                  ...prev,
                  status: 'completed',
                  message: data.message,
                  is_downloading: false,
                  download_queue: prev.download_queue || [],
                  currently_downloading: null,
                  models: prev.models || {}
                }));
              
              // Load models after download completion
              try {
                const loadResponse = await fetch(getApiUrl('/load-models'), {
                  method: 'POST',
                });
                if (loadResponse.ok) {
                  console.log('Models loaded successfully');
                  }
                } catch (loadError) {
                  console.error('Failed to load models:', loadError);
                }

              // Trigger voice preview generation so audio presets are available
              try {
                console.log('[Settings] Triggering voice preview generation...');
                const resp = await fetch(getApiUrl('/generate-voice-previews'), { method: 'POST' });
                console.log('[Settings] Voice preview generation requested. ok =', resp.ok);
              } catch (e) {
                console.warn('[Settings] Failed to request voice preview generation:', e);
              }
                
                // Refresh models list
                fetchModels();
                break;
              } else if (data.type === 'error') {
                // Download failed
                setDownloadStatus(prev => ({
                  ...prev,
                  status: 'error',
                  error: data.message,
                  is_downloading: false
                }));
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
      setDownloadStatus(prev => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        is_downloading: false
      }));
    }
  };

  // REMOVED: Individual model download function
  
  const processDownloadQueue = async () => {
    const currentState = downloadStatus;
    if (currentState.currently_downloading || (currentState.download_queue || []).length === 0) {
      return; // Already downloading or no queue
    }
    
    const nextModel = (currentState.download_queue || [])[0];
    if (!nextModel) return;
    
    // Update status to start downloading the next model
    setDownloadStatus(prev => ({
      ...prev,
      currently_downloading: nextModel,
      download_queue: (prev.download_queue || []).slice(1),
      models: {
        ...(prev.models || {}),
        [nextModel]: {
          ...(prev.models?.[nextModel] || {}),
          name: nextModel,
          status: 'downloading',
          progress: 0,
          downloaded_mb: 0,
          repo_id: prev.models?.[nextModel]?.repo_id,
          local_dir: prev.models?.[nextModel]?.local_dir,
          size_gb: prev.models?.[nextModel]?.size_gb,
          eta_seconds: prev.models?.[nextModel]?.eta_seconds,
          files_verified: prev.models?.[nextModel]?.files_verified
        }
      } as typeof prev.models
    }));
    
    try {
      // Use streaming endpoint for real-time progress
      const response = await fetch(getApiUrl(`/download-model-stream/${nextModel}?force=false`), {
        headers: {
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      if (!reader) {
        throw new Error('No response body reader available');
      }
      
      // Process streaming data
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'progress') {
                // Update per-model progress
                setDownloadStatus(prev => ({
                  ...prev,
                  is_downloading: true,
                  message: data.message || prev.message,
                  models: {
                    ...(prev.models || {}),
                    [nextModel]: {
                      name: nextModel,
                      repo_id: prev.models?.[nextModel]?.repo_id,
                      local_dir: prev.models?.[nextModel]?.local_dir,
                      size_gb: prev.models?.[nextModel]?.size_gb,
                      status: 'downloading',
                      progress: data.progress ?? 0,
                      downloaded_mb: data.downloaded_mb ?? 0,
                      eta_seconds: prev.models?.[nextModel]?.eta_seconds,
                      files_verified: prev.models?.[nextModel]?.files_verified
                    }
                  } as typeof prev.models
                }));
              } else if (data.type === 'log') {
                // Update message with log content
                setDownloadStatus(prev => ({
                  ...prev,
                  message: data.message
                }));
                console.log('Download log:', data.message);
              } else if (data.type === 'success') {
                // Download completed successfully (per-model)
                setDownloadStatus(prev => {
                  const updatedModels = {
                    ...(prev.models || {}),
                    [nextModel]: {
                      name: nextModel,
                      repo_id: prev.models?.[nextModel]?.repo_id,
                      local_dir: prev.models?.[nextModel]?.local_dir,
                      size_gb: prev.models?.[nextModel]?.size_gb,
                      status: 'done',
                      progress: 100,
                      downloaded_mb: prev.models?.[nextModel]?.downloaded_mb ?? 0,
                      eta_seconds: prev.models?.[nextModel]?.eta_seconds,
                      files_verified: true
                    }
                  } as typeof prev.models;
                  const anyActive = Object.values(updatedModels || {}).some((m: any) => m && m.status === 'downloading') || (prev.download_queue || []).length > 0;
                  return {
                    ...prev,
                    status: anyActive ? 'downloading' : 'completed',
                    message: data.message,
                    is_downloading: anyActive,
                    currently_downloading: null,
                    models: updatedModels as typeof prev.models
                  };
                });
                
                // Refresh models list
                fetchModels();
                
                // Process next in queue
                setTimeout(() => processDownloadQueue(), 1000);
                break;
              } else if (data.type === 'error') {
                // Download failed (per-model)
                setDownloadStatus(prev => {
                  const updatedModels = {
                    ...(prev.models || {}),
                    [nextModel]: {
                      name: nextModel,
                      repo_id: prev.models?.[nextModel]?.repo_id,
                      local_dir: prev.models?.[nextModel]?.local_dir,
                      size_gb: prev.models?.[nextModel]?.size_gb,
                      status: 'error',
                      progress: prev.models?.[nextModel]?.progress ?? 0,
                      downloaded_mb: prev.models?.[nextModel]?.downloaded_mb ?? 0,
                      eta_seconds: prev.models?.[nextModel]?.eta_seconds,
                      files_verified: false
                    }
                  } as typeof prev.models;
                  const anyActive = Object.values(updatedModels || {}).some((m: any) => m && m.status === 'downloading') || (prev.download_queue || []).length > 0;
                  return {
                    ...prev,
                    status: 'error',
                    error: data.message,
                    is_downloading: anyActive,
                    currently_downloading: null,
                    models: updatedModels as typeof prev.models
                  };
                });
                
                // Process next in queue even on error
                setTimeout(() => processDownloadQueue(), 1000);
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
      setDownloadStatus(prev => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        is_downloading: Object.values(prev.models || {}).some(m => (m as any).status === 'downloading') || (prev.download_queue || []).length > 0,
        currently_downloading: null
      }));
      
      // Process next in queue even on error
      setTimeout(() => processDownloadQueue(), 1000);
    }
  };

  const deleteModel = async (modelName: string, modelType: 'video' | 'audio' | 'image') => {
    // Find the model to check if it has grouped models
    const allModels = [...models.video_models, ...models.audio_models, ...models.image_models];
    const model = allModels.find(m => m.id === modelName);
    
    let confirmMessage = `Are you sure you want to delete ${model?.name || modelName}?\n\n`;
    confirmMessage += `This will free up storage space but you'll need to download it again to use it.\n\n`;
    confirmMessage += `This action cannot be undone.`;
    
    const confirmed = window.confirm(confirmMessage);
    
    if (!confirmed) {
      return;
    }
    
    try {
      // Best-effort unload before deleting to avoid file locking (especially for SVD)
      if (modelType === 'video' && modelName === 'svd') {
        try {
          await fetch(getApiUrl(`/models/video/${modelName}/unload`), { method: 'POST' });
        } catch (e) {
          // Ignore unload errors and continue
        }
      }

      const response = await fetch(getApiUrl(`/delete-model/${modelName}`), {
        method: 'DELETE',
      });
      
      const result = await response.json();
      
      if (response.ok && result.deleted) {
        alert(`✅ ${result.message}\n\nFreed up ${result.size_gb || result.size_freed_gb || 0} GB of storage space.`);
        // Refresh models list
        fetchModels();
      } else {
        alert(`❌ Failed to delete ${modelName}: ${result.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error deleting model:', error);
      alert(`❌ Error deleting ${modelName}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const deleteModels = async () => {
    // First, cleanup any existing download state
    try {
      await fetch(getApiUrl('/download-cleanup'), {
        method: 'POST',
      });
    } catch (cleanupError) {
      console.warn('Cleanup failed, continuing with delete:', cleanupError);
    }
    
    try {
      const response = await fetch(getApiUrl('/delete-models'), {
        method: 'POST',
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Reset models state to force refresh
        setModels({
          video_models: [],
          image_models: [],
          audio_models: []
        });
        
        // Wait a moment for backend to process deletion
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Refresh models after deletion
        await fetchModels();
        setError(null);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to delete models');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Network error occurred');
    }
  };


  useEffect(() => {
    fetchSettings();
    fetchModels();
    fetchDownloadStatus();
    
    // Cleanup function to reset download status when component unmounts
    return () => {
      // Only cleanup if download is in progress
      if (downloadStatus.is_downloading) {
        fetch(getApiUrl('/download-cleanup'), {
          method: 'POST',
        }).catch(err => console.warn('Cleanup on unmount failed:', err));
      }
    };
  }, []);

  const clearOutputs = async () => {
    setClearingData(true);
    try {
      const response = await fetch(getApiUrl('/outputs/clear'), { method: 'POST' });
      
      if (response.ok) {
        const result = await response.json();
        await fetchSettings();
        alert(`Successfully cleared ${result.deleted_files} files and freed ${result.freed_space_mb} MB of space.`);
      } else {
        const errorData = await response.json();
        alert(`Failed to clear data: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (err) {
      alert('Network error occurred while clearing data');
    } finally {
      setClearingData(false);
    }
  };


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
          {/* Model Details - Always show lists; per-model progress renders inline */}
          {(models.video_models.length > 0 || models.audio_models.length > 0 || (models.image_models && models.image_models.length > 0)) && (
            <div className="p-4 rounded-xl bg-accent-green/5 border border-accent-green/20">
              <div className="flex items-center space-x-2 mb-4">
                <CheckCircle className="h-5 w-5 text-accent-green" />
                <span className="font-medium text-accent-green">Available Models</span>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Video Models */}
                {models.video_models.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2 p-3 rounded-lg bg-accent-blue/5 border border-accent-blue/20">
                      <Play className="h-5 w-5 text-accent-blue" />
                      <h4 className="font-semibold text-sm text-gray-900 dark:text-slate-100">Video Models</h4>
                    </div>
                    {models.video_models.map((model, index) => {
                      const normId = normalizeModelId(model.id);
                      const per = downloadStatus.models?.[normId];
                      const isThisDownloading = per?.status === 'downloading';
                      const isThisQueued = per?.status === 'queued';
                      const thisProgress = per?.progress ?? 0;
                      const thisDownloaded = per?.downloaded_mb ?? 0;
                      return (
                      <div key={model.id || `video-${index}`} className="p-4 rounded-xl bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-600 shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <span className="font-semibold text-sm text-gray-900 dark:text-slate-100">{model.name}</span>
                            {model.description && (
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{model.description}</p>
                            )}
                          </div>
                          <div className="flex items-center space-x-2">
                            <div className={`px-2 py-1 rounded-full text-xs ${
                              model.size_gb && model.size_gb > 0
                                ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                                : 'bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400'
                            }`}>
                              {model.size_gb && model.size_gb > 0 ? 'Available' : 'Download'}
                            </div>
                            {model.size_gb && model.size_gb > 0 ? (
                              <button
                                onClick={() => deleteModel(model.id, 'video')}
                                className="p-1 rounded-full hover:bg-red-100 dark:hover:bg-red-900/20 transition-colors"
                                title={`Delete ${model.name}`}
                              >
                                <Trash2 className="h-3 w-3 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300" />
                              </button>
                            ) : (
                              <>
                              {isThisDownloading ? (
                                <button
                                  onClick={async () => {
                                    await fetch(getApiUrl(`/cancel-download/${normId}`), { method: 'POST' });
                                    fetchModels();
                                    setDownloadStatus(prev => ({ ...prev, is_downloading: false, current_model: '', message: '', overall_progress: 0 }));
                                  }}
                                  className="px-2 py-1 rounded text-xs bg-red-500/10 text-red-600 border border-red-500/30"
                                >
                                  Cancel
                                </button>
                              ) : null}
                              </>
                            )}
                          </div>
                        </div>
                        
                        {/* Download Progress */}
                        {isThisDownloading && (
                          <div className="mb-2">
                            <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                              <span>Downloading...</span>
                              <span>
                                {thisDownloaded} MB • {thisProgress}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${thisProgress}%` }}
                              ></div>
                            </div>
                            <p className="text-xs text-gray-500 mt-1 truncate">{downloadStatus.message}</p>
                          </div>
                        )}
                        
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          {model.size_gb && model.size_gb > 0 && (
                            <div className="flex items-center space-x-1">
                              <HardDrive className="h-3 w-3" />
                              <span>{model.size_gb}GB</span>
                            </div>
                          )}
                        </div>
                      </div>
                    );})}
                  </div>
                )}

                {/* Image Models */}
                {models.image_models && models.image_models.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2 p-3 rounded-lg bg-accent-green/5 border border-accent-green/20">
                      <ImageIcon className="h-5 w-5 text-accent-green" />
                      <h4 className="font-semibold text-sm text-gray-900 dark:text-slate-100">Image Models</h4>
                    </div>
                    {models.image_models && models.image_models.map((model, index) => {
                      const normId = normalizeModelId(model.id);
                      const per = downloadStatus.models?.[normId];
                      const isThisDownloading = per?.status === 'downloading';
                      const isThisQueued = per?.status === 'queued';
                      const thisProgress = per?.progress ?? 0;
                      const thisDownloaded = per?.downloaded_mb ?? 0;
                      return (
                      <div key={model.id || `image-${index}`} className="p-4 rounded-xl bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-600 shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <span className="font-semibold text-sm text-gray-900 dark:text-slate-100">{model.name}</span>
                            {model.description && (
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{model.description}</p>
                            )}
                          </div>
                          <div className="flex items-center space-x-2">
                            <div className={`px-2 py-1 rounded-full text-xs ${
                              model.size_gb && model.size_gb > 0
                                ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                                : 'bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400'
                            }`}>
                            {model.size_gb && model.size_gb > 0 ? 'Available' : 'Download'}
                            </div>
                            {model.size_gb && model.size_gb > 0 ? (
                              <button
                                onClick={() => deleteModel(model.id, 'image')}
                                className="p-1 rounded-full hover:bg-red-100 dark:hover:bg-red-900/20 transition-colors"
                                title={`Delete ${model.name}`}
                              >
                                <Trash2 className="h-3 w-3 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300" />
                              </button>
                            ) : (
                              <>
                              {isThisDownloading ? (
                                <button
                                  onClick={async () => {
                                    await fetch(getApiUrl(`/cancel-download/${normId}`), { method: 'POST' });
                                    fetchModels();
                                    setDownloadStatus(prev => ({ ...prev, is_downloading: false, current_model: '', message: '', overall_progress: 0 }));
                                  }}
                                  className="px-2 py-1 rounded text-xs bg-red-500/10 text-red-600 border border-red-500/30"
                                >
                                  Cancel
                                </button>
                              ) : null}
                              </>
                            )}
                          </div>
                        </div>
                        
                        {/* Download Progress */}
                        {isThisDownloading && (
                          <div className="mb-2">
                            <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                              <span>Downloading...</span>
                              <span>
                                {thisDownloaded} MB • {thisProgress}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${thisProgress}%` }}
                              ></div>
                            </div>
                            <p className="text-xs text-gray-500 mt-1 truncate">{downloadStatus.message}</p>
                          </div>
                        )}
                        
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          {model.size_gb && model.size_gb > 0 && (
                            <div className="flex items-center space-x-1">
                              <HardDrive className="h-3 w-3" />
                              <span>{model.size_gb}GB</span>
                            </div>
                          )}
                        </div>
                      </div>
                    );})}
                  </div>
                )}

                {/* Audio Models */}
                {models.audio_models.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2 p-3 rounded-lg bg-accent-violet/5 border border-accent-violet/20">
                      <Volume2 className="h-5 w-5 text-accent-violet" />
                      <h4 className="font-semibold text-sm text-gray-900 dark:text-slate-100">Audio Models</h4>
                    </div>
                    {models.audio_models.map((model, index) => {
                      const normId = normalizeModelId(model.id);
                      const per = downloadStatus.models?.[normId];
                      const isThisDownloading = per?.status === 'downloading';
                      const isThisQueued = per?.status === 'queued';
                      const thisProgress = per?.progress ?? 0;
                      return (
                      <div key={model.id || `audio-${index}`} className="p-4 rounded-xl bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-600 shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <span className="font-semibold text-sm text-gray-900 dark:text-slate-100">{model.name}</span>
                            {model.description && (
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{model.description}</p>
                            )}
                          </div>
                          <div className="flex items-center space-x-2">
                            <div className={`px-2 py-1 rounded-full text-xs ${
                              model.size_gb && model.size_gb > 0
                                ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                                : 'bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400'
                            }`}>
                              {model.size_gb && model.size_gb > 0 ? 'Available' : 'Download'}
                            </div>
                            {model.size_gb && model.size_gb > 0 ? (
                              <button
                                onClick={() => deleteModel(model.id, 'audio')}
                                className="p-1 rounded-full hover:bg-red-100 dark:hover:bg-red-900/20 transition-colors"
                                title={`Delete ${model.name}`}
                              >
                                <Trash2 className="h-3 w-3 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300" />
                              </button>
                            ) : null}
                          </div>
                        </div>
                        
                        {/* Download Progress */}
                        {isThisDownloading && (
                          <div className="mb-2">
                            <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                              <span>Downloading...</span>
                              <span>{thisProgress}%</span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${thisProgress}%` }}
                              ></div>
                            </div>
                            <p className="text-xs text-gray-500 mt-1 truncate">{downloadStatus.message}</p>
                          </div>
                        )}
                        
                        {/* Queue Status */}
                        {isThisQueued && (
                          <div className="mb-2">
                            <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
                              <span>Queued for download...</span>
                              <button
                                onClick={() => {
                                  setDownloadStatus(prev => ({
                                    ...prev,
                                    download_queue: (prev.download_queue || []).filter(id => id !== normId),
                                    models: {
                                      ...(prev.models || {}),
                                      [normId]: {
                                        name: normId,
                                        repo_id: prev.models?.[normId]?.repo_id,
                                        local_dir: prev.models?.[normId]?.local_dir,
                                        size_gb: prev.models?.[normId]?.size_gb,
                                        status: 'pending',
                                        progress: prev.models?.[normId]?.progress ?? 0,
                                        downloaded_mb: prev.models?.[normId]?.downloaded_mb ?? 0,
                                        eta_seconds: prev.models?.[normId]?.eta_seconds,
                                        files_verified: prev.models?.[normId]?.files_verified
                                      }
                                    } as typeof prev.models
                                  }));
                                }}
                                className="text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
                                title="Remove from queue"
                              >
                                Remove
                              </button>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <div className="bg-yellow-500 h-2 rounded-full w-full animate-pulse"></div>
                            </div>
                          </div>
                        )}
                        
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          {model.size_gb && model.size_gb > 0 && (
                            <div className="flex items-center space-x-1">
                              <HardDrive className="h-3 w-3" />
                              <span>{model.size_gb}GB</span>
                            </div>
                          )}
                        </div>
                      </div>
                    );})}
                  </div>
                )}
              </div>
              
              <div className="mt-4 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                <div className="flex items-center space-x-2">
                  <HardDrive className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium text-blue-800 dark:text-blue-200">Storage Information</span>
                </div>
                <p className="text-xs text-blue-600 dark:text-blue-300 mt-1">
                  Download models to use them for generation. Downloaded models are stored locally and can be deleted to free up disk space.
                </p>
              </div>
            </div>
          )}

          {/* Removed global multi-model download UI to show only per-model progress */}

          {/* Download Status Messages */}
          {downloadStatus.status === 'completed' && !downloadStatus.is_downloading && (
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

          {downloadStatus.status === 'error' && downloadStatus.is_downloading && (
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
                {models.video_models.length > 0 || models.audio_models.length > 0 || (models.image_models && models.image_models.length > 0) 
                  ? "Manage Downloaded Models" 
                  : "Download All Models"
                }
              </h3>
              <p className={`text-sm ${colors.text.secondary} mt-1`}>
                {models.video_models.length > 0 || models.audio_models.length > 0 || (models.image_models && models.image_models.length > 0) 
                  ? "Models are downloaded and ready to use. Delete them to free up space."
                  : "Download Stable Video Diffusion, Stable Diffusion, and Bark models (~54GB total) with resume capability"
                }
              </p>
            </div>
            
            {/* Always show download button */}
            <div className="flex flex-col space-y-2">
              <button
                onClick={() => startDownload(false)}
                className="px-6 py-3 rounded-xl font-semibold transition-all duration-200 hover:scale-105 bg-accent-blue/10 text-accent-blue hover:bg-accent-blue/20 border border-accent-blue/30"
              >
                <div className="flex items-center space-x-2">
                  <CloudDownload className="h-4 w-4" />
                  <span>Download Models</span>
                </div>
              </button>
            </div>
              
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
                <span className="text-sm font-medium">GPU Type</span>
                <span className="text-sm capitalize">{settings?.gpu_info?.type || 'CPU'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Device</span>
                <span className="text-sm">{settings?.gpu_info?.name || 'CPU'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Memory</span>
                <span className="text-sm">{settings?.gpu_info?.memory_gb ? `${settings.gpu_info.memory_gb.toFixed(1)} GB` : '—'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">System RAM</span>
                <span className="text-sm">{settings?.memory_info?.total_gb ? `${settings.memory_info.total_gb} GB` : '—'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Platform</span>
                <span className="text-sm">{settings?.system_info?.platform || '—'}</span>
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
                  {settings?.storage_usage_bytes ? (settings.storage_usage_bytes / (1024*1024)).toFixed(2) : '—'} MB
                </span>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="space-y-4">
            <h3 className="font-semibold text-lg flex items-center space-x-2">
              <SettingsIcon className="h-5 w-5 text-accent-green" />
              <span>Actions</span>
            </h3>
            <div className="space-y-3">
              <button 
                disabled={clearingData} 
                onClick={clearOutputs} 
                className="w-full inline-flex items-center justify-center space-x-2 px-4 py-3 rounded-xl text-red-600 hover:bg-red-500/10 border border-red-500/30 disabled:opacity-50 transition-all duration-200 hover:scale-105"
              >
                <Trash2 className="h-4 w-4" />
                <span>{clearingData ? 'Clearing...' : 'Clear Data'}</span>
              </button>
              
            </div>
          </div>
        </div>
      </div>


    </div>
  );
}


