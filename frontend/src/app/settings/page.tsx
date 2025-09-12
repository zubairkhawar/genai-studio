'use client';

import { useEffect, useState } from 'react';
import { Trash2, Cpu, RefreshCw, Volume2, Play, CheckCircle, XCircle, AlertTriangle, Zap, Settings as SettingsIcon, CloudDownload, HardDrive, Clock, FileText } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
import { ProgressModal } from '@/components/ProgressModal';

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
  } | null>(null);
  const [clearing, setClearing] = useState(false);
  const [models, setModels] = useState<{
    video_models: Model[];
    audio_models: Model[];
  }>({ video_models: [], audio_models: [] });
  const [loadingModels, setLoadingModels] = useState(false);
  
  // Download state
  const [downloadStatus, setDownloadStatus] = useState({
    is_downloading: false,
    progress: 0,
    current_model: "",
    status: "idle",
    message: "",
    error: null
  });
  
  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalType, setModalType] = useState<'download' | 'delete'>('download');
  const [modalProgress, setModalProgress] = useState(0);
  const [modalStatus, setModalStatus] = useState<'idle' | 'in_progress' | 'completed' | 'error'>('idle');
  const [modalCurrentStep, setModalCurrentStep] = useState('');
  const [modalDetails, setModalDetails] = useState<string[]>([]);
  const [modalError, setModalError] = useState<string | null>(null);
  const [modalModelProgress, setModalModelProgress] = useState<Array<{
    name: string;
    size: string;
    progress: number;
    status: 'pending' | 'downloading' | 'completed' | 'error';
    speed?: string;
    eta?: string;
  }>>([]);

  const fetchSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/settings');
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
    setModalType('download');
    setModalOpen(true);
    setModalStatus('in_progress');
    setModalProgress(0);
    setModalCurrentStep('Starting download...');
    setModalDetails(['Preparing to download AI models', 'This may take several minutes']);
    setModalError(null);
    
    // Initialize model progress tracking
    setModalModelProgress([
      {
        name: 'Stable Video Diffusion',
        size: '~4GB',
        progress: 0,
        status: 'pending'
      },
      {
        name: 'Stable Diffusion',
        size: '~4GB', 
        progress: 0,
        status: 'pending'
      },
      {
        name: 'Bark',
        size: '~5GB',
        progress: 0,
        status: 'pending'
      }
    ]);
    
    try {
      const response = await fetch('http://localhost:8000/download-models', {
        method: 'POST',
      });
      
      if (response.ok) {
        // Start polling for download status
        const pollInterval = setInterval(async () => {
          await fetchDownloadStatus();
          
          // Update modal with current progress
          setModalProgress(downloadStatus.progress);
          setModalCurrentStep(downloadStatus.message || 'Downloading...');
          
          // Update individual model progress
          if (downloadStatus.current_model) {
            setModalModelProgress(prev => prev.map(model => {
              if (model.name.toLowerCase().includes(downloadStatus.current_model.toLowerCase()) || 
                  downloadStatus.current_model.toLowerCase().includes(model.name.toLowerCase().split(' ')[0])) {
                return {
                  ...model,
                  status: 'downloading' as const,
                  progress: Math.min(downloadStatus.progress, 100),
                  speed: downloadStatus.progress > 0 ? 'Downloading...' : undefined,
                  eta: downloadStatus.progress > 0 ? 'Calculating...' : undefined
                };
              }
              return model;
            }));
            
            setModalDetails([
              `Downloading: ${downloadStatus.current_model}`,
              `Progress: ${downloadStatus.progress}%`,
              'This may take several minutes depending on your internet speed'
            ]);
          }
          
          // Stop polling if download is complete or error
          if (downloadStatus.status === 'completed') {
            clearInterval(pollInterval);
            setModalStatus('completed');
            setModalProgress(100);
            setModalCurrentStep('Download completed successfully!');
            setModalDetails(['All models downloaded successfully', 'Models are now ready to use']);
            
            // Mark all models as completed
            setModalModelProgress(prev => prev.map(model => ({
              ...model,
              status: 'completed' as const,
              progress: 100
            })));
            
            // Refresh models after download
            await fetchModels();
          } else if (downloadStatus.status === 'error') {
            clearInterval(pollInterval);
            setModalStatus('error');
            setModalError(downloadStatus.error || 'Download failed');
            setModalDetails(['Download encountered an error', 'Please check your internet connection and try again']);
            
            // Mark current model as error
            setModalModelProgress(prev => prev.map(model => {
              if (model.status === 'downloading') {
                return {
                  ...model,
                  status: 'error' as const
                };
              }
              return model;
            }));
          }
        }, 1000);
      } else {
        const errorData = await response.json();
        setModalStatus('error');
        setModalError(errorData.detail || 'Failed to start download');
        setModalDetails(['Failed to initiate download', 'Please try again']);
      }
    } catch (err: unknown) {
      setModalStatus('error');
      setModalError(err instanceof Error ? err.message : 'Network error occurred');
      setModalDetails(['Network error occurred', 'Please check your connection']);
    }
  };

  const deleteModels = async () => {
    setModalType('delete');
    setModalOpen(true);
    setModalStatus('in_progress');
    setModalProgress(0);
    setModalCurrentStep('Preparing to delete models...');
    setModalDetails(['This will free up significant disk space', 'Deleting model files and cache directories']);
    setModalError(null);
    
    try {
      const response = await fetch('http://localhost:8000/delete-models', {
        method: 'POST',
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Simulate progress steps
        setModalProgress(25);
        setModalCurrentStep('Deleting model directories...');
        setModalDetails(['Removing Stable Video Diffusion models', 'Removing Stable Diffusion models', 'Removing Bark models']);
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setModalProgress(50);
        setModalCurrentStep('Clearing cache directories...');
        setModalDetails(['Clearing HuggingFace cache', 'Clearing PyTorch cache', 'Freeing up disk space']);
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setModalProgress(75);
        setModalCurrentStep('Restarting model system...');
        setModalDetails(['Updating model registry', 'Refreshing available models']);
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setModalProgress(100);
        setModalStatus('completed');
        setModalCurrentStep('Deletion completed successfully!');
        setModalDetails([
          `Deleted ${result.deleted_directories || 0} model directories`,
          `Freed ${result.freed_space_mb || 0} MB of disk space`,
          'Models have been removed from the system'
        ]);
        
        // Refresh models after deletion
        await fetchModels();
        setError(null);
      } else {
        const errorData = await response.json();
        setModalStatus('error');
        setModalError(errorData.detail || 'Failed to delete models');
        setModalDetails(['Deletion failed', 'Please try again or check system permissions']);
      }
    } catch (err: unknown) {
      setModalStatus('error');
      setModalError(err instanceof Error ? err.message : 'Network error occurred');
      setModalDetails(['Network error occurred', 'Please check your connection and try again']);
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
          {/* Model Details - Show when models are downloaded */}
          {(models.video_models.length > 0 || models.audio_models.length > 0) && (
            <div className="p-4 rounded-xl bg-accent-green/5 border border-accent-green/20">
              <div className="flex items-center space-x-2 mb-4">
                <CheckCircle className="h-5 w-5 text-accent-green" />
                <span className="font-medium text-accent-green">Models Downloaded</span>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Video Models */}
                {models.video_models.length > 0 && (
                  <div className="space-y-3">
                    <h4 className="font-semibold text-sm text-gray-900 dark:text-slate-100 flex items-center space-x-2">
                      <Play className="h-4 w-4 text-accent-blue" />
                      <span>Video Models</span>
                    </h4>
                    {models.video_models.map((model) => (
                      <div key={model.id} className="p-3 rounded-lg bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-600">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-sm">{model.name}</span>
                          <div className={`px-2 py-1 rounded-full text-xs ${
                            model.loaded 
                              ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400' 
                              : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                          }`}>
                            {model.loaded ? 'Loaded' : 'Available'}
                          </div>
                        </div>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <div className="flex items-center space-x-1">
                            <Clock className="h-3 w-3" />
                            <span>Max: {model.max_duration}s</span>
                          </div>
                          {model.resolution && (
                            <div className="flex items-center space-x-1">
                              <FileText className="h-3 w-3" />
                              <span>{model.resolution}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Audio Models */}
                {models.audio_models.length > 0 && (
                  <div className="space-y-3">
                    <h4 className="font-semibold text-sm text-gray-900 dark:text-slate-100 flex items-center space-x-2">
                      <Volume2 className="h-4 w-4 text-accent-violet" />
                      <span>Audio Models</span>
                    </h4>
                    {models.audio_models.map((model) => (
                      <div key={model.id} className="p-3 rounded-lg bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-600">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-sm">{model.name}</span>
                          <div className={`px-2 py-1 rounded-full text-xs ${
                            model.loaded 
                              ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400' 
                              : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                          }`}>
                            {model.loaded ? 'Loaded' : 'Available'}
                          </div>
                        </div>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <div className="flex items-center space-x-1">
                            <Clock className="h-3 w-3" />
                            <span>Max: {model.max_duration}s</span>
                          </div>
                          {model.sample_rate && (
                            <div className="flex items-center space-x-1">
                              <FileText className="h-3 w-3" />
                              <span>{model.sample_rate}Hz</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              <div className="mt-4 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                <div className="flex items-center space-x-2">
                  <HardDrive className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium text-blue-800 dark:text-blue-200">Storage Information</span>
                </div>
                <p className="text-xs text-blue-600 dark:text-blue-300 mt-1">
                  Models are stored locally and ready for generation. You can delete them to free up disk space.
                </p>
              </div>
            </div>
          )}

          {/* Download Progress - Show when downloading */}
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
              
              <div className="mt-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                <div className="flex items-center space-x-2 mb-2">
                  <CloudDownload className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium text-blue-800 dark:text-blue-200">Download Details</span>
                </div>
                <div className="space-y-1 text-xs text-blue-600 dark:text-blue-300">
                  <p>• Stable Video Diffusion (~4GB) - For video generation</p>
                  <p>• Stable Diffusion (~4GB) - For image processing</p>
                  <p>• Bark (~5GB) - For audio generation</p>
                  <p>• Total size: ~13GB</p>
                </div>
              </div>
            </div>
          )}

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
                className="px-6 py-3 rounded-xl font-semibold transition-all duration-200 hover:scale-105 bg-red-500/10 text-red-600 hover:bg-red-500/20 border border-red-500/30"
              >
                <div className="flex items-center space-x-2">
                  <Trash2 className="h-4 w-4" />
                  <span>Delete Models</span>
                </div>
              </button>
              ) : (
              /* Show download button if no models are downloaded */
              <button
                onClick={startDownload}
                className="px-6 py-3 rounded-xl font-semibold transition-all duration-200 hover:scale-105 bg-accent-blue/10 text-accent-blue hover:bg-accent-blue/20 border border-accent-blue/30"
              >
                <div className="flex items-center space-x-2">
                  <CloudDownload className="h-4 w-4" />
                  <span>Download Models</span>
                </div>
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
                  {settings?.storage_usage_bytes ? (settings.storage_usage_bytes / (1024*1024)).toFixed(2) : '—'} MB
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
                <span>Clear Data</span>
              </button>
              
            </div>
          </div>
        </div>
      </div>

      {/* Progress Modal */}
      <ProgressModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        title={modalType === 'download' ? 'Downloading AI Models' : 'Deleting AI Models'}
        description={modalType === 'download' 
          ? 'Downloading the latest AI models for optimal performance' 
          : 'Removing downloaded models to free up disk space'
        }
        progress={modalProgress}
        status={modalStatus}
        currentStep={modalCurrentStep}
        details={modalDetails}
        error={modalError || undefined}
        type={modalType}
        modelProgress={modalModelProgress}
      />
    </div>
  );
}


