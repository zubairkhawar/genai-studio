'use client';

import { useState, useEffect } from 'react';
import { Loader2, AlertCircle, Sparkles, Settings } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
import { getApiUrl } from '@/config';

interface GenerationFormProps {
  onJobCreated: () => void;
  initialModelType?: 'video' | 'audio';
}

interface Model {
  id: string;
  name: string;
  description: string;
  max_duration: number;
  resolution?: string;
  sample_rate?: number;
  loaded: boolean;
}

export function GenerationForm({ onJobCreated, initialModelType }: GenerationFormProps) {
  const [prompt, setPrompt] = useState('');
  const [modelType, setModelType] = useState<'video' | 'audio'>(initialModelType ?? 'video');
  const [selectedModel, setSelectedModel] = useState('');
  const [duration, setDuration] = useState(5);
  const [sampleRate, setSampleRate] = useState(22050);
  const [outputFormat, setOutputFormat] = useState('mp4');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [models, setModels] = useState<{
    video_models: Model[];
    audio_models: Model[];
  }>({ video_models: [], audio_models: [] });
  const colors = useThemeColors();

  // Load models on component mount
  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch(getApiUrl('/models'));
      const data = await response.json();
      setModels(data);
      
      // Set default model
      const savedVideo = localStorage.getItem('default_video_model') || '';
      const savedAudio = localStorage.getItem('default_audio_model') || '';
      if (data.video_models.length > 0 && modelType === 'video') {
        const fallback = data.video_models[0].id;
        const preferred = data.video_models.find((m: any) => m.id === savedVideo)?.id || fallback;
        setSelectedModel(preferred);
      } else if (data.audio_models.length > 0 && modelType === 'audio') {
        const fallback = data.audio_models[0].id;
        const preferred = data.audio_models.find((m: any) => m.id === savedAudio)?.id || fallback;
        setSelectedModel(preferred);
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch(getApiUrl('/generate'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          model_type: modelType,
          model_name: selectedModel,
          duration: modelType === 'video' ? duration : undefined,
          sample_rate: modelType === 'audio' ? sampleRate : undefined,
          output_format: outputFormat,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Generation failed');
      }

      const result = await response.json();
      console.log('Generation started:', result);
      
      // Clear form
      setPrompt('');
      
      // Refresh job list
      onJobCreated();
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const currentModels = modelType === 'video' ? models.video_models : models.audio_models;
  const isLockedType = Boolean(initialModelType);
  const selectedModelObj = currentModels.find(m => m.id === selectedModel);

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Stepper */}
      <div className="flex items-center gap-3">
        {[1,2].map((n, idx) => (
          <div key={n} className={`flex-1 h-2 rounded-full ${idx === 0 ? 'bg-accent-blue' : 'bg-gray-200 dark:bg-gray-800'}`}></div>
        ))}
      </div>

      {/* Step 1: Model Selection */}
      <div className="space-y-4 p-6 rounded-2xl border shadow-xl bg-white dark:bg-gradient-to-br dark:from-dark-card/90 dark:to-dark-bg-secondary/50 backdrop-blur-md">
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-8 h-8 bg-accent-violet text-white rounded-full text-sm font-bold">
            1
          </div>
          <h3 className={`text-lg font-semibold text-gray-900 dark:text-slate-100`}>
            Select AI Model
          </h3>
        </div>
        <div className="ml-11">
          <label className={`block text-sm font-medium text-gray-600 dark:text-slate-300 mb-3`}>
            Choose the AI model for generation
          </label>
        <div className="relative">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className={`w-full px-4 py-4 bg-white dark:bg-slate-800/50 border-2 border-gray-200 dark:border-slate-600 rounded-2xl shadow-lg focus:outline-none focus:ring-4 focus:ring-accent-blue/20 focus:border-accent-blue transition-all duration-300 text-gray-900 dark:text-slate-100 hover:border-accent-blue/50 hover:shadow-xl appearance-none cursor-pointer`}
            required
          >
            <option value="">✨ Choose your AI model...</option>
            {currentModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
          <div className="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none">
            <svg className="w-5 h-5 text-accent-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
        {selectedModelObj && (
          <div className="mt-3 p-4 rounded-xl border border-accent-blue/30 bg-accent-blue/5">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="font-semibold text-accent-blue mb-2">{selectedModelObj.name}</h4>
                <p className={`text-sm text-gray-600 dark:text-slate-300 mb-3`}>
                  {selectedModelObj.description}
                </p>
                <div className="flex items-center gap-2 text-xs">
                  {modelType === 'video' && selectedModelObj.resolution && (
                    <span className="px-2 py-1 rounded-full bg-accent-blue/10 text-accent-blue">
                      {selectedModelObj.resolution}
                    </span>
                  )}
                  {modelType === 'audio' && selectedModelObj.sample_rate && (
                    <span className="px-2 py-1 rounded-full bg-accent-violet/10 text-accent-violet">
                      {selectedModelObj.sample_rate}Hz
                    </span>
                  )}
                  <span className="px-2 py-1 rounded-full bg-accent-violet/10 text-accent-violet">
                    Max: {selectedModelObj.max_duration}s
                  </span>
                </div>
              </div>
            </div>
            <div className="mt-3 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
              <div className="flex items-center space-x-2">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                <span className="text-sm text-yellow-700 dark:text-yellow-400">
                  Make sure to load this model in Settings before generating
                </span>
              </div>
            </div>
          </div>
        )}
        </div>
      </div>

      {/* Step 2: Prompt Input */}
      <div className="space-y-4 p-6 rounded-2xl border shadow-xl bg-white dark:bg-gradient-to-br dark:from-dark-card/90 dark:to-dark-bg-secondary/50 backdrop-blur-md">
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-8 h-8 bg-accent-green text-white rounded-full text-sm font-bold">
            2
          </div>
          <h3 className={`text-lg font-semibold text-gray-900 dark:text-slate-100`}>
            Enter Your Prompt
          </h3>
        </div>
        <div className="ml-11">
          <label className={`block text-sm font-medium text-gray-600 dark:text-slate-300 mb-3`}>
            Describe what you want to create
          </label>
        <div className="relative">
          <textarea
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value);
              // Auto-expand textarea
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
            }}
            placeholder="✨ Describe what you want to create... Be as detailed as possible for better results!"
            className={`w-full px-4 py-4 bg-white dark:bg-slate-800/50 border-2 border-gray-200 dark:border-slate-600 rounded-2xl shadow-lg focus:outline-none focus:ring-4 focus:ring-accent-blue/20 focus:border-accent-blue transition-all duration-300 text-gray-900 dark:text-slate-100 resize-none min-h-[100px] hover:border-accent-blue/50 hover:shadow-xl`}
            rows={3}
            required
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {['cinematic', 'highly detailed', 'dramatic lighting', '4k', 'realistic', 'fast-paced'].map(tag => (
              <button
                key={tag}
                type="button"
                onClick={() => setPrompt(prev => (prev ? prev + `, ${tag}` : tag))}
                className="text-xs px-3 py-1.5 rounded-full border border-accent-blue/30 text-accent-blue hover:bg-accent-blue/10 hover:border-accent-blue/50 transition-all duration-200 hover:scale-105"
              >
                {tag}
              </button>
            ))}
          </div>
          <div className="absolute bottom-3 right-3 flex items-center space-x-2">
            <div className={`text-xs px-2 py-1 rounded-full ${
              prompt.length > 400 ? 'bg-accent-red/10 text-accent-red' : 
              prompt.length > 300 ? 'bg-accent-violet/10 text-accent-violet' : 
              'bg-accent-green/10 text-accent-green'
            }`}>
              {prompt.length}/500
            </div>
            <div className="text-xs text-gray-400">
              {Math.ceil(prompt.length / 4)} tokens
            </div>
          </div>
        </div>
        </div>
      </div>

      {/* Advanced Settings Toggle */}
      <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className={`flex items-center justify-between w-full p-3 text-sm font-medium text-gray-600 dark:text-slate-300 hover:text-gray-900 dark:text-slate-100 transition-colors rounded-lg hover:${colors.bg.secondary}`}
        >
          <div className="flex items-center space-x-2">
            <Settings className="h-4 w-4" />
            <span>Advanced Settings</span>
          </div>
          <div className={`transform transition-transform duration-200 ${showAdvanced ? 'rotate-180' : ''}`}>
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </button>
      </div>

      {/* Advanced Settings */}
      {showAdvanced && (
        <div className={`space-y-6 p-6 ${colors.bg.secondary} rounded-2xl border border-gray-200 dark:border-slate-600 animate-slide-down shadow-xl backdrop-blur-md`}>
          <div className="flex items-center space-x-2 mb-4">
            <div className="w-2 h-2 bg-accent-blue rounded-full"></div>
            <h4 className={`text-sm font-semibold text-gray-900 dark:text-slate-100`}>Generation Parameters</h4>
          </div>

          {/* Duration (for video) */}
          {modelType === 'video' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className={`text-sm font-medium text-gray-900 dark:text-slate-100`}>
                  Duration
                </label>
                <span className={`text-sm text-gray-600 dark:text-slate-300`}>{duration}s</span>
              </div>
              <div className="relative">
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={duration}
                  onChange={(e) => setDuration(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>1s</span>
                  <span>30s</span>
                </div>
              </div>
            </div>
          )}

          {/* Sample Rate (for audio) */}
          {modelType === 'audio' && (
            <div className="space-y-3">
              <label className={`text-sm font-medium text-gray-900 dark:text-slate-100`}>
                Sample Rate
              </label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { value: 16000, label: '16kHz', desc: 'Low quality' },
                  { value: 22050, label: '22kHz', desc: 'Standard' },
                  { value: 44100, label: '44kHz', desc: 'CD quality' },
                  { value: 48000, label: '48kHz', desc: 'Studio' }
                ].map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setSampleRate(option.value)}
                    className={`p-3 rounded-xl border text-left transition-all duration-200 hover:scale-105 ${
                      sampleRate === option.value
                        ? 'border-accent-blue bg-accent-blue/10 text-accent-blue shadow-lg'
                        : `border-gray-200 dark:border-slate-600 hover:border-accent-blue/50 hover:bg-accent-blue/5`
                    }`}
                  >
                    <div className="font-medium text-sm">{option.label}</div>
                    <div className="text-xs text-gray-500">{option.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Output Format */}
          <div className="space-y-3">
            <label className={`text-sm font-medium text-gray-900 dark:text-slate-100`}>
              Output Format
            </label>
          <div className="grid grid-cols-3 gap-2">
              {(modelType === 'video' 
                ? [
                    { value: 'mp4', label: 'MP4', desc: 'Universal' },
                    { value: 'webm', label: 'WebM', desc: 'Web optimized' },
                    { value: 'avi', label: 'AVI', desc: 'Legacy' }
                  ]
                : [
                  { value: 'wav', label: 'WAV', desc: 'Uncompressed' },
                  { value: 'mp3', label: 'MP3', desc: 'Compressed' }
                  ]
              ).map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setOutputFormat(option.value)}
                  className={`p-3 rounded-xl border text-center transition-all duration-200 hover:scale-105 ${
                    outputFormat === option.value
                      ? 'border-accent-violet bg-accent-violet/10 text-accent-violet shadow-lg'
                      : `border-gray-200 dark:border-slate-600 hover:border-accent-violet/50 hover:bg-accent-violet/5`
                  }`}
                >
                  <div className="font-medium text-sm">{option.label}</div>
                  <div className="text-xs text-gray-500">{option.desc}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className={`flex items-center space-x-3 p-4 bg-accent-red/10 border border-accent-red/20 rounded-xl text-accent-red`}>
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm font-medium">{error}</span>
        </div>
      )}

      {/* Submit Button */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-500">
            {!prompt.trim() && "Enter a prompt to continue"}
            {prompt.trim() && !selectedModel && "Select a model to continue"}
            {prompt.trim() && selectedModel && !isGenerating && "Ready to generate!"}
            {isGenerating && "Generating your content..."}
          </div>
          <div className="text-xs text-gray-400">
            Step 2 of 2
          </div>
        </div>
        <button
          type="submit"
          disabled={isGenerating || !prompt.trim() || !selectedModel}
          className={`group relative w-full flex items-center justify-center px-6 py-4 rounded-xl font-semibold text-white transition-all duration-300 transform ${
            isGenerating || !prompt.trim() || !selectedModel
              ? 'opacity-50 cursor-not-allowed bg-gray-400'
              : `bg-gradient-to-r from-accent-blue to-accent-violet hover:from-accent-blue-light hover:to-accent-violet-light hover:scale-105 hover:shadow-lg active:scale-95`
          }`}
        >
          {isGenerating && (
            <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-accent-blue to-accent-violet opacity-20 animate-pulse"></div>
          )}
          <div className="relative flex items-center space-x-3">
            {isGenerating ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                <span>Generating {modelType === 'video' ? 'Video' : 'Audio'}...</span>
              </>
            ) : (
              <>
                <div className="relative">
                  <Sparkles className="h-5 w-5 group-hover:animate-pulse" />
                  {!isGenerating && (
                    <div className="absolute inset-0 h-5 w-5 rounded-full bg-white/20 animate-ping"></div>
                  )}
                </div>
                <span>Generate {modelType === 'video' ? 'Video' : 'Audio'}</span>
              </>
            )}
          </div>
        </button>
      </div>
    </form>
  );
}
