'use client';

import { useState, useEffect } from 'react';
import { Play, Settings, Loader2, Type, Upload, X, Monitor, Zap, Clock, Target, RotateCcw } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
import { useGenerating } from '@/contexts/GeneratingContext';
import { useModelCheck } from '@/hooks/useModelCheck';
import ModelDownloadModal from '@/components/ModelDownloadModal';
import { getApiUrl } from '@/config';

export default function Page() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [settings, setSettings] = useState({
    format: 'mp4',
    model: 'animatediff',
    // Advanced settings
    width: 512,
    height: 512,
    numFrames: 24,
    numInferenceSteps: 30,
    guidanceScale: 7.5,
    motionScale: 1.5,
    fps: 8,
    seed: 42
  });
  const [showSettings, setShowSettings] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const colors = useThemeColors();
  const { startGenerating, stopGenerating } = useGenerating();
  
  // Model checking
  const { getMissingModels, checkModels } = useModelCheck();
  const [showDownloadModal, setShowDownloadModal] = useState(false);

  // Preset configurations for different hardware
  const presets = {
    'ultra-fast': {
      name: 'Ultra Fast',
      description: 'Quick generation, basic quality',
      width: 256,
      height: 256,
      numFrames: 8,
      numInferenceSteps: 15,
      guidanceScale: 7.0,
      motionScale: 1.2,
      fps: 6,
      estimatedTime: '30-60s'
    },
    'balanced': {
      name: 'Balanced',
      description: 'Good quality, reasonable speed',
      width: 384,
      height: 384,
      numFrames: 16,
      numInferenceSteps: 25,
      guidanceScale: 7.5,
      motionScale: 1.4,
      fps: 8,
      estimatedTime: '1-2min'
    },
    'high-quality': {
      name: 'High Quality',
      description: 'Great quality, longer generation',
      width: 512,
      height: 512,
      numFrames: 24,
      numInferenceSteps: 30,
      guidanceScale: 7.5,
      motionScale: 1.5,
      fps: 8,
      estimatedTime: '2-4min'
    },
  };

  const applyPreset = (presetKey: keyof typeof presets) => {
    const preset = presets[presetKey];
    setSettings(prev => ({
      ...prev,
      width: preset.width,
      height: preset.height,
      numFrames: preset.numFrames,
      numInferenceSteps: preset.numInferenceSteps,
      guidanceScale: preset.guidanceScale,
      motionScale: preset.motionScale,
      fps: preset.fps
    }));
  };

  const resetToDefaults = () => {
    setSettings(prev => ({
      ...prev,
      width: 512,
      height: 512,
      numFrames: 24,
      numInferenceSteps: 30,
      guidanceScale: 7.5,
      motionScale: 1.5,
      fps: 8,
      seed: 42
    }));
  };

  const calculateVideoDuration = () => {
    return (settings.numFrames / settings.fps).toFixed(1);
  };

  const estimateFileSize = () => {
    const pixels = settings.width * settings.height;
    const frames = settings.numFrames;
    // Rough estimation: ~0.5-1KB per pixel per frame
    const estimatedKB = (pixels * frames * 0.75) / 1024;
    if (estimatedKB < 1024) {
      return `${estimatedKB.toFixed(0)} KB`;
    } else {
      return `${(estimatedKB / 1024).toFixed(1)} MB`;
    }
  };


  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    
    // Check if required models are available
    const missingModels = getMissingModels('video', settings.model);
    if (missingModels.length > 0) {
      setShowDownloadModal(true);
      return;
    }
    
    setIsGenerating(true);
    
    try {
      const requestData = {
        prompt: prompt.trim(),
        model_type: 'video',
        model_name: settings.model,
        duration: parseInt(calculateVideoDuration()),
        output_format: settings.format,
        // Advanced settings
        width: settings.width,
        height: settings.height,
        num_frames: settings.numFrames,
        num_inference_steps: settings.numInferenceSteps,
        guidance_scale: settings.guidanceScale,
        motion_scale: settings.motionScale,
        fps: settings.fps,
        seed: settings.seed
      };
      
      const response = await fetch(getApiUrl('/generate'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });
      
      const result = await response.json();
      
      if (result.job_id) {
        // Start global generating modal
        startGenerating(
          result.job_id,
          'video',
          'Generating Video',
          'Your video is being created from text'
        );
      }
    } catch (error) {
      console.error('Generation failed:', error);
      setIsGenerating(false);
      stopGenerating();
    }
  };

  return (
    <div className="space-y-8">
      <div className="text-center mb-6">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-accent-blue to-accent-violet mb-4 shadow-2xl">
          <Play className="h-8 w-8 text-white" />
        </div>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent">Video Generator</h1>
        <p className={`mt-1 ${colors.text.secondary}`}>Generate high-quality videos from text prompts using AI</p>
      </div>

      <div className="max-w-4xl mx-auto space-y-6">

        {/* Main Generation Form */}
        <div className="bg-white dark:bg-slate-800/50 rounded-2xl border p-8">
          <div className="space-y-6">
            {/* Prompt Input */}
            <div>
              <label className="block text-sm font-semibold mb-3 text-gray-900 dark:text-white">
                Text Prompt
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={4}
                placeholder="Describe the video you want to generate..."
                className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-accent-blue focus:border-transparent transition-all duration-200"
              />
            </div>


            {/* Settings */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="block text-sm font-semibold text-gray-900 dark:text-white">
                  Generation Settings
                </label>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                    className="flex items-center space-x-2 px-3 py-1 text-sm text-accent-blue hover:bg-accent-blue/10 rounded-lg transition-colors"
                  >
                    <Settings className="h-4 w-4" />
                    <span>{showAdvancedSettings ? 'Hide' : 'Advanced'} Settings</span>
                  </button>
                  <button
                    onClick={resetToDefaults}
                    className="flex items-center space-x-2 px-3 py-1 text-sm text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-slate-700 rounded-lg transition-colors"
                  >
                    <RotateCcw className="h-4 w-4" />
                    <span>Reset</span>
                  </button>
                </div>
              </div>

              {/* Basic Settings */}
              <div className="p-4 bg-gray-50 dark:bg-slate-700/50 rounded-xl border mb-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Model
                    </label>
                    <select
                      value={settings.model}
                      onChange={(e) => setSettings(prev => ({ ...prev, model: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-slate-800 text-gray-900 dark:text-white"
                    >
                      <option value="animatediff">AnimateDiff</option>
                      <option value="stable-video-diffusion">Stable Video Diffusion</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Output Format
                    </label>
                    <select
                      value={settings.format}
                      onChange={(e) => setSettings(prev => ({ ...prev, format: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-slate-800 text-gray-900 dark:text-white"
                    >
                      <option value="mp4">MP4 (Recommended)</option>
                      <option value="gif">GIF</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Preset Configurations */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Quick Presets
                </label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {Object.entries(presets).map(([key, preset]) => (
                    <button
                      key={key}
                      onClick={() => applyPreset(key as keyof typeof presets)}
                      className="p-3 text-left border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-slate-800 hover:border-accent-blue hover:bg-accent-blue/5 transition-colors"
                    >
                      <div className="font-medium text-sm text-gray-900 dark:text-white">{preset.name}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">{preset.description}</div>
                      <div className="text-xs text-accent-blue mt-1">{preset.estimatedTime}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Advanced Settings */}
              {showAdvancedSettings && (
                <div className="p-4 bg-gray-50 dark:bg-slate-700/50 rounded-xl border">
                  <div className="space-y-6">
                    {/* Video Info Display */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-3 bg-white dark:bg-slate-800 rounded-lg border">
                      <div className="text-center">
                        <Monitor className="h-5 w-5 mx-auto text-accent-blue mb-1" />
                        <div className="text-sm font-medium text-gray-900 dark:text-white">{settings.width}×{settings.height}</div>
                        <div className="text-xs text-gray-500">Resolution</div>
                      </div>
                      <div className="text-center">
                        <Clock className="h-5 w-5 mx-auto text-accent-blue mb-1" />
                        <div className="text-sm font-medium text-gray-900 dark:text-white">{calculateVideoDuration()}s</div>
                        <div className="text-xs text-gray-500">Duration</div>
                      </div>
                      <div className="text-center">
                        <Zap className="h-5 w-5 mx-auto text-accent-blue mb-1" />
                        <div className="text-sm font-medium text-gray-900 dark:text-white">{settings.numFrames}</div>
                        <div className="text-xs text-gray-500">Frames</div>
                      </div>
                      <div className="text-center">
                        <Target className="h-5 w-5 mx-auto text-accent-blue mb-1" />
                        <div className="text-sm font-medium text-gray-900 dark:text-white">{estimateFileSize()}</div>
                        <div className="text-xs text-gray-500">Est. Size</div>
                      </div>
                    </div>

                    {/* Resolution Settings */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Resolution
                      </label>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">Width</label>
                          <select
                            value={settings.width}
                            onChange={(e) => setSettings(prev => ({ ...prev, width: parseInt(e.target.value) }))}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-slate-800 text-gray-900 dark:text-white"
                          >
                            <option value={256}>256px</option>
                            <option value={384}>384px</option>
                            <option value={512}>512px</option>
                            <option value={768}>768px</option>
                            <option value={1024}>1024px</option>
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">Height</label>
                          <select
                            value={settings.height}
                            onChange={(e) => setSettings(prev => ({ ...prev, height: parseInt(e.target.value) }))}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-slate-800 text-gray-900 dark:text-white"
                          >
                            <option value={256}>256px</option>
                            <option value={384}>384px</option>
                            <option value={512}>512px</option>
                            <option value={768}>768px</option>
                            <option value={1024}>1024px</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    {/* Video Settings */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Frames: {settings.numFrames} ({calculateVideoDuration()}s)
                        </label>
                        <input
                          type="range"
                          min="4"
                          max="48"
                          step="4"
                          value={settings.numFrames}
                          onChange={(e) => setSettings(prev => ({ ...prev, numFrames: parseInt(e.target.value) }))}
                          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>4 (0.5s)</span>
                          <span>48 (6s)</span>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Frame Rate: {settings.fps} FPS
                        </label>
                        <select
                          value={settings.fps}
                          onChange={(e) => setSettings(prev => ({ ...prev, fps: parseInt(e.target.value) }))}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-slate-800 text-gray-900 dark:text-white"
                        >
                          <option value={6}>6 FPS (Cinematic)</option>
                          <option value={8}>8 FPS (Standard)</option>
                          <option value={12}>12 FPS (Smooth)</option>
                          <option value={24}>24 FPS (Very Smooth)</option>
                        </select>
                      </div>
                    </div>

                    {/* Quality Settings */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Inference Steps: {settings.numInferenceSteps}
                        </label>
                        <input
                          type="range"
                          min="10"
                          max="50"
                          step="5"
                          value={settings.numInferenceSteps}
                          onChange={(e) => setSettings(prev => ({ ...prev, numInferenceSteps: parseInt(e.target.value) }))}
                          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>10 (Fast)</span>
                          <span>50 (Ultra)</span>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Guidance Scale: {settings.guidanceScale}
                        </label>
                        <input
                          type="range"
                          min="5.0"
                          max="12.0"
                          step="0.5"
                          value={settings.guidanceScale}
                          onChange={(e) => setSettings(prev => ({ ...prev, guidanceScale: parseFloat(e.target.value) }))}
                          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>5.0 (Creative)</span>
                          <span>12.0 (Strict)</span>
                        </div>
                      </div>
                    </div>

                    {/* Motion and Seed Settings */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Motion Scale: {settings.motionScale}
                        </label>
                        <input
                          type="range"
                          min="0.5"
                          max="2.5"
                          step="0.1"
                          value={settings.motionScale}
                          onChange={(e) => setSettings(prev => ({ ...prev, motionScale: parseFloat(e.target.value) }))}
                          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>0.5 (Subtle)</span>
                          <span>2.5 (Dynamic)</span>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Seed: {settings.seed}
                        </label>
                        <div className="flex space-x-2">
                          <input
                            type="number"
                            min="0"
                            max="1000000"
                            value={settings.seed}
                            onChange={(e) => setSettings(prev => ({ ...prev, seed: parseInt(e.target.value) || 42 }))}
                            className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-slate-800 text-gray-900 dark:text-white"
                          />
                          <button
                            onClick={() => setSettings(prev => ({ ...prev, seed: Math.floor(Math.random() * 1000000) }))}
                            className="px-3 py-2 bg-accent-blue text-white rounded-lg hover:bg-accent-blue/90 transition-colors"
                          >
                            Random
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            {/* Generate Button */}
            <div className="pt-4">
              <button
                onClick={handleGenerate}
                disabled={!prompt.trim()}
                className="group relative w-full flex items-center justify-center space-x-4 px-10 py-6 bg-gradient-to-r from-accent-blue to-accent-violet text-white rounded-2xl font-bold text-xl hover:from-accent-blue/90 hover:to-accent-violet/90 transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-2xl hover:shadow-accent-blue/25"
              >
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-accent-blue to-accent-violet opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                <Play className="h-7 w-7 group-hover:scale-110 transition-transform duration-300" />
                <span>Generate Video</span>
                <div className="absolute inset-0 rounded-2xl border-2 border-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Model Download Modal */}
      <ModelDownloadModal
        isOpen={showDownloadModal}
        onClose={() => setShowDownloadModal(false)}
        missingModels={getMissingModels('video', settings.model)}
        modelType="video"
        onModelsDownloaded={() => {
          checkModels();
          setShowDownloadModal(false);
        }}
      />
    </div>
  );
}