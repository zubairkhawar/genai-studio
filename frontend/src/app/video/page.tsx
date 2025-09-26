'use client';

import { useState, useEffect, useRef } from 'react';
import { Play, Download, RefreshCw, Settings, CheckCircle, Clock, Loader2, Upload, Image as ImageIcon, Type } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
import { getApiUrl } from '@/config';

interface GenerationResult {
  id: string;
  prompt: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  outputFile?: string;
  error?: string;
  message?: string;
  createdAt: string;
  settings: {
    format: string;
  };
}

export default function Page() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [results, setResults] = useState<GenerationResult[]>([]);
  const [currentStep, setCurrentStep] = useState<'prompt' | 'progress' | 'results'>('prompt');
  const [settings, setSettings] = useState({
    format: 'mp4',
    model: 'animatediff'  // Default to AnimateDiff for GIF generation
  });
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [generationMode, setGenerationMode] = useState<'text-to-video' | 'image-to-video'>('text-to-video');
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const colors = useThemeColors();
  
  // Check for existing job on mount
  useEffect(() => {
    const checkExistingJob = async () => {
      try {
        const response = await fetch(getApiUrl('/jobs'));
        if (response.ok) {
          const jobs = await response.json();
          const activeJob = jobs.find((job: any) => 
            job.status === 'processing' || job.status === 'queued'
          );
          
          if (activeJob) {
            setCurrentJobId(activeJob.job_id);
            setCurrentStep('progress');
            setIsGenerating(true);
            
            // Add to results if not already there
            const existingResult = results.find(r => r.id === activeJob.job_id);
            if (!existingResult) {
              const newResult: GenerationResult = {
                id: activeJob.job_id,
                prompt: activeJob.prompt || '',
                status: activeJob.status === 'processing' ? 'processing' : 'queued',
                progress: activeJob.progress || 0,
                createdAt: activeJob.created_at,
                settings
              };
              setResults([newResult]);
            }
          }
        }
      } catch (error) {
        console.error('Failed to check existing jobs:', error);
      }
    };
    
    checkExistingJob();
  }, []);
  
  // Poll for job updates when there's an active job
  useEffect(() => {
    if (!currentJobId) return;
    
    const pollJobStatus = async () => {
      try {
        const response = await fetch(getApiUrl('/jobs'));
        if (response.ok) {
          const jobs = await response.json();
          const currentJob = jobs.find((job: any) => job.job_id === currentJobId);
          
          if (currentJob) {
            setResults(prev => prev.map(result => 
              result.id === currentJobId 
                ? {
                    ...result,
                    status: currentJob.status === 'completed' ? 'completed' : 
                           currentJob.status === 'failed' ? 'failed' : 
                           currentJob.status === 'processing' ? 'processing' : 'queued',
                    progress: currentJob.progress || 0,
                    outputFile: currentJob.output_file,
                    error: currentJob.error,
                    message: currentJob.message
                  }
                : result
            ));
            
            // Update step based on status
            if (currentJob.status === 'completed') {
              setCurrentStep('results');
              setIsGenerating(false);
            } else if (currentJob.status === 'failed') {
              setCurrentStep('results');
              setIsGenerating(false);
            }
          }
        }
      } catch (error) {
        console.error('Failed to poll job status:', error);
      }
    };
    
    const interval = setInterval(pollJobStatus, 2000);
    return () => clearInterval(interval);
  }, [currentJobId]);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
      }
      
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('Image file size must be less than 10MB');
        return;
      }
      
      setUploadedImage(file);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setUploadedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const fetchJobs = async () => {
    if (currentStep !== 'progress' && currentStep !== 'results') return;
    
    try {
      const response = await fetch(getApiUrl('/jobs'));
      const jobs = await response.json();
      
      // Filter for video jobs and convert to our format
      const videoJobs = jobs
        .filter((job: any) => job.model_type === 'video')
        .map((job: any) => ({
          id: job.job_id,
          prompt: job.prompt,
          status: job.status,
          progress: job.progress || 0,
          outputFile: job.output_file,
          error: job.error,
          createdAt: job.created_at,
          settings: {
            resolution: '1024x576', // Default, could be enhanced
            duration: job.duration || 4,
            format: job.output_format || 'mp4'
          }
        }))
        .sort((a: any, b: any) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
      
      // Find current job or most recent
      const currentJob = currentJobId 
        ? videoJobs.find((job: any) => job.id === currentJobId)
        : videoJobs[0];
      
      if (currentJob) {
        setResults([currentJob]);
        
        // Check if job is completed and move to results step
        if (currentJob.status === 'completed' && currentStep === 'progress') {
          setCurrentStep('results');
          setIsGenerating(false);
        }
        
        // Check if job failed and move to results step
        if (currentJob.status === 'failed' && currentStep === 'progress') {
          setCurrentStep('results');
          setIsGenerating(false);
        }
      }
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
    }
  };

  useEffect(() => {
    if (currentStep === 'progress' || currentStep === 'results') {
    fetchJobs();
    const interval = setInterval(fetchJobs, 2000);
    return () => clearInterval(interval);
    }
  }, [currentStep, currentJobId]);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    
    // For image-to-video mode, require an uploaded image
    if (generationMode === 'image-to-video' && !uploadedImage) {
      alert('Please upload an image for image-to-video generation');
      return;
    }
    
    setIsGenerating(true);
    setCurrentStep('progress');
    
    // Clear previous results when starting new generation
    setResults([]);
    
    try {
      const requestData = {
        prompt: prompt.trim(),
        model_type: 'video',
        model_name: settings.model,
        duration: 4, // Optimal duration for video generation
        output_format: settings.format
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
        setCurrentJobId(result.job_id);
        
        // Add to results immediately
        const newResult: GenerationResult = {
          id: result.job_id,
          prompt: prompt.trim(),
          status: 'queued',
          progress: 0,
          createdAt: new Date().toISOString(),
          settings
        };
        setResults([newResult]);
      }
    } catch (error) {
      console.error('Generation failed:', error);
      setIsGenerating(false);
      setCurrentStep('prompt');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'processing': return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'queued': return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'failed': return <div className="h-5 w-5 rounded-full bg-red-500" />;
      default: return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'completed': return 'Completed';
      case 'processing': return 'Generating...';
      case 'queued': return 'Queued';
      case 'failed': return 'Failed';
      default: return 'Unknown';
    }
  };

  const renderStepIndicator = () => {
    const steps = [
      { key: 'prompt', label: 'Prompt & Settings', icon: Play },
      { key: 'progress', label: 'Generating', icon: Loader2 },
      { key: 'results', label: 'Results', icon: CheckCircle }
    ];

  return (
      <div className="flex items-center justify-center mb-8">
        <div className="flex items-center space-x-4">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = currentStep === step.key;
            const isCompleted = (currentStep === 'progress' && step.key === 'prompt') || 
                               (currentStep === 'results' && (step.key === 'prompt' || step.key === 'progress'));
            
            return (
              <div key={step.key} className="flex items-center">
                <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all duration-300 ${
                  isActive 
                    ? 'border-accent-blue bg-accent-blue text-white' 
                    : isCompleted 
                      ? 'border-green-500 bg-green-500 text-white'
                      : 'border-gray-300 bg-white text-gray-400'
                }`}>
                  {isCompleted && step.key !== 'results' ? (
                    <CheckCircle className="h-5 w-5" />
                  ) : (
                    <Icon className={`h-5 w-5 ${isActive && step.key === 'progress' ? 'animate-spin' : ''}`} />
                  )}
                </div>
                <span className={`ml-2 text-sm font-medium ${
                  isActive ? 'text-accent-blue' : isCompleted ? 'text-green-600' : 'text-gray-400'
                }`}>
                  {step.label}
                </span>
                {index < steps.length - 1 && (
                  <div className={`w-8 h-0.5 mx-4 ${
                    isCompleted ? 'bg-green-500' : 'bg-gray-300'
                  }`} />
                )}
              </div>
            );
          })}
                </div>
                </div>
    );
  };

  const renderPromptStep = () => (
    <div className="max-w-5xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-r from-accent-blue to-accent-violet mb-6 shadow-2xl">
          <Play className="h-10 w-10 text-white" />
        </div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent mb-4">
            Video Generator
          </h1>
        <p className={`text-xl ${colors.text.secondary} max-w-2xl mx-auto leading-relaxed`}>
          Transform your ideas into stunning videos with the power of AI. Describe your vision and watch it come to life.
        </p>
      </div>

      {/* Main Content Card */}
      <div className={`relative overflow-hidden rounded-3xl border shadow-2xl bg-white dark:bg-gradient-to-br dark:from-slate-900/90 dark:to-slate-800/50 backdrop-blur-md dark:border-slate-700/50`}>
        {/* Decorative Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-accent-blue/5 via-transparent to-accent-violet/5 dark:from-accent-blue/10 dark:via-transparent dark:to-accent-violet/10"></div>
        <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-bl from-accent-blue/10 to-transparent dark:from-accent-blue/20 rounded-full -translate-y-32 translate-x-32"></div>
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-gradient-to-tr from-accent-violet/10 to-transparent dark:from-accent-violet/20 rounded-full translate-y-24 -translate-x-24"></div>
        
        <div className="relative p-10">
          <div className="space-y-8">
            {/* Generation Mode Selection */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-accent-blue/10">
                  <Play className="h-5 w-5 text-accent-blue" />
                </div>
                <h2 className={`text-2xl font-bold ${colors.text.primary}`}>
                  Choose Generation Mode
                </h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button
                  onClick={() => setGenerationMode('text-to-video')}
                  className={`p-6 rounded-2xl border-2 transition-all duration-300 text-left ${
                    generationMode === 'text-to-video'
                      ? 'border-accent-blue bg-accent-blue/10 shadow-lg'
                      : 'border-gray-200 dark:border-slate-600 hover:border-accent-blue/50'
                  }`}
                >
                  <div className="flex items-center space-x-4">
                    <div className={`flex items-center justify-center w-12 h-12 rounded-xl ${
                      generationMode === 'text-to-video' ? 'bg-accent-blue/20' : 'bg-gray-100 dark:bg-slate-700'
                    }`}>
                      <Type className={`h-6 w-6 ${generationMode === 'text-to-video' ? 'text-accent-blue' : 'text-gray-500'}`} />
                    </div>
                    <div>
                      <h3 className={`text-lg font-bold ${colors.text.primary}`}>
                        Text-to-Video
                      </h3>
                      <p className={`text-sm ${colors.text.secondary}`}>
                        Generate video from text description only
          </p>
        </div>
                  </div>
                </button>
                
        <button 
                  onClick={() => setGenerationMode('image-to-video')}
                  className={`p-6 rounded-2xl border-2 transition-all duration-300 text-left ${
                    generationMode === 'image-to-video'
                      ? 'border-accent-blue bg-accent-blue/10 shadow-lg'
                      : 'border-gray-200 dark:border-slate-600 hover:border-accent-blue/50'
                  }`}
                >
                  <div className="flex items-center space-x-4">
                    <div className={`flex items-center justify-center w-12 h-12 rounded-xl ${
                      generationMode === 'image-to-video' ? 'bg-accent-blue/20' : 'bg-gray-100 dark:bg-slate-700'
                    }`}>
                      <ImageIcon className={`h-6 w-6 ${generationMode === 'image-to-video' ? 'text-accent-blue' : 'text-gray-500'}`} />
                    </div>
                    <div>
                      <h3 className={`text-lg font-bold ${colors.text.primary}`}>
                        Image-to-Video
                      </h3>
                      <p className={`text-sm ${colors.text.secondary}`}>
                        Animate your own image with text prompt
                      </p>
                    </div>
                  </div>
        </button>
              </div>
            </div>

            {/* Image Upload Section (for image-to-video mode) */}
            {generationMode === 'image-to-video' && (
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-accent-violet/10">
                    <Upload className="h-5 w-5 text-accent-violet" />
                  </div>
                  <h2 className={`text-2xl font-bold ${colors.text.primary}`}>
                    Upload Your Image
                  </h2>
      </div>

                <div className="space-y-4">
                  {!imagePreview ? (
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className="border-2 border-dashed border-gray-300 dark:border-slate-600 rounded-2xl p-8 text-center hover:border-accent-violet hover:bg-accent-violet/5 transition-all duration-300 cursor-pointer"
                    >
                      <div className="flex flex-col items-center space-y-4">
                        <div className="flex items-center justify-center w-16 h-16 rounded-full bg-accent-violet/10">
                          <Upload className="h-8 w-8 text-accent-violet" />
                        </div>
                        <div>
                          <p className={`text-lg font-medium ${colors.text.primary}`}>
                            Click to upload an image
                          </p>
                          <p className={`text-sm ${colors.text.secondary}`}>
                            PNG, JPG, JPEG up to 10MB
                          </p>
                        </div>
                      </div>
          </div>
        ) : (
                    <div className="relative">
                      <img
                        src={imagePreview}
                        alt="Upload preview"
                        className="w-full max-w-md mx-auto rounded-2xl shadow-lg"
                      />
                      <button
                        onClick={removeImage}
                        className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                      >
                        ×
                      </button>
                    </div>
                  )}
                  
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </div>
              </div>
            )}

            {/* Prompt Section */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-accent-blue/10">
                  <Type className="h-5 w-5 text-accent-blue" />
                </div>
                <h2 className={`text-2xl font-bold ${colors.text.primary}`}>
                  {generationMode === 'text-to-video' ? 'Describe Your Video' : 'Describe the Animation'}
                </h2>
              </div>

              <div className="relative">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={
                    generationMode === 'text-to-video' 
                      ? "A cinematic shot of a futuristic city at sunset, with flying cars and neon lights reflecting on glass buildings..."
                      : "Add gentle wind movement, make the clouds drift slowly, create a peaceful atmosphere..."
                  }
                  className={`w-full px-6 py-6 bg-white dark:bg-slate-800/50 border-2 border-gray-200 dark:border-slate-600 rounded-2xl shadow-lg focus:outline-none focus:ring-4 focus:ring-accent-blue/20 focus:border-accent-blue transition-all duration-300 text-gray-900 dark:text-slate-100 hover:border-accent-blue/50 hover:shadow-xl resize-none text-lg leading-relaxed`}
                  rows={5}
                />
                <div className="absolute bottom-4 right-4 flex items-center space-x-4">
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                    prompt.length > 200 ? 'bg-accent-green/10 text-accent-green' : 
                    prompt.length > 100 ? 'bg-accent-blue/10 text-accent-blue' : 
                    'bg-gray-100 text-gray-500'
                  }`}>
                    {prompt.length} characters
                </div>
                </div>
              </div>

              {/* Prompt Suggestions */}
              <div className="flex flex-wrap gap-3">
                {(generationMode === 'text-to-video' ? [
                  'cinematic lighting',
                  '4K quality',
                  'smooth camera movement',
                  'dramatic atmosphere',
                  'highly detailed',
                  'professional grade'
                ] : [
                  'gentle movement',
                  'smooth animation',
                  'natural motion',
                  'subtle effects',
                  'flowing motion',
                  'dynamic elements'
                ]).map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => setPrompt(prev => prev ? `${prev}, ${suggestion}` : suggestion)}
                    className="px-4 py-2 rounded-full border border-accent-blue/30 text-accent-blue hover:bg-accent-blue/10 hover:border-accent-blue/50 transition-all duration-200 hover:scale-105 text-sm font-medium"
                  >
                    + {suggestion}
                  </button>
                ))}
              </div>
            </div>

            {/* Collapsible Settings Section */}
            <div className="space-y-4">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="flex items-center justify-between w-full p-4 rounded-2xl border border-accent-blue/20 bg-accent-blue/5 dark:bg-accent-blue/10 hover:bg-accent-blue/10 dark:hover:bg-accent-blue/20 transition-all duration-300 group"
              >
                <div className="flex items-center space-x-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-accent-blue/10 group-hover:bg-accent-blue/20 transition-colors">
                    <Settings className="h-5 w-5 text-accent-blue" />
                  </div>
                  <div className="text-left">
                    <h3 className={`text-xl font-bold ${colors.text.primary}`}>
                      Video Settings
                    </h3>
                    <p className={`text-sm ${colors.text.secondary}`}>
                      Optimized settings for best quality
                    </p>
                  </div>
                </div>
                <div className={`transform transition-transform duration-300 ${showSettings ? 'rotate-180' : ''}`}>
                  <svg className="h-6 w-6 text-accent-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>
              
              {showSettings && (
                <div className={`p-6 rounded-2xl border border-accent-blue/20 bg-white dark:bg-slate-800/30 backdrop-blur-md animate-slide-down`}>
                  <div className="space-y-6">
                    {/* Optimal Settings Info */}
                    <div className="p-4 rounded-xl bg-accent-blue/10 border border-accent-blue/20">
                      <div className="flex items-start space-x-3">
                        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent-blue/20">
                          <CheckCircle className="h-5 w-5 text-accent-blue" />
                        </div>
                        <div>
                          <h4 className={`font-semibold ${colors.text.primary} mb-1`}>
                            Optimized for Video Generation
                          </h4>
                          <p className={`text-sm ${colors.text.secondary}`}>
                            Resolution: 576×1024 (optimal for video) • Duration: 4 seconds (best quality)
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Format Selection */}
                    <div className="space-y-3">
                      <label className={`block text-sm font-semibold ${colors.text.primary} uppercase tracking-wide`}>
                        AI Model
                      </label>
                      <select
                        value={settings.model}
                        onChange={(e) => setSettings(prev => ({ ...prev, model: e.target.value }))}
                        className={`w-full px-4 py-4 bg-white dark:bg-slate-700/50 border-2 border-gray-200 dark:border-slate-600 rounded-xl focus:outline-none focus:ring-4 focus:ring-accent-blue/20 focus:border-accent-blue transition-all duration-200 text-gray-900 dark:text-slate-100 font-medium`}
                      >
                        <option value="animatediff">AnimateDiff - Perfect for GIFs</option>
                        <option value="kandinsky">Kandinsky - Artistic Images</option>
                      </select>
                    </div>

                    <div className="space-y-3">
                      <label className={`block text-sm font-semibold ${colors.text.primary} uppercase tracking-wide`}>
                        Output Format
                      </label>
                      <select
                        value={settings.format}
                        onChange={(e) => setSettings(prev => ({ ...prev, format: e.target.value }))}
                        className={`w-full px-4 py-4 bg-white dark:bg-slate-700/50 border-2 border-gray-200 dark:border-slate-600 rounded-xl focus:outline-none focus:ring-4 focus:ring-accent-blue/20 focus:border-accent-blue transition-all duration-200 text-gray-900 dark:text-slate-100 font-medium`}
                      >
                        <option value="gif">GIF (AnimateDiff - Recommended)</option>
                        <option value="mp4">MP4 (Universal)</option>
                        <option value="webm">WebM (Web optimized)</option>
                        <option value="mov">MOV (QuickTime)</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            {/* Generate Button */}
            <div className="pt-4">
              <button
                onClick={handleGenerate}
                disabled={!prompt.trim() || (generationMode === 'image-to-video' && !uploadedImage)}
                className="group relative w-full flex items-center justify-center space-x-4 px-10 py-6 bg-gradient-to-r from-accent-blue to-accent-violet text-white rounded-2xl font-bold text-xl hover:from-accent-blue/90 hover:to-accent-violet/90 transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-2xl hover:shadow-accent-blue/25"
              >
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-accent-blue to-accent-violet opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                <Play className="h-7 w-7 group-hover:scale-110 transition-transform duration-300" />
                <span>
                  {generationMode === 'text-to-video' ? 'Generate Video from Text' : 'Animate Image to Video'}
                </span>
                <div className="absolute inset-0 rounded-2xl border-2 border-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </button>
              
              {/* Reset Button */}
              <div className="mt-4 text-center">
                <button
                  onClick={() => {
                    setPrompt('');
                    setUploadedImage(null);
                    setImagePreview(null);
                    setGenerationMode('text-to-video');
                    if (fileInputRef.current) {
                      fileInputRef.current.value = '';
                    }
                  }}
                  className="px-6 py-2 text-accent-blue hover:bg-accent-blue/10 rounded-lg transition-colors text-sm font-medium"
                >
                  Reset Form
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderProgressStep = () => {
    const currentResult = results[0];
    if (!currentResult) return null;

    return (
      <div className="max-w-2xl mx-auto text-center">
        <div className="mb-8">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent mb-2">
            Generating Your Video
          </h1>
          <p className={`text-lg ${colors.text.secondary}`}>
            Please wait while we create your video...
          </p>
        </div>

        <div className={`p-8 rounded-2xl border shadow-xl bg-white dark:bg-gradient-to-br dark:from-dark-card/90 dark:to-dark-bg-secondary/50 backdrop-blur-md`}>
          <div className="space-y-6">
            {/* Prompt Display */}
            <div className="text-left">
              <p className={`text-sm ${colors.text.secondary} mb-2`}>
                {generationMode === 'text-to-video' ? 'Generating from text:' : 'Animating image:'}
              </p>
              <p className={`${colors.text.primary} font-medium text-lg`}>{currentResult.prompt}</p>
              {generationMode === 'image-to-video' && imagePreview && (
                <div className="mt-3">
                  <img
                    src={imagePreview}
                    alt="Source image"
                    className="w-32 h-20 object-cover rounded-lg shadow-md"
                  />
                </div>
              )}
            </div>

            {/* Status */}
            <div className="flex items-center justify-center space-x-3">
              {getStatusIcon(currentResult.status)}
              <span className="text-xl font-medium">{getStatusText(currentResult.status)}</span>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="w-full bg-gray-200 rounded-full h-4">
                <div 
                  className="bg-gradient-to-r from-accent-blue to-accent-violet h-4 rounded-full transition-all duration-500"
                  style={{ width: `${currentResult.progress}%` }}
                ></div>
              </div>
              <p className="text-lg font-medium">{currentResult.progress}% complete</p>
              {currentResult.message && (
                <p className="text-sm text-gray-600 dark:text-gray-400">{currentResult.message}</p>
              )}
            </div>

            {/* Settings Display */}
            <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Resolution:</span>
                  <p className="font-medium">576×1024 (Video Optimal)</p>
                </div>
                <div>
                  <span className="text-gray-500">Duration:</span>
                  <p className="font-medium">4s (Best Quality)</p>
                </div>
                <div>
                  <span className="text-gray-500">Format:</span>
                  <p className="font-medium">{currentResult.settings.format.toUpperCase()}</p>
                </div>
                <div>
                  <span className="text-gray-500">Model:</span>
                  <p className="font-medium">
                    {settings.model === 'animatediff' ? 'AnimateDiff' :
                     settings.model === 'kandinsky' ? 'Kandinsky 2.2' :
                     settings.model}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderResultsStep = () => {
    const currentResult = results[0];
    if (!currentResult) return null;

    return (
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent mb-2">
            Video Generated!
          </h1>
          <p className={`text-lg ${colors.text.secondary}`}>
            Your video is ready for download
          </p>
        </div>

        <div className={`p-8 rounded-2xl border shadow-xl bg-white dark:bg-gradient-to-br dark:from-dark-card/90 dark:to-dark-bg-secondary/50 backdrop-blur-md`}>
          <div className="space-y-6">
            {/* Prompt Display */}
            <div>
              <p className={`text-sm ${colors.text.secondary} mb-2`}>
                {generationMode === 'text-to-video' ? 'Generated from text:' : 'Animated from image:'}
              </p>
              <p className={`${colors.text.primary} font-medium text-lg`}>{currentResult.prompt}</p>
              {generationMode === 'image-to-video' && imagePreview && (
                <div className="mt-3">
                  <p className={`text-sm ${colors.text.secondary} mb-2`}>Source image:</p>
                  <img
                    src={imagePreview}
                    alt="Source image"
                    className="w-32 h-20 object-cover rounded-lg shadow-md"
                  />
                </div>
              )}
            </div>

              {/* Video Preview */}
            {currentResult.status === 'completed' && currentResult.outputFile && (
              <div className="text-center">
                  <video 
                    controls 
                  className="w-full max-w-2xl rounded-lg shadow-lg mx-auto"
                  src={getApiUrl(currentResult.outputFile.replace('../outputs', '/outputs'))}
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
              )}

            {/* Error Display */}
            {currentResult.status === 'failed' && currentResult.error && (
              <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                <p className="text-red-600 dark:text-red-400 text-sm">{currentResult.error}</p>
              </div>
            )}

            {/* Settings Display */}
            <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800">
              <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Resolution:</span>
                  <p className="font-medium">576×1024 (Video Optimal)</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Duration:</span>
                  <p className="font-medium">4s (Best Quality)</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Format:</span>
                  <p className="font-medium">{currentResult.settings.format.toUpperCase()}</p>
                </div>
                <div>
                  <span className="text-gray-500">Model:</span>
                  <p className="font-medium">
                    {settings.model === 'animatediff' ? 'AnimateDiff' :
                     settings.model === 'kandinsky' ? 'Kandinsky 2.2' :
                     settings.model}
                  </p>
                </div>
                </div>
              </div>

              {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {currentResult.status === 'completed' && currentResult.outputFile && (
                  <a
                  href={getApiUrl(currentResult.outputFile.replace('../outputs', '/outputs'))}
                    download
                  className="flex items-center justify-center space-x-2 px-6 py-3 bg-accent-blue text-white rounded-lg hover:bg-accent-blue/90 transition-colors"
                  >
                  <Download className="h-5 w-5" />
                  <span>Download Video</span>
                  </a>
              )}
                  <button
                    onClick={() => {
                      setPrompt('');
                      setUploadedImage(null);
                      setImagePreview(null);
                      setGenerationMode('text-to-video');
                      setCurrentStep('prompt');
                      setCurrentJobId(null);
                      setResults([]);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
                    }}
                    className="flex items-center justify-center space-x-2 px-6 py-3 border border-accent-blue/30 text-accent-blue rounded-lg hover:bg-accent-blue/10 transition-colors"
                  >
                    <RefreshCw className="h-5 w-5" />
                    <span>Generate Another</span>
                  </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto">
        {currentStep === 'prompt' && renderPromptStep()}
        {currentStep === 'progress' && renderProgressStep()}
        {currentStep === 'results' && renderResultsStep()}
      </div>
    </div>
  );
}


