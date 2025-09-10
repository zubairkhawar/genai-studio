'use client';

import { useState, useEffect } from 'react';
import { Volume2, Download, RefreshCw, Settings, CheckCircle, Clock, Loader2, Play, Pause } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';

interface GenerationResult {
  id: string;
  prompt: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  outputFile?: string;
  error?: string;
  createdAt: string;
  settings: {
    format: string;
    voiceStyle: string;
  };
}

export default function Page() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [results, setResults] = useState<GenerationResult[]>([]);
  const [currentStep, setCurrentStep] = useState<'prompt' | 'progress' | 'results'>('prompt');
  const [settings, setSettings] = useState({
    format: 'wav',
    voiceStyle: 'auto'
  });
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [playingAudio, setPlayingAudio] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [voicePreviews, setVoicePreviews] = useState<{[key: string]: string}>({});
  const [playingPreview, setPlayingPreview] = useState<string | null>(null);
  const colors = useThemeColors();

  const fetchJobs = async () => {
    if (currentStep !== 'progress' && currentStep !== 'results') return;
    
    try {
      const response = await fetch('http://localhost:8000/jobs');
      const jobs = await response.json();
      
      // Filter for audio jobs and convert to our format
      const audioJobs = jobs
        .filter((job: any) => job.model_type === 'audio')
        .map((job: any) => ({
          id: job.job_id,
          prompt: job.prompt,
          status: job.status,
          progress: job.progress || 0,
          outputFile: job.output_file,
          error: job.error,
          createdAt: job.created_at,
          settings: {
            voice: 'default', // Default, could be enhanced
            sampleRate: job.sample_rate || 22050,
            format: job.output_format || 'wav'
          }
        }))
        .sort((a: any, b: any) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
      
      // Find current job or most recent
      const currentJob = currentJobId 
        ? audioJobs.find((job: any) => job.id === currentJobId)
        : audioJobs[0];
      
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

  const fetchVoicePreviews = async () => {
    try {
      const response = await fetch('http://localhost:8000/voice-previews');
      const data = await response.json();
      
      const previewMap: {[key: string]: string} = {};
      data.previews.forEach((preview: any) => {
        previewMap[preview.voice_style] = `http://localhost:8000${preview.url}`;
      });
      
      setVoicePreviews(previewMap);
    } catch (error) {
      console.error('Error fetching voice previews:', error);
    }
  };

  const playVoicePreview = async (voiceStyle: string) => {
    if (playingPreview === voiceStyle) {
      setPlayingPreview(null);
      return;
    }

    setPlayingPreview(voiceStyle);
    
    const audioUrl = voicePreviews[voiceStyle];
    if (audioUrl) {
      const audio = new Audio(audioUrl);
      audio.onended = () => setPlayingPreview(null);
      audio.onerror = () => setPlayingPreview(null);
      await audio.play();
    }
  };

  useEffect(() => {
    if (currentStep === 'progress' || currentStep === 'results') {
    fetchJobs();
    const interval = setInterval(fetchJobs, 2000);
    return () => clearInterval(interval);
    }
  }, [currentStep, currentJobId]);

  useEffect(() => {
    fetchVoicePreviews();
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    
    setIsGenerating(true);
    setCurrentStep('progress');
    
    // Clear previous results when starting new generation
    setResults([]);
    
    try {
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          model_type: 'audio',
          model_name: 'bark',
          sample_rate: 22050, // Optimal for Bark
          output_format: settings.format,
          voice_style: settings.voiceStyle
        })
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

  const toggleAudio = (audioId: string, audioUrl: string) => {
    if (playingAudio === audioId) {
      setPlayingAudio(null);
      // Stop audio
      const audio = document.getElementById(`audio-${audioId}`) as HTMLAudioElement;
      if (audio) audio.pause();
    } else {
      setPlayingAudio(audioId);
      // Play audio
      const audio = document.getElementById(`audio-${audioId}`) as HTMLAudioElement;
      if (audio) {
        audio.play();
        audio.onended = () => setPlayingAudio(null);
      }
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
      { key: 'prompt', label: 'Prompt & Settings', icon: Volume2 },
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
                    ? 'border-accent-violet bg-accent-violet text-white' 
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
                  isActive ? 'text-accent-violet' : isCompleted ? 'text-green-600' : 'text-gray-400'
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
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-r from-accent-violet to-accent-blue mb-6 shadow-2xl">
          <Volume2 className="h-10 w-10 text-white" />
        </div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-accent-violet to-accent-blue bg-clip-text text-transparent mb-4">
            Audio Generator
          </h1>
        <p className={`text-xl ${colors.text.secondary} max-w-2xl mx-auto leading-relaxed`}>
          Bring your words to life with AI-powered voice synthesis. Create natural-sounding speech from any text.
        </p>
      </div>

      {/* Main Content Card */}
      <div className={`relative overflow-hidden rounded-3xl border shadow-2xl bg-white dark:bg-gradient-to-br dark:from-slate-900/90 dark:to-slate-800/50 backdrop-blur-md dark:border-slate-700/50`}>
        {/* Decorative Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-accent-violet/5 via-transparent to-accent-blue/5 dark:from-accent-violet/10 dark:via-transparent dark:to-accent-blue/10"></div>
        <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-bl from-accent-violet/10 to-transparent dark:from-accent-violet/20 rounded-full -translate-y-32 translate-x-32"></div>
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-gradient-to-tr from-accent-blue/10 to-transparent dark:from-accent-blue/20 rounded-full translate-y-24 -translate-x-24"></div>
        
        <div className="relative p-10">
          <div className="space-y-8">
            {/* Prompt Section */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-accent-violet/10">
                  <Volume2 className="h-5 w-5 text-accent-violet" />
          </div>
                <h2 className={`text-2xl font-bold ${colors.text.primary}`}>
                  Enter Your Text
                </h2>
              </div>

              <div className="relative">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Hello! I'm excited to demonstrate the capabilities of this AI voice synthesis system. The quality and naturalness of the generated speech is truly remarkable..."
                  className={`w-full px-6 py-6 bg-white dark:bg-slate-800/50 border-2 border-gray-200 dark:border-slate-600 rounded-2xl shadow-lg focus:outline-none focus:ring-4 focus:ring-accent-violet/20 focus:border-accent-violet transition-all duration-300 text-gray-900 dark:text-slate-100 hover:border-accent-violet/50 hover:shadow-xl resize-none text-lg leading-relaxed`}
                  rows={5}
                />
                <div className="absolute bottom-4 right-4 flex items-center space-x-4">
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                    prompt.length > 200 ? 'bg-accent-green/10 text-accent-green' : 
                    prompt.length > 100 ? 'bg-accent-violet/10 text-accent-violet' : 
                    'bg-gray-100 text-gray-500'
                  }`}>
                    {prompt.length} characters
                </div>
                </div>
              </div>

              {/* Text Suggestions */}
              <div className="flex flex-wrap gap-3">
                {[
                  'natural speech',
                  'clear pronunciation',
                  'emotional tone',
                  'professional voice',
                  'conversational style',
                  'engaging delivery'
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => setPrompt(prev => prev ? `${prev}, ${suggestion}` : suggestion)}
                    className="px-4 py-2 rounded-full border border-accent-violet/30 text-accent-violet hover:bg-accent-violet/10 hover:border-accent-violet/50 transition-all duration-200 hover:scale-105 text-sm font-medium"
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
                className="flex items-center justify-between w-full p-4 rounded-2xl border border-accent-violet/20 bg-accent-violet/5 dark:bg-accent-violet/10 hover:bg-accent-violet/10 dark:hover:bg-accent-violet/20 transition-all duration-300 group"
              >
                <div className="flex items-center space-x-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-accent-violet/10 group-hover:bg-accent-violet/20 transition-colors">
                    <Settings className="h-5 w-5 text-accent-violet" />
                  </div>
                  <div className="text-left">
                    <h3 className={`text-xl font-bold ${colors.text.primary}`}>
                      Audio Settings
                    </h3>
                    <p className={`text-sm ${colors.text.secondary}`}>
                      Optimized settings for best quality
                    </p>
                  </div>
                </div>
                <div className={`transform transition-transform duration-300 ${showSettings ? 'rotate-180' : ''}`}>
                  <svg className="h-6 w-6 text-accent-violet" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>
              
              {showSettings && (
                <div className={`p-6 rounded-2xl border border-accent-violet/20 bg-white dark:bg-slate-800/30 backdrop-blur-md animate-slide-down`}>
                  <div className="space-y-6">
                    {/* Optimal Settings Info */}
                    <div className="p-4 rounded-xl bg-accent-violet/10 border border-accent-violet/20">
                      <div className="flex items-start space-x-3">
                        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent-violet/20">
                          <CheckCircle className="h-5 w-5 text-accent-violet" />
                        </div>
                        <div>
                          <h4 className={`font-semibold ${colors.text.primary} mb-1`}>
                            Optimized for Bark Model
                          </h4>
                          <p className={`text-sm ${colors.text.secondary}`}>
                            Sample Rate: 22,050 Hz (Bark optimal) • Quality: High
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Voice Style Selection */}
                    <div className="space-y-3">
                      <label className={`block text-sm font-semibold ${colors.text.primary} uppercase tracking-wide`}>
                        Voice Style
                      </label>
                      
                      {/* Voice Style Options */}
                      <div className="space-y-2">
                        {[
                          { value: 'auto', label: 'Auto (Recommended)', description: 'Smart voice detection' },
                          { value: 'professional', label: 'Professional', description: 'Business, formal tone' },
                          { value: 'casual', label: 'Casual', description: 'Friendly, conversational' },
                          { value: 'storytelling', label: 'Storytelling', description: 'Narrative, engaging' },
                          { value: 'character', label: 'Character', description: 'Distinctive, animated' },
                          { value: 'male', label: 'Male Voice', description: 'Deep, masculine tone' },
                          { value: 'female', label: 'Female Voice', description: 'Clear, feminine tone' },
                          { value: 'child', label: 'Child Voice', description: 'Young, energetic' },
                          { value: 'narrator', label: 'Narrator', description: 'Clear, authoritative' }
                        ].map((option) => (
                          <div
                            key={option.value}
                            className={`flex items-center justify-between p-3 rounded-lg border-2 transition-all duration-200 cursor-pointer ${
                              settings.voiceStyle === option.value
                                ? 'border-accent-violet bg-accent-violet/10'
                                : 'border-gray-200 dark:border-slate-600 bg-white dark:bg-slate-700/50 hover:border-accent-violet/50'
                            }`}
                            onClick={() => setSettings(prev => ({ ...prev, voiceStyle: option.value }))}
                          >
                            <div className="flex-1">
                              <div className="flex items-center space-x-2">
                                <div className={`w-4 h-4 rounded-full border-2 ${
                                  settings.voiceStyle === option.value
                                    ? 'border-accent-violet bg-accent-violet'
                                    : 'border-gray-300 dark:border-slate-500'
                                }`}>
                                  {settings.voiceStyle === option.value && (
                                    <div className="w-2 h-2 bg-white rounded-full m-0.5"></div>
                                  )}
                                </div>
                                <div>
                                  <p className={`font-medium ${colors.text.primary}`}>
                                    {option.label}
                                  </p>
                                  <p className={`text-xs ${colors.text.secondary}`}>
                                    {option.description}
                                  </p>
                                </div>
                              </div>
                            </div>
                            
                            {/* Play Button */}
                            {option.value !== 'auto' && voicePreviews[option.value] && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  playVoicePreview(option.value);
                                }}
                                className={`ml-3 p-2 rounded-lg transition-all duration-200 ${
                                  playingPreview === option.value
                                    ? 'bg-accent-violet text-white'
                                    : 'bg-gray-100 dark:bg-slate-600 text-gray-600 dark:text-slate-300 hover:bg-accent-violet hover:text-white'
                                }`}
                                title={`Preview ${option.label} voice`}
                              >
                                {playingPreview === option.value ? (
                                  <Pause className="h-4 w-4" />
                                ) : (
                                  <Play className="h-4 w-4" />
                                )}
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                      
                      <p className={`text-xs ${colors.text.secondary} mt-2`}>
                        {settings.voiceStyle === 'auto' 
                          ? 'Bark will automatically select the best voice based on your text content'
                          : `Selected voice style will guide Bark's voice selection for your text. Click play to preview.`
                        }
                      </p>
                    </div>

                    {/* Format Selection */}
                    <div className="space-y-3">
                      <label className={`block text-sm font-semibold ${colors.text.primary} uppercase tracking-wide`}>
                        Output Format
                      </label>
                      <select
                        value={settings.format}
                        onChange={(e) => setSettings(prev => ({ ...prev, format: e.target.value }))}
                        className={`w-full px-4 py-4 bg-white dark:bg-slate-700/50 border-2 border-gray-200 dark:border-slate-600 rounded-xl focus:outline-none focus:ring-4 focus:ring-accent-violet/20 focus:border-accent-violet transition-all duration-200 text-gray-900 dark:text-slate-100 font-medium`}
                      >
                        <option value="wav">WAV (Uncompressed - Recommended)</option>
                        <option value="mp3">MP3 (Compressed)</option>
                        <option value="flac">FLAC (Lossless)</option>
                        <option value="ogg">OGG (Open source)</option>
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
                disabled={!prompt.trim()}
                className="group relative w-full flex items-center justify-center space-x-4 px-10 py-6 bg-gradient-to-r from-accent-violet to-accent-blue text-white rounded-2xl font-bold text-xl hover:from-accent-violet/90 hover:to-accent-blue/90 transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-2xl hover:shadow-accent-violet/25"
              >
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-accent-violet to-accent-blue opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                <Volume2 className="h-7 w-7 group-hover:scale-110 transition-transform duration-300" />
                <span>Generate Audio</span>
                <div className="absolute inset-0 rounded-2xl border-2 border-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </button>
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
          <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-violet to-accent-blue bg-clip-text text-transparent mb-2">
            Generating Your Audio
          </h1>
          <p className={`text-lg ${colors.text.secondary}`}>
            Please wait while we create your audio...
          </p>
        </div>

        <div className={`p-8 rounded-2xl border shadow-xl bg-white dark:bg-gradient-to-br dark:from-dark-card/90 dark:to-dark-bg-secondary/50 backdrop-blur-md`}>
          <div className="space-y-6">
            {/* Prompt Display */}
            <div className="text-left">
              <p className={`text-sm ${colors.text.secondary} mb-2`}>Generating:</p>
              <p className={`${colors.text.primary} font-medium text-lg`}>{currentResult.prompt}</p>
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
                  className="bg-gradient-to-r from-accent-violet to-accent-blue h-4 rounded-full transition-all duration-500"
                  style={{ width: `${currentResult.progress}%` }}
                ></div>
              </div>
              <p className="text-lg font-medium">{currentResult.progress}% complete</p>
            </div>

            {/* Settings Display */}
            <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Sample Rate:</span>
                  <p className="font-medium">22,050 Hz (Bark Optimal)</p>
                </div>
                <div>
                  <span className="text-gray-500">Voice Style:</span>
                  <p className="font-medium">
                    {currentResult.settings.voiceStyle === 'auto' ? 'Auto-detected' : currentResult.settings.voiceStyle}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500">Format:</span>
                  <p className="font-medium">{currentResult.settings.format.toUpperCase()}</p>
                </div>
                <div>
                  <span className="text-gray-500">Model:</span>
                  <p className="font-medium">Bark TTS</p>
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
          <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-violet to-accent-blue bg-clip-text text-transparent mb-2">
            Audio Generated!
          </h1>
          <p className={`text-lg ${colors.text.secondary}`}>
            Your audio is ready for download
          </p>
        </div>

        <div className={`p-8 rounded-2xl border shadow-xl bg-white dark:bg-gradient-to-br dark:from-dark-card/90 dark:to-dark-bg-secondary/50 backdrop-blur-md`}>
          <div className="space-y-6">
            {/* Prompt Display */}
            <div>
              <p className={`text-sm ${colors.text.secondary} mb-2`}>Generated:</p>
              <p className={`${colors.text.primary} font-medium text-lg`}>{currentResult.prompt}</p>
            </div>

              {/* Audio Preview */}
            {currentResult.status === 'completed' && currentResult.outputFile && (
              <div className="text-center">
                <div className="flex items-center justify-center space-x-4 p-6 rounded-lg bg-gray-50 dark:bg-gray-800 max-w-md mx-auto">
                    <button
                    onClick={() => toggleAudio(currentResult.id, `http://localhost:8000${currentResult.outputFile!.replace('../outputs', '/outputs')}`)}
                    className="flex items-center justify-center w-16 h-16 rounded-full bg-accent-violet text-white hover:bg-accent-violet/90 transition-colors"
                    >
                    {playingAudio === currentResult.id ? (
                      <Pause className="h-8 w-8" />
                      ) : (
                      <Play className="h-8 w-8 ml-1" />
                      )}
                    </button>
                    <div className="flex-1">
                      <div className="w-full bg-gray-300 rounded-full h-2">
                        <div className="bg-accent-violet h-2 rounded-full" style={{ width: '0%' }}></div>
                      </div>
                    <p className="text-sm text-gray-500 mt-2">Click to play audio preview</p>
                    </div>
                  </div>
                  <audio
                  id={`audio-${currentResult.id}`}
                  src={`http://localhost:8000${currentResult.outputFile.replace('../outputs', '/outputs')}`}
                    preload="metadata"
                  />
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
                  <span className="text-gray-500">Sample Rate:</span>
                  <p className="font-medium">22,050 Hz (Bark Optimal)</p>
                </div>
                <div>
                  <span className="text-gray-500">Voice Style:</span>
                  <p className="font-medium">
                    {currentResult.settings.voiceStyle === 'auto' ? 'Auto-detected' : currentResult.settings.voiceStyle}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500">Format:</span>
                  <p className="font-medium">{currentResult.settings.format.toUpperCase()}</p>
                </div>
                <div>
                  <span className="text-gray-500">Model:</span>
                  <p className="font-medium">Bark TTS</p>
                </div>
              </div>
            </div>

              {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {currentResult.status === 'completed' && currentResult.outputFile && (
                  <a
                  href={`http://localhost:8000${currentResult.outputFile.replace('../outputs', '/outputs')}`}
                    download
                  className="flex items-center justify-center space-x-2 px-6 py-3 bg-accent-violet text-white rounded-lg hover:bg-accent-violet/90 transition-colors"
                  >
                  <Download className="h-5 w-5" />
                  <span>Download Audio</span>
                  </a>
              )}
                  <button
                    onClick={() => {
                  setPrompt('');
                  setCurrentStep('prompt');
                  setCurrentJobId(null);
                  setResults([]);
                }}
                className="flex items-center justify-center space-x-2 px-6 py-3 border border-accent-violet/30 text-accent-violet rounded-lg hover:bg-accent-violet/10 transition-colors"
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
      {renderStepIndicator()}
      
      <div className="flex-1 overflow-y-auto">
        {currentStep === 'prompt' && renderPromptStep()}
        {currentStep === 'progress' && renderProgressStep()}
        {currentStep === 'results' && renderResultsStep()}
      </div>
    </div>
  );
}


