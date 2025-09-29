'use client';

import { useState, useEffect } from 'react';
import { Volume2, Settings, Loader2, Sparkles } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
import { useGenerating } from '@/contexts/GeneratingContext';
import { useModelCheck } from '@/hooks/useModelCheck';
import ModelDownloadModal from '@/components/ModelDownloadModal';
import { getApiUrl } from '@/config';

export default function Page() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [settings, setSettings] = useState({
    format: 'wav',
    voiceStyle: 'auto',
    voiceId: null as string | null,
    model: 'bark'
  });
  const [showSettings, setShowSettings] = useState(false);
  const [voicePreviews, setVoicePreviews] = useState<{[key: string]: string}>({});
  const [playingPreview, setPlayingPreview] = useState<string | null>(null);
  const [barkVoices, setBarkVoices] = useState<any[]>([]);
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [customVoiceUrl, setCustomVoiceUrl] = useState<string | null>(null);
  const [customVoiceName, setCustomVoiceName] = useState<string>("");
  const [customVoices, setCustomVoices] = useState<any[]>([]);
  const [savedVoices, setSavedVoices] = useState<any[]>([]);
  const colors = useThemeColors();
  const { startGenerating, stopGenerating } = useGenerating();
  
  // Model checking
  const { getMissingModels, checkModels } = useModelCheck();
  const [showDownloadModal, setShowDownloadModal] = useState(false);

  const fetchVoicePreviews = async () => {
    try {
      // Preload preset audio URLs for English speakers directly
      const englishSpeakers = [
        'v2/en_speaker_0', 'v2/en_speaker_1', 'v2/en_speaker_2', 'v2/en_speaker_3', 'v2/en_speaker_4',
        'v2/en_speaker_5', 'v2/en_speaker_6', 'v2/en_speaker_7', 'v2/en_speaker_8', 'v2/en_speaker_9'
      ];
      
      const previewMap: {[key: string]: string} = {};
      englishSpeakers.forEach(voiceId => {
        previewMap[voiceId] = getApiUrl(`/outputs/voice-previews/${voiceId.replace('/', '_')}-preview.mp3`);
      });
      
      setVoicePreviews(previewMap);
    } catch (error) {
      console.error('Error fetching voice previews:', error);
    }
  };

  const fetchBarkVoices = async () => {
    try {
      const response = await fetch(getApiUrl('/bark-voices'));
      if (response.ok) {
        const voices = await response.json();
        setBarkVoices(voices.voices || []);
      }
    } catch (error) {
      console.error('Error fetching Bark voices:', error);
    }
  };

  const fetchCustomVoices = async () => {
    try {
      const response = await fetch(getApiUrl('/custom-voices'));
      if (response.ok) {
        const data = await response.json();
        setCustomVoices(data.voices || []);
        setSavedVoices(data.saved_voices || []);
      }
    } catch (error) {
      console.error('Error fetching custom voices:', error);
    }
  };

  useEffect(() => {
    fetchVoicePreviews();
    fetchBarkVoices();
    fetchCustomVoices();
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    
    // Check if required models are available
    const missingModels = getMissingModels('audio', settings.model);
    if (missingModels.length > 0) {
      setShowDownloadModal(true);
      return;
    }
    
    setIsGenerating(true);
    
    try {
      let voice_id: string | null = settings.voiceId;
      const response = await fetch(getApiUrl('/generate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          model_type: 'audio',
          model_name: settings.model,
          sample_rate: 22050,
          output_format: settings.format,
          voice_style: settings.voiceStyle,
          voice_id
        })
      });
      
      const result = await response.json();
      
      if (result.job_id) {
        // Start global generating modal
        startGenerating(
          result.job_id,
          'audio',
          'Generating Audio',
          'Your audio is being created from text'
        );
      }
    } catch (error) {
      console.error('Generation failed:', error);
      setIsGenerating(false);
      stopGenerating();
    }
  };

  // Recording helpers for XTTS custom voice
  const startRecording = async () => {
    if (recording) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks: BlobPart[] = [];

    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/wav' });
      const formData = new FormData();
      formData.append('audio', blob, 'recording.wav');
      formData.append('voice_name', customVoiceName);

      try {
        const response = await fetch(getApiUrl('/upload-custom-voice'), {
          method: 'POST',
          body: formData
        });
        
        if (response.ok) {
          const result = await response.json();
          setCustomVoiceUrl(result.voice_url);
          fetchCustomVoices(); // Refresh the list
        }
      } catch (error) {
        console.error('Failed to upload custom voice:', error);
      }
      
      stream.getTracks().forEach(track => track.stop());
    };

    recorder.start();
    setMediaRecorder(recorder);
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorder && recording) {
    mediaRecorder.stop();
      setRecording(false);
      setMediaRecorder(null);
    }
  };

  const playVoicePreview = async (voiceId: string) => {
    try {
      setPlayingPreview(voiceId);
      const audio = new Audio(voicePreviews[voiceId]);
      audio.onended = () => setPlayingPreview(null);
      audio.onerror = () => setPlayingPreview(null);
      await audio.play();
    } catch (error) {
      console.error('Failed to play voice preview:', error);
      setPlayingPreview(null);
    }
  };

  const handleVoiceSelect = (voiceId: string) => {
    setSettings(prev => ({ ...prev, voiceId }));
  };

  return (
    <div className="space-y-8">
      <div className="text-center mb-6">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-accent-violet to-accent-blue mb-4 shadow-2xl">
          <Volume2 className="h-8 w-8 text-white" />
        </div>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-violet to-accent-blue bg-clip-text text-transparent">Text-to-Audio Generator</h1>
        <p className={`mt-1 ${colors.text.secondary}`}>Generate high-quality audio from text prompts using AI voices</p>
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
                placeholder="Enter the text you want to convert to speech... You can add emotions like [laughter], [sigh], [gasp], [whisper], [cry], [scream], [sing], [breathing], [music], [♪], [clears throat], [cough], [yawn]"
                className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-accent-violet focus:border-transparent transition-all duration-200"
              />
              
              {/* Emotion Guide */}
              <div className="mt-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
                <div className="flex items-start space-x-2">
                  <div className="flex-shrink-0 w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center mt-0.5">
                    <span className="text-white text-xs font-bold">i</span>
                </div>
                  <div>
                    <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">
                      💡 Add Emotions to Your Text
                    </h4>
                    <p className="text-sm text-blue-800 dark:text-blue-200 mb-3">
                      You can enhance your audio with emotions and sounds by adding these tags to your text:
                    </p>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[laughter]</code>
                        <span className="text-blue-700 dark:text-blue-300">laughing</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[sigh]</code>
                        <span className="text-blue-700 dark:text-blue-300">sighing</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[gasp]</code>
                        <span className="text-blue-700 dark:text-blue-300">gasping</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[whisper]</code>
                        <span className="text-blue-700 dark:text-blue-300">whispering</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[cry]</code>
                        <span className="text-blue-700 dark:text-blue-300">crying</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[scream]</code>
                        <span className="text-blue-700 dark:text-blue-300">screaming</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[sing]</code>
                        <span className="text-blue-700 dark:text-blue-300">singing</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[clears throat]</code>
                        <span className="text-blue-700 dark:text-blue-300">clearing throat</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[cough]</code>
                        <span className="text-blue-700 dark:text-blue-300">coughing</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[yawn]</code>
                        <span className="text-blue-700 dark:text-blue-300">yawning</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[breathing]</code>
                        <span className="text-blue-700 dark:text-blue-300">breathing</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">[music]</code>
                        <span className="text-blue-700 dark:text-blue-300">music</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">♪</code>
                        <span className="text-blue-700 dark:text-blue-300">song notes</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">...</code>
                        <span className="text-blue-700 dark:text-blue-300">pause</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded text-blue-800 dark:text-blue-200">—</code>
                        <span className="text-blue-700 dark:text-blue-300">hesitation</span>
                      </div>
                    </div>
                    <p className="text-xs text-blue-700 dark:text-blue-300 mt-3">
                      <strong>Example:</strong> "Hello there! [laughter] I'm so excited to see you. [sigh] It's been a long day. [whisper] Let me tell you a secret... [gasp] Oh my!"
                    </p>
                    <div className="mt-3 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                      <p className="text-xs text-yellow-800 dark:text-yellow-200">
                        <strong>💡 Note:</strong> These emotion tags work best with <strong>Speaker 6</strong> (v2/en_speaker_6). For optimal results, select Speaker 6 when using emotion tags.
                      </p>
                    </div>
            </div>
                </div>
              </div>
            </div>

            {/* Voice Selection */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="block text-sm font-semibold text-gray-900 dark:text-white">
                  Voice Selection
                </label>
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="flex items-center space-x-2 px-3 py-1 text-sm text-accent-violet hover:bg-accent-violet/10 rounded-lg transition-colors"
                >
                  <Settings className="h-4 w-4" />
                  <span>Settings</span>
                </button>
                  </div>
              
              {/* Voice Options */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-64 overflow-y-auto">
                {barkVoices.map((voice) => (
                  <button
                              key={voice.id}
                    onClick={() => handleVoiceSelect(voice.id)}
                    className={`p-3 rounded-lg border text-left transition-all duration-200 ${
                                settings.voiceId === voice.id
                        ? 'border-accent-violet bg-accent-violet/10 text-accent-violet'
                        : 'border-gray-200 dark:border-gray-600 hover:border-accent-violet/50 hover:bg-accent-violet/5'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-sm text-gray-900 dark:text-white">
                          {voice.name}
                          </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {voice.language}
                                </div>
                              </div>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                    playVoicePreview(voice.id);
                        }}
                        className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                        disabled={playingPreview === voice.id}
                              >
                                {playingPreview === voice.id ? (
                          <Loader2 className="h-4 w-4 animate-spin text-accent-violet" />
                                ) : (
                          <Volume2 className="h-4 w-4 text-gray-500" />
                                )}
                              </button>
                            </div>
                  </button>
                ))}
                      </div>
                            </div>

            {/* Settings Panel */}
            {showSettings && (
              <div className="p-4 bg-gray-50 dark:bg-slate-700/50 rounded-xl border">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Generation Settings</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Output Format
                      </label>
                      <select
                        value={settings.format}
                        onChange={(e) => setSettings(prev => ({ ...prev, format: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-slate-800 text-gray-900 dark:text-white"
                      >
                        <option value="wav">WAV (Uncompressed - Recommended)</option>
                        <option value="mp3">MP3 (Compressed)</option>
                      </select>
                    </div>
                  </div>
            </div>
            )}
            
            {/* Generate Button */}
            <div className="pt-4">
              <button
                onClick={handleGenerate}
                disabled={!prompt.trim() || !settings.voiceId}
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

      {/* Model Download Modal */}
      <ModelDownloadModal
        isOpen={showDownloadModal}
        onClose={() => setShowDownloadModal(false)}
        missingModels={getMissingModels('audio', settings.model)}
        modelType="audio"
        onModelsDownloaded={() => {
          checkModels();
          setShowDownloadModal(false);
        }}
      />
    </div>
  );
}