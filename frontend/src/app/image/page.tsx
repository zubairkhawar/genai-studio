'use client';

import { useEffect, useState } from 'react';
import { Sparkles, CheckCircle, Loader2, Image as ImageIcon, Download, RefreshCw } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
import { useGenerating } from '@/contexts/GeneratingContext';
import { useModelCheck } from '@/hooks/useModelCheck';
import ModelDownloadModal from '@/components/ModelDownloadModal';
import { getApiUrl } from '@/config';


export default function Page() {
  const colors = useThemeColors();
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const { startGenerating, updateProgress, stopGenerating } = useGenerating();
  
  // Model checking
  const { getMissingModels, checkModels } = useModelCheck();
  const [showDownloadModal, setShowDownloadModal] = useState(false);


  useEffect(() => {
    if (!currentJobId) return;
    const poll = async () => {
      try {
        const res = await fetch(getApiUrl(`/job/${currentJobId}`));
        if (!res.ok) return;
        const job = await res.json();
        setProgress(job.progress || 0);
        setCurrentStep(job.message || 'Generating image...');
        
        // Update global progress
        updateProgress(job.progress || 0, job.message || 'Generating image...');
        
        if (job.status === 'completed' || job.status === 'failed') {
          setIsGenerating(false);
          
          if (job.status === 'completed' && job.output_file) {
            // Extract filename from output_file path
            const filename = job.output_file.split('/').pop();
            if (filename) {
              setGeneratedImage(`http://localhost:8000/outputs/image/${filename}`);
              setShowResults(true);
            }
          }
        }
      } catch {}
    };
    // Immediate fetch, then start interval
    poll();
    const t = setInterval(poll, 1000);
    return () => clearInterval(t);
  }, [currentJobId]);

  const onGenerate = async () => {
    if (!prompt.trim()) return;
    
    // Check if required models are available
    const missingModels = getMissingModels('image', 'stable-diffusion');
    if (missingModels.length > 0) {
      setShowDownloadModal(true);
      return;
    }
    
    setIsGenerating(true);
    setProgress(0);
    setCurrentStep('Queuing job...');
    try {
      const res = await fetch(getApiUrl('/generate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          model_type: 'image',
          model_name: 'stable-diffusion',
          output_format: 'png'
        })
      });
      if (!res.ok) throw new Error('Failed to start');
      const data = await res.json();
      setCurrentJobId(data.job_id);
      setCurrentStep('Waiting for worker...');
      
      // Start global generating modal
      startGenerating(
        data.job_id,
        'image',
        'Generating Image',
        'Your image is being created from text'
      );
    } catch (e) {
      setIsGenerating(false);
      setCurrentStep('Failed to start');
      stopGenerating();
    }
  };

  const generateAnother = () => {
    setShowResults(false);
    setGeneratedImage(null);
    setPrompt('');
    setCurrentJobId(null);
    setProgress(0);
    setCurrentStep('');
  };

  const downloadImage = () => {
    if (generatedImage) {
      const link = document.createElement('a');
      link.href = generatedImage;
      link.download = `generated_image_${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="space-y-8">
      <div className="text-center mb-6">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-accent-blue to-accent-violet mb-4 shadow-2xl">
          <ImageIcon className="h-8 w-8 text-white" />
        </div>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent">Image Generator</h1>
        <p className={`mt-1 ${colors.text.secondary}`}>Generate high-quality images from text prompts</p>
      </div>

      <div className="max-w-3xl mx-auto space-y-6">
        <div className="p-6 rounded-2xl border bg-white dark:bg-slate-800/50">
          <label className="block text-sm font-semibold mb-2">Prompt</label>
          <textarea
            value={prompt}
            onChange={(e)=>setPrompt(e.target.value)}
            rows={4}
            placeholder="A dreamy landscape with floating islands, volumetric light, high detail"
            className="w-full px-4 py-3 rounded-xl border bg-white dark:bg-slate-800"
          />
          <div className="mt-4">
            <button
              onClick={onGenerate}
              disabled={isGenerating || !prompt.trim()}
              className={`inline-flex items-center space-x-2 px-6 py-3 rounded-xl text-white ${isGenerating? 'bg-gray-400' : 'bg-gradient-to-r from-accent-blue to-accent-violet hover:opacity-90'}`}
            >
              {isGenerating ? <Loader2 className="h-5 w-5 animate-spin"/> : <Sparkles className="h-5 w-5"/>}
              <span>{isGenerating ? 'Generating...' : 'Generate Image'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {showResults && generatedImage && (
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-6">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-green-500 to-emerald-500 mb-4 shadow-2xl">
              <CheckCircle className="h-8 w-8 text-white" />
            </div>
            <h2 className="text-2xl font-bold bg-gradient-to-r from-green-500 to-emerald-500 bg-clip-text text-transparent">Image Generated Successfully!</h2>
            <p className={`mt-1 ${colors.text.secondary}`}>Your image has been created from your prompt</p>
          </div>

          <div className="bg-white dark:bg-slate-800/50 rounded-2xl border p-6">
            {/* Generated Image Display */}
            <div className="text-center mb-6">
              <div className="inline-block p-4 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 dark:from-blue-900/20 dark:via-purple-900/20 dark:to-pink-900/20 rounded-2xl border border-blue-200 dark:border-blue-800">
                <img
                  src={generatedImage}
                  alt="Generated image"
                  className="max-w-full max-h-96 rounded-xl shadow-lg"
                  style={{ maxWidth: '512px', maxHeight: '512px' }}
                />
              </div>
            </div>

            {/* Prompt Display */}
            <div className="mb-6">
              <div className="p-4 bg-gray-50 dark:bg-slate-700/50 rounded-xl border">
                <div className="flex items-center space-x-2 mb-2">
                  <ImageIcon className="h-5 w-5 text-blue-500" />
                  <span className="font-semibold text-gray-900 dark:text-white">Prompt</span>
                </div>
                <p className="text-gray-700 dark:text-gray-300 italic">"{prompt}"</p>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={downloadImage}
                className="group relative flex items-center justify-center space-x-3 px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-purple-600 transition-all duration-300 hover:scale-105 shadow-lg hover:shadow-blue-500/25"
              >
                <Download className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Download Image</span>
              </button>
              
              <button
                onClick={generateAnother}
                className="group relative flex items-center justify-center space-x-3 px-8 py-4 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-xl font-semibold hover:from-gray-700 hover:to-gray-800 transition-all duration-300 hover:scale-105 shadow-lg hover:shadow-gray-500/25"
              >
                <RefreshCw className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Generate Another</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Model Download Modal */}
      <ModelDownloadModal
        isOpen={showDownloadModal}
        onClose={() => setShowDownloadModal(false)}
        missingModels={getMissingModels('image', 'stable-diffusion')}
        modelType="image"
        onModelsDownloaded={() => {
          checkModels();
          setShowDownloadModal(false);
        }}
      />

    </div>
  );
}


