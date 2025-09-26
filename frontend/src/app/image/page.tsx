'use client';

import { useEffect, useState } from 'react';
import { Sparkles, CheckCircle, Loader2, Image as ImageIcon } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
import { getApiUrl } from '@/config';
import { ProgressModal } from '@/components/ProgressModal';

interface Model {
  id: string;
  name: string;
  description: string;
  loaded: boolean;
}

export default function Page() {
  const colors = useThemeColors();
  const [prompt, setPrompt] = useState('');
  const [models, setModels] = useState<{ image_models: Model[] }>({ image_models: [] });
  const [selectedModel, setSelectedModel] = useState('stable-diffusion');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [showProgress, setShowProgress] = useState(false);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const res = await fetch(getApiUrl('/models'));
        const data = await res.json();
        setModels({ image_models: data.image_models || [] });
        if (data.image_models?.length) {
          const saved = localStorage.getItem('default_image_model');
          setSelectedModel(saved && data.image_models.find((m: any)=>m.id===saved) ? saved : data.image_models[0].id);
        }
      } catch (e) {}
    };
    loadModels();
  }, []);

  useEffect(() => {
    if (!currentJobId) return;
    const poll = async () => {
      try {
        const res = await fetch(getApiUrl(`/job/${currentJobId}`));
        if (!res.ok) return;
        const job = await res.json();
        setProgress(job.progress || 0);
        setCurrentStep(job.message || 'Generating image...');
        if (job.status === 'completed' || job.status === 'failed') {
          setIsGenerating(false);
          setTimeout(()=>setShowProgress(false), 1000);
        }
      } catch {}
    };
    // Immediate fetch, then start interval
    poll();
    const t = setInterval(poll, 1000);
    return () => clearInterval(t);
  }, [currentJobId]);

  const onGenerate = async () => {
    if (!prompt.trim() || !selectedModel) return;
    setIsGenerating(true);
    setShowProgress(true);
    setProgress(0);
    setCurrentStep('Queuing job...');
    try {
      const res = await fetch(getApiUrl('/generate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          model_type: 'image',
          model_name: selectedModel,
          output_format: 'png'
        })
      });
      if (!res.ok) throw new Error('Failed to start');
      const data = await res.json();
      setCurrentJobId(data.job_id);
      setCurrentStep('Waiting for worker...');
    } catch (e) {
      setIsGenerating(false);
      setCurrentStep('Failed to start');
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
          <label className="block text-sm font-semibold mb-2">AI Model</label>
          <select
            value={selectedModel}
            onChange={(e)=>{
              setSelectedModel(e.target.value);
              localStorage.setItem('default_image_model', e.target.value);
            }}
            className="w-full px-4 py-3 rounded-xl border bg-white dark:bg-slate-800"
          >
            {models.image_models.map(m => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
        </div>

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
              disabled={isGenerating || !prompt.trim() || !selectedModel}
              className={`inline-flex items-center space-x-2 px-6 py-3 rounded-xl text-white ${isGenerating? 'bg-gray-400' : 'bg-gradient-to-r from-accent-blue to-accent-violet hover:opacity-90'}`}
            >
              {isGenerating ? <Loader2 className="h-5 w-5 animate-spin"/> : <Sparkles className="h-5 w-5"/>}
              <span>{isGenerating ? 'Generating...' : 'Generate Image'}</span>
            </button>
          </div>
        </div>
      </div>

      <ProgressModal
        isOpen={showProgress}
        onClose={()=>setShowProgress(false)}
        title="Generating Image"
        description="Your image is being created"
        progress={progress}
        status={isGenerating ? 'in_progress' : 'completed'}
        currentStep={currentStep}
        type="download"
      />
    </div>
  );
}


