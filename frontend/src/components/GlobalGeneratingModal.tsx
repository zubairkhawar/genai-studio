'use client';

import { useEffect, useState } from 'react';
import { CheckCircle, AlertCircle, Loader2, Image as ImageIcon, Video, Volume2 } from 'lucide-react';
import { useGenerating } from '@/contexts/GeneratingContext';
import { useThemeColors } from '@/hooks/useThemeColors';

export function GlobalGeneratingModal() {
  const { generatingState, stopGenerating } = useGenerating();
  const colors = useThemeColors();
  const [showSuccess, setShowSuccess] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);

  useEffect(() => {
    if (generatingState.isGenerating && generatingState.currentJobId) {
      const pollJob = async () => {
        try {
          const response = await fetch(`http://localhost:8000/job/${generatingState.currentJobId}`);
          if (response.ok) {
            const job = await response.json();
            
            if (job.status === 'completed' || job.status === 'failed') {
              setIsCompleted(true);
              setShowSuccess(job.status === 'completed');
              
              // Auto-close after 2 seconds
              setTimeout(() => {
                stopGenerating();
                setShowSuccess(false);
                setIsCompleted(false);
              }, 2000);
            }
          }
        } catch (error) {
          console.error('Failed to poll job status:', error);
        }
      };

      const interval = setInterval(pollJob, 1000);
      return () => clearInterval(interval);
    }
  }, [generatingState.isGenerating, generatingState.currentJobId, stopGenerating]);

  if (!generatingState.isGenerating) return null;

  const getIcon = () => {
    if (showSuccess) return <CheckCircle className="h-8 w-8 text-green-500" />;
    if (isCompleted && !showSuccess) return <AlertCircle className="h-8 w-8 text-red-500" />;
    
    // Always show spinning loader when generating with appropriate color
    const colorClass = getStatusColor();
    return <Loader2 className={`h-8 w-8 animate-spin ${colorClass}`} />;
  };

  const getStatusColor = () => {
    if (showSuccess) return 'text-green-500';
    if (isCompleted && !showSuccess) return 'text-red-500';
    
    switch (generatingState.type) {
      case 'image':
        return 'text-accent-blue';
      case 'video':
        return 'text-accent-violet';
      case 'audio':
        return 'text-accent-green';
      default:
        return 'text-accent-blue';
    }
  };

  const getTypeColor = () => {
    switch (generatingState.type) {
      case 'image':
        return 'bg-accent-blue/10';
      case 'video':
        return 'bg-accent-violet/10';
      case 'audio':
        return 'bg-accent-green/10';
      default:
        return 'bg-accent-blue/10';
    }
  };

  return (
    <div className="fixed top-0 left-0 right-0 bottom-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4" style={{ width: '100vw', height: '100vh' }}>
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl max-w-md w-full p-6 animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-xl ${getTypeColor()}`}>
              {getIcon()}
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-slate-100">{generatingState.title}</h3>
              <p className={`text-sm ${colors.text.secondary}`}>{generatingState.description}</p>
            </div>
          </div>
        </div>

        {/* Progress Section */}
        <div className="mb-6">
          {!isCompleted && (
            <div className="flex items-center justify-center mb-4">
              <div className="relative w-24 h-24">
                <div className={`w-24 h-24 border-4 border-gray-200 dark:border-slate-600 rounded-full animate-spin ${
                  generatingState.type === 'image' ? 'border-t-accent-blue' :
                  generatingState.type === 'video' ? 'border-t-accent-violet' :
                  generatingState.type === 'audio' ? 'border-t-accent-green' :
                  'border-t-accent-blue'
                }`}></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className={`w-8 h-8 rounded-full animate-pulse ${
                    generatingState.type === 'image' ? 'bg-accent-blue/20' :
                    generatingState.type === 'video' ? 'bg-accent-violet/20' :
                    generatingState.type === 'audio' ? 'bg-accent-green/20' :
                    'bg-accent-blue/20'
                  }`}></div>
                </div>
              </div>
            </div>
          )}

          {showSuccess && (
            <div className="text-center mb-4">
              <div className="w-16 h-16 mx-auto mb-3 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center">
                <CheckCircle className="h-8 w-8 text-green-500" />
              </div>
              <p className="text-green-600 dark:text-green-400 font-medium">
                {generatingState.type === 'image' ? 'Image Generated!' :
                 generatingState.type === 'video' ? 'Video Generated!' :
                 generatingState.type === 'audio' ? 'Audio Generated!' :
                 'Generation Complete!'}
              </p>
            </div>
          )}

          {isCompleted && !showSuccess && (
            <div className="text-center mb-4">
              <div className="w-16 h-16 mx-auto mb-3 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
                <AlertCircle className="h-8 w-8 text-red-500" />
              </div>
              <p className="text-red-600 dark:text-red-400 font-medium">Generation Failed</p>
            </div>
          )}

          {/* Current Step */}
          {generatingState.currentStep && !isCompleted && (
            <div className="text-center mb-4">
              <p className={`text-sm font-medium ${getStatusColor()}`}>{generatingState.currentStep}</p>
            </div>
          )}

          {/* Progress Bar */}
          {generatingState.progress > 0 && !isCompleted && (
            <div className="w-full bg-gray-200 dark:bg-slate-600 rounded-full h-2 mb-2">
              <div 
                className="bg-gradient-to-r from-accent-blue to-accent-violet h-2 rounded-full transition-all duration-300"
                style={{ width: `${generatingState.progress}%` }}
              ></div>
            </div>
          )}
        </div>

        {/* Note */}
        <div className="text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {!isCompleted ? 
              `Generating your ${generatingState.type}...` : 
              `Your ${generatingState.type} generation is complete!`
            }
          </p>
        </div>
      </div>
    </div>
  );
}
