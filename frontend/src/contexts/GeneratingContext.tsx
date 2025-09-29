'use client';

import { createContext, useContext, useState, ReactNode } from 'react';

interface GeneratingState {
  isGenerating: boolean;
  currentJobId: string | null;
  progress: number;
  currentStep: string;
  type: 'image' | 'video' | 'audio' | null;
  title: string;
  description: string;
}

interface GeneratingContextType {
  generatingState: GeneratingState;
  startGenerating: (jobId: string, type: 'image' | 'video' | 'audio', title: string, description: string) => void;
  updateProgress: (progress: number, step: string) => void;
  stopGenerating: () => void;
}

const GeneratingContext = createContext<GeneratingContextType | undefined>(undefined);

export function GeneratingProvider({ children }: { children: ReactNode }) {
  const [generatingState, setGeneratingState] = useState<GeneratingState>({
    isGenerating: false,
    currentJobId: null,
    progress: 0,
    currentStep: '',
    type: null,
    title: '',
    description: ''
  });

  const startGenerating = (jobId: string, type: 'image' | 'video' | 'audio', title: string, description: string) => {
    setGeneratingState({
      isGenerating: true,
      currentJobId: jobId,
      progress: 0,
      currentStep: 'Generating...',
      type,
      title,
      description
    });
  };

  const updateProgress = (progress: number, step: string) => {
    setGeneratingState(prev => ({
      ...prev,
      progress,
      currentStep: step
    }));
  };

  const stopGenerating = () => {
    setGeneratingState({
      isGenerating: false,
      currentJobId: null,
      progress: 0,
      currentStep: '',
      type: null,
      title: '',
      description: ''
    });
  };

  return (
    <GeneratingContext.Provider value={{
      generatingState,
      startGenerating,
      updateProgress,
      stopGenerating
    }}>
      {children}
    </GeneratingContext.Provider>
  );
}

export function useGenerating() {
  const context = useContext(GeneratingContext);
  if (context === undefined) {
    throw new Error('useGenerating must be used within a GeneratingProvider');
  }
  return context;
}
