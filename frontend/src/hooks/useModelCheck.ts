import { useState, useEffect } from 'react';
import { getApiUrl } from '@/config';

interface Model {
  id: string;
  name: string;
  description: string;
  max_duration: number;
  resolution?: string;
  sample_rate?: number;
  size_gb?: number;
  loaded: boolean;
}

interface ModelCheckResult {
  models: {
    video_models: Model[];
    audio_models: Model[];
    image_models: Model[];
  };
  isLoading: boolean;
  error: string | null;
  checkModels: () => Promise<void>;
  getMissingModels: (modelType: 'video' | 'audio' | 'image', modelName?: string) => Model[];
}

export function useModelCheck(): ModelCheckResult {
  const [models, setModels] = useState<{
    video_models: Model[];
    audio_models: Model[];
    image_models: Model[];
  }>({ video_models: [], audio_models: [], image_models: [] });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkModels = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(getApiUrl('/models'));
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      
      const data = await response.json();
      setModels(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check models');
      console.error('Error checking models:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const getMissingModels = (modelType: 'video' | 'audio' | 'image', modelName?: string): Model[] => {
    const modelList = models[`${modelType}_models` as keyof typeof models] as Model[];
    
    if (!modelList || modelList.length === 0) {
      return [];
    }

    // If specific model name is provided, check only that model
    if (modelName) {
      const specificModel = modelList.find(model => model.id === modelName);
      if (specificModel && (!specificModel.size_gb || specificModel.size_gb === 0)) {
        return [specificModel];
      }
      return [];
    }

    // Return all models that are not downloaded (size_gb is 0 or undefined)
    return modelList.filter(model => !model.size_gb || model.size_gb === 0);
  };

  useEffect(() => {
    checkModels();
  }, []);

  return {
    models,
    isLoading,
    error,
    checkModels,
    getMissingModels
  };
}
