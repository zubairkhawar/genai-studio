'use client';

import { useState } from 'react';
import { Cpu, Zap, ChevronDown, Monitor, HardDrive, Activity } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';

interface GPUInfo {
  type: string;
  device: string;
  cuda_available: boolean;
  rocm_available: boolean;
  memory_gb: number;
  name: string;
}

interface GPUInfoProps {
  gpuInfo: GPUInfo | null;
}

export function GPUInfo({ gpuInfo }: GPUInfoProps) {
  const [isOpen, setIsOpen] = useState(false);
  const colors = useThemeColors();

  if (!gpuInfo) {
    return (
      <div className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800">
        <Cpu className="h-4 w-4 text-gray-500" />
        <span className="text-sm text-gray-500">Loading...</span>
      </div>
    );
  }

  const getGPUIcon = () => {
    if (gpuInfo.cuda_available) {
      return <Zap className="h-4 w-4 text-green-500" />;
    } else if (gpuInfo.rocm_available) {
      return <Zap className="h-4 w-4 text-purple-500" />;
    } else {
      return <Cpu className="h-4 w-4 text-gray-500" />;
    }
  };

  const getGPUType = () => {
    if (gpuInfo.cuda_available) return 'NVIDIA CUDA';
    if (gpuInfo.rocm_available) return 'AMD ROCm';
    if (gpuInfo.type === 'mps') return 'Apple Silicon';
    return 'CPU';
  };

  const getStatusColor = () => {
    if (gpuInfo.cuda_available || gpuInfo.rocm_available) return 'text-green-500';
    if (gpuInfo.type === 'mps') return 'text-blue-500';
    return 'text-gray-500';
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center space-x-2 px-3 py-2 rounded-lg bg-gray-50 dark:bg-slate-700/50 hover:bg-gray-100 dark:hover:bg-slate-600/50 transition-all duration-200 group`}
        title="Click to view system details"
      >
        {getGPUIcon()}
        <div className="flex items-center space-x-1">
          <span className={`text-sm font-medium text-gray-900 dark:text-slate-100`}>
            {getGPUType()}
          </span>
          <span className={`text-xs text-gray-600 dark:text-slate-300`}>
            {gpuInfo.memory_gb.toFixed(1)}GB
          </span>
        </div>
        <ChevronDown className={`h-3 w-3 text-gray-600 dark:text-slate-300 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className={`absolute top-full right-0 mt-2 w-80 p-4 bg-${colors.card} border border-${colors.border} rounded-xl shadow-lg z-50 animate-slide-down`}>
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Monitor className="h-5 w-5 text-accent-blue" />
              <h3 className={`text-lg font-semibold text-gray-900 dark:text-slate-100`}>System Info</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className={`text-sm text-gray-600 dark:text-slate-300`}>GPU Type</span>
                <div className="flex items-center space-x-2">
                  {getGPUIcon()}
                  <span className={`text-sm font-medium text-gray-900 dark:text-slate-100`}>{getGPUType()}</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className={`text-sm text-gray-600 dark:text-slate-300`}>Device</span>
                <span className={`text-sm font-medium text-gray-900 dark:text-slate-100`}>{gpuInfo.name}</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className={`text-sm text-gray-600 dark:text-slate-300`}>Memory</span>
                <span className={`text-sm font-medium text-gray-900 dark:text-slate-100`}>{gpuInfo.memory_gb.toFixed(1)} GB</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className={`text-sm text-gray-600 dark:text-slate-300`}>CUDA</span>
                <span className={`text-sm font-medium ${gpuInfo.cuda_available ? 'text-green-500' : 'text-gray-500'}`}>
                  {gpuInfo.cuda_available ? 'Available' : 'Not Available'}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className={`text-sm text-gray-600 dark:text-slate-300`}>ROCm</span>
                <span className={`text-sm font-medium ${gpuInfo.rocm_available ? 'text-purple-500' : 'text-gray-500'}`}>
                  {gpuInfo.rocm_available ? 'Available' : 'Not Available'}
                </span>
              </div>
            </div>
            
            <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-2 text-xs text-gray-500">
                <Activity className="h-3 w-3" />
                <span>Backend running on localhost:8000</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
