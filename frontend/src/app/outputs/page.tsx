"use client";

import { useEffect, useState } from 'react';
import { Play, Volume2, Download, Calendar, HardDrive, RefreshCw, Pause, Trash2, Image as ImageIcon, Edit2, Check, X } from 'lucide-react';
import { useThemeColors } from '@/hooks/useThemeColors';
import { getApiUrl } from '@/config';

interface OutputFile {
  filename: string;
  path: string;
  size: number;
  created: number;
}

interface OutputsData {
  videos: OutputFile[];
  audio: OutputFile[];
  images: OutputFile[];
}

export default function Page() {
  const [outputs, setOutputs] = useState<OutputsData>({ videos: [], audio: [], images: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [playingAudio, setPlayingAudio] = useState<string | null>(null);
  const [editingFilename, setEditingFilename] = useState<string | null>(null);
  const [newFilename, setNewFilename] = useState<string>('');
  const [activeFilter, setActiveFilter] = useState<'videos' | 'image' | 'audio'>('videos');
  const colors = useThemeColors();

  const fetchOutputs = async (isInitialLoad = false) => {
    if (isInitialLoad) {
      setLoading(true);
    }
    setError(null);
    try {
      const [videosRes, audioRes, imagesRes] = await Promise.all([
        fetch(getApiUrl('/outputs/videos')),
        fetch(getApiUrl('/outputs/audio')),
        fetch(getApiUrl('/outputs/images'))
      ]);

      const videosData = videosRes.ok ? await videosRes.json() : { videos: [] };
      const audioData = audioRes.ok ? await audioRes.json() : { audio: [] };
      const imagesData = imagesRes.ok ? await imagesRes.json() : { images: [] };

      setOutputs({
        videos: videosData.videos || [],
        audio: audioData.audio || [],
        images: imagesData.images || []
      });
    } catch (err) {
      setError('Failed to fetch outputs');
      console.error('Error fetching outputs:', err);
    } finally {
      if (isInitialLoad) {
        setLoading(false);
      }
    }
  };

  useEffect(() => { fetchOutputs(true); }, []);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const getFileIcon = (filename: string) => {
    if (filename.match(/\.(mp4|avi|mov|webm)$/i)) {
      return <Play className="h-5 w-5 text-accent-blue" />;
    }
    if (filename.match(/\.(wav|mp3|m4a|flac)$/i)) {
      return <Volume2 className="h-5 w-5 text-accent-violet" />;
    }
    if (filename.match(/\.(png|jpg|jpeg|gif|webp)$/i)) {
      return <ImageIcon className="h-5 w-5 text-accent-green" />;
    }
    return <HardDrive className="h-5 w-5 text-gray-500" />;
  };

  const getFileType = (filename: string) => {
    if (filename.match(/\.(mp4|avi|mov|webm)$/i)) return 'video';
    if (filename.match(/\.(wav|mp3|m4a|flac)$/i)) return 'audio';
    if (filename.match(/\.(png|jpg|jpeg|gif|webp)$/i)) return 'image';
    return 'file';
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

  const handleDeleteFile = async (file: any) => {
    if (!confirm(`Are you sure you want to delete ${file.filename}?`)) {
      return;
    }

    try {
      const response = await fetch(getApiUrl(`/outputs/${file.type}/${file.filename}`), {
        method: 'DELETE',
      });

      if (response.ok) {
        // Refresh the outputs list
        await fetchOutputs(false);
      } else {
        // Try to get error message from response
        let errorMessage = 'Failed to delete file';
        try {
          const errorData = await response.json();
          if (errorData && errorData.detail) {
            errorMessage = errorData.detail;
          } else if (errorData && errorData.message) {
            errorMessage = errorData.message;
          }
        } catch (parseError) {
          // If JSON parsing fails, use the status text
          errorMessage = response.statusText || 'Failed to delete file';
        }
        
        console.error('Failed to delete file:', {
          status: response.status,
          statusText: response.statusText,
          message: errorMessage
        });
        alert(errorMessage);
      }
    } catch (error) {
      console.error('Error deleting file:', error);
      const errorMessage = error instanceof Error ? error.message : 'Network error: Unable to delete file';
      alert(errorMessage);
    }
  };

  const handleRenameFile = async (file: any, newName: string) => {
    if (!newName.trim() || newName === file.filename) {
      setEditingFilename(null);
      setNewFilename('');
      return;
    }

    try {
      const response = await fetch(getApiUrl(`/outputs/${file.type}/${file.filename}/rename`), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ new_filename: newName }),
      });

      if (response.ok) {
        await fetchOutputs(false);
        setEditingFilename(null);
        setNewFilename('');
      } else {
        const errorData = await response.json();
        alert(errorData.detail || 'Failed to rename file');
      }
    } catch (error) {
      console.error('Error renaming file:', error);
      alert('Network error: Unable to rename file');
    }
  };

  const startEditing = (filename: string) => {
    setEditingFilename(filename);
    setNewFilename(filename);
  };

  const cancelEditing = () => {
    setEditingFilename(null);
    setNewFilename('');
  };

  const allFiles = [
    ...outputs.videos.map(file => ({ ...file, type: 'videos' as const })),
    ...outputs.images.map(file => ({ ...file, type: 'image' as const })),
    ...outputs.audio.map(file => ({ ...file, type: 'audio' as const }))
  ].sort((a, b) => b.created - a.created);

  const filteredFiles = allFiles.filter(file => file.type === activeFilter);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent">
            Generated Outputs
          </h1>
          <p className={`text-${colors.text.secondary} mt-1`}>
            View, download, and manage your generated media files
          </p>
        </div>
        <button 
          onClick={() => fetchOutputs(false)} 
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 rounded-xl border border-gray-200 dark:border-slate-600 text-accent-blue hover:bg-accent-blue/10 transition-all duration-200 hover:scale-105"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          <span className="text-sm font-medium">Refresh</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-600">
          <span className="font-medium">{error}</span>
        </div>
      )}

      {/* Filter Chips */}
      <div className="flex flex-wrap gap-3">
        {[
          { key: 'videos', label: 'Videos', count: outputs.videos.length, icon: Play, color: 'accent-blue' },
          { key: 'image', label: 'Images', count: outputs.images.length, icon: ImageIcon, color: 'accent-green' },
          { key: 'audio', label: 'Audio', count: outputs.audio.length, icon: Volume2, color: 'accent-violet' },
        ].map(({ key, label, count, icon: Icon, color }) => (
          <button
            key={key}
            onClick={() => setActiveFilter(key as any)}
            className={`group relative flex items-center space-x-3 px-4 py-3 rounded-2xl border-2 transition-all duration-300 hover:scale-105 ${
              activeFilter === key
                ? `border-${color} bg-${color}/10 shadow-lg`
                : 'border-gray-200 dark:border-slate-600 bg-white dark:bg-slate-700/50 hover:border-gray-300 dark:hover:border-slate-500'
            }`}
          >
            <div className={`p-2 rounded-xl ${
              activeFilter === key ? `bg-${color}/20` : `bg-${color}/10`
            }`}>
              <Icon className={`h-5 w-5 ${
                activeFilter === key ? `text-${color}` : `text-${color}`
              }`} />
            </div>
            <div className="text-left">
              <h3 className={`font-bold text-lg ${
                activeFilter === key ? `text-${color}` : colors.text.primary
              }`}>
                {label}
              </h3>
              <div className="flex items-center space-x-2">
                <span className={`text-sm ${
                  activeFilter === key ? `text-${color}/80` : colors.text.secondary
                }`}>
                  {count} files
                </span>
                {count > 0 && (
                  <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                    activeFilter === key 
                      ? `bg-${color} text-white` 
                      : `bg-${color}/20 text-${color}`
                  }`}>
                    {count}
                  </span>
                )}
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Files List */}
    <div className="space-y-6">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-accent-blue to-accent-violet bg-clip-text text-transparent">
          Generated Media
        </h2>
        
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center space-x-3">
              <div className="w-6 h-6 border-2 border-accent-blue border-t-transparent rounded-full animate-spin"></div>
              <span className={`${colors.text.secondary}`}>Loading files...</span>
            </div>
          </div>
        ) : filteredFiles.length === 0 ? (
          <div className="text-center py-12">
            <HardDrive className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h3 className={`text-lg font-semibold ${colors.text.primary} mb-2`}>
              No {activeFilter} files yet
            </h3>
            <p className={`${colors.text.secondary}`}>
              Generate some {activeFilter} content to see your outputs here
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredFiles.map((file) => (
              <div
                key={file.path}
                className={`p-4 rounded-2xl border-2 shadow-md bg-white dark:bg-gradient-to-br dark:from-slate-900/90 dark:to-slate-800/50 backdrop-blur-md transition-all duration-300 hover:scale-[1.02] hover:shadow-lg ${
                  file.type === 'videos' 
                    ? 'border-accent-blue/30 hover:border-accent-blue/50' 
                    : file.type === 'image'
                    ? 'border-accent-green/30 hover:border-accent-green/50'
                    : 'border-accent-violet/30 hover:border-accent-violet/50'
                }`}
              >
                {/* File Info Header */}
                <div className="mb-4">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-xl ${
                        file.type === 'videos' ? 'bg-accent-blue/10' 
                        : file.type === 'image' ? 'bg-accent-green/10'
                        : 'bg-accent-violet/10'
                      }`}>
                        {getFileIcon(file.filename)}
                      </div>
                      <div className="flex-1 min-w-0">
                        {editingFilename === file.filename ? (
                          <div className="flex items-center space-x-2">
                            <input
                              type="text"
                              value={newFilename}
                              onChange={(e) => setNewFilename(e.target.value)}
                              className="flex-1 px-2 py-1 text-sm border border-gray-300 dark:border-slate-600 rounded bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-100"
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  handleRenameFile(file, newFilename);
                                } else if (e.key === 'Escape') {
                                  cancelEditing();
                                }
                              }}
                              autoFocus
                            />
                            <button
                              onClick={() => handleRenameFile(file, newFilename)}
                              className="p-1 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/20 rounded"
                            >
                              <Check className="h-4 w-4" />
                            </button>
                            <button
                              onClick={cancelEditing}
                              className="p-1 text-red-600 hover:bg-red-100 dark:hover:bg-red-900/20 rounded"
                            >
                              <X className="h-4 w-4" />
                            </button>
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2">
                            <h3 className="font-bold text-lg truncate flex-1">{file.filename}</h3>
                            <button
                              onClick={() => startEditing(file.filename)}
                              className="p-1 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-slate-600 rounded"
                              title="Rename file"
                            >
                              <Edit2 className="h-4 w-4" />
                            </button>
                          </div>
                        )}
                        <div className="flex items-center space-x-4 text-sm text-gray-500 mt-1">
                          <span className="flex items-center space-x-1">
                            <HardDrive className="h-4 w-4" />
                            <span>{formatFileSize(file.size)}</span>
                          </span>
                          <span className="flex items-center space-x-1">
                            <Calendar className="h-4 w-4" />
                            <span>{formatDate(file.created)}</span>
                          </span>
                        </div>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      file.type === 'videos' 
                        ? 'bg-accent-blue/10 text-accent-blue' 
                        : file.type === 'image'
                        ? 'bg-accent-green/10 text-accent-green'
                        : 'bg-accent-violet/10 text-accent-violet'
                    }`}>
                      {file.type.toUpperCase()}
                    </span>
                  </div>
                </div>

                {/* Media Player */}
                {file.type === 'videos' ? (
                  <div className="mb-4">
                    <video 
                      controls 
                      className="w-full rounded-lg shadow-lg"
                      src={getApiUrl(file.path)}
                    >
                      Your browser does not support the video tag.
                    </video>
                  </div>
                ) : file.type === 'image' ? (
                  <div className="mb-4">
                    <img 
                      src={getApiUrl(file.path.replace('/outputs/', '/outputs/image/'))}
                      alt={file.filename}
                      className="w-full rounded-lg shadow-lg object-cover"
                      style={{ maxHeight: '300px' }}
                    />
                  </div>
                ) : (
                  <div className="mb-4">
                    <audio
                      controls
                      preload="metadata"
                      className="w-full rounded-lg shadow-lg"
                      src={getApiUrl(file.path)}
                    >
                      Your browser does not support the audio tag.
                    </audio>
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center justify-between space-x-3">
                  <a
                    href={getApiUrl(file.type === 'image' ? file.path.replace('/outputs/', '/outputs/image/') : file.path)}
                    download
                    className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-xl font-semibold transition-all duration-200 hover:scale-105 border-2 ${
                      file.type === 'videos'
                        ? 'bg-accent-blue text-white hover:bg-accent-blue/90 border-accent-blue'
                        : file.type === 'image'
                        ? 'bg-accent-green text-white hover:bg-accent-green/90 border-accent-green'
                        : 'bg-accent-violet text-white hover:bg-accent-violet/90 border-accent-violet'
                    }`}
                    title="Download file"
                  >
                    <Download className="h-4 w-4" />
                    <span>Download</span>
                  </a>
                  
                  <button
                    onClick={() => handleDeleteFile(file)}
                    className="flex items-center justify-center w-10 h-10 rounded-xl font-medium transition-all duration-200 hover:scale-105 bg-red-500/10 text-red-600 hover:bg-red-500/20 border border-red-500/30"
                    title="Delete file"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
          </div>
        ))}
          </div>
        )}
      </div>
    </div>
  );
}


