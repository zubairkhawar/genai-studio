// API Configuration
// Update this with your Colab ngrok URL when running backend on Colab

export const API_CONFIG = {
  // For local development (backend running locally)
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  
  // For Colab backend (working tunnel URL)
  // BASE_URL: 'https://text-to-media-app.loca.lt',
  
  ENDPOINTS: {
    HEALTH: '/health',
    MODELS: '/models',
    GENERATE: '/generate',
    JOBS: '/jobs',
  }
};

// Helper function to get full API URL
export const getApiUrl = (endpoint: string) => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};
