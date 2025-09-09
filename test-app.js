#!/usr/bin/env node

/**
 * GenAI Media Studio Test Script
 * Tests the complete application flow
 */

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

// Test configuration
const config = {
  backendUrl: 'http://localhost:8000',
  frontendUrl: 'http://localhost:3000',
  timeout: 30000, // 30 seconds
  retryDelay: 2000 // 2 seconds
};

// Test results
let testResults = {
  passed: 0,
  failed: 0,
  total: 0,
  details: []
};

// Utility functions
function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logTest(testName, status, details = '') {
  testResults.total++;
  if (status === 'PASS') {
    testResults.passed++;
    log(`✓ ${testName}`, 'green');
  } else {
    testResults.failed++;
    log(`✗ ${testName}`, 'red');
    if (details) {
      log(`  ${details}`, 'yellow');
    }
  }
  testResults.details.push({ testName, status, details });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function makeRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(url, options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const jsonData = JSON.parse(data);
          resolve({ status: res.statusCode, data: jsonData });
        } catch (e) {
          resolve({ status: res.statusCode, data: data });
        }
      });
    });
    
    req.on('error', reject);
    req.setTimeout(config.timeout, () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });
    
    if (options.body) {
      req.write(options.body);
    }
    req.end();
  });
}

// Test functions
async function testFileStructure() {
  log('\n📁 Testing file structure...', 'cyan');
  
  const requiredFiles = [
    'package.json',
    'backend/main.py',
    'backend/requirements.txt',
    'frontend/package.json',
    'frontend/src/app/page.tsx',
    'frontend/src/components/GenerationForm.tsx',
    'frontend/src/components/JobQueue.tsx',
    'frontend/src/components/MediaPreview.tsx',
    'frontend/src/components/GPUInfo.tsx',
    'frontend/src/components/ModelManager.tsx',
    'frontend/src/contexts/ThemeContext.tsx',
    'frontend/src/hooks/useThemeColors.ts',
    'setup.sh',
    'setup.bat'
  ];
  
  for (const file of requiredFiles) {
    if (fs.existsSync(file)) {
      logTest(`File exists: ${file}`, 'PASS');
    } else {
      logTest(`File exists: ${file}`, 'FAIL', 'File not found');
    }
  }
}

async function testBackendAPI() {
  log('\n🔧 Testing backend API...', 'cyan');
  
  try {
    // Test root endpoint
    const rootResponse = await makeRequest(`${config.backendUrl}/`);
    if (rootResponse.status === 200) {
      logTest('Backend root endpoint', 'PASS');
    } else {
      logTest('Backend root endpoint', 'FAIL', `Status: ${rootResponse.status}`);
    }
    
    // Test GPU info endpoint
    const gpuResponse = await makeRequest(`${config.backendUrl}/gpu-info`);
    if (gpuResponse.status === 200) {
      logTest('GPU info endpoint', 'PASS');
    } else {
      logTest('GPU info endpoint', 'FAIL', `Status: ${gpuResponse.status}`);
    }
    
    // Test models endpoint
    const modelsResponse = await makeRequest(`${config.backendUrl}/models`);
    if (modelsResponse.status === 200) {
      logTest('Models endpoint', 'PASS');
    } else {
      logTest('Models endpoint', 'FAIL', `Status: ${modelsResponse.status}`);
    }
    
    // Test jobs endpoint
    const jobsResponse = await makeRequest(`${config.backendUrl}/jobs`);
    if (jobsResponse.status === 200) {
      logTest('Jobs endpoint', 'PASS');
    } else {
      logTest('Jobs endpoint', 'FAIL', `Status: ${jobsResponse.status}`);
    }
    
  } catch (error) {
    logTest('Backend API connectivity', 'FAIL', error.message);
  }
}

async function testFrontendBuild() {
  log('\n🎨 Testing frontend build...', 'cyan');
  
  return new Promise((resolve) => {
    const buildProcess = spawn('npm', ['run', 'build'], {
      cwd: path.join(__dirname, 'frontend'),
      stdio: 'pipe'
    });
    
    let output = '';
    buildProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    buildProcess.stderr.on('data', (data) => {
      output += data.toString();
    });
    
    buildProcess.on('close', (code) => {
      if (code === 0) {
        logTest('Frontend build', 'PASS');
      } else {
        logTest('Frontend build', 'FAIL', `Exit code: ${code}`);
      }
      resolve();
    });
    
    buildProcess.on('error', (error) => {
      logTest('Frontend build', 'FAIL', error.message);
      resolve();
    });
  });
}

async function testDependencies() {
  log('\n📦 Testing dependencies...', 'cyan');
  
  // Check Python dependencies
  try {
    const pythonCheck = await new Promise((resolve) => {
      exec('python -c "import torch, fastapi, uvicorn, diffusers, transformers"', (error) => {
        resolve(!error);
      });
    });
    
    if (pythonCheck) {
      logTest('Python dependencies', 'PASS');
    } else {
      logTest('Python dependencies', 'FAIL', 'Missing required Python packages');
    }
  } catch (error) {
    logTest('Python dependencies', 'FAIL', error.message);
  }
  
  // Check Node.js dependencies
  try {
    const packageJson = JSON.parse(fs.readFileSync('frontend/package.json', 'utf8'));
    const nodeModulesExist = fs.existsSync('frontend/node_modules');
    
    if (nodeModulesExist) {
      logTest('Node.js dependencies', 'PASS');
    } else {
      logTest('Node.js dependencies', 'FAIL', 'node_modules not found. Run npm install');
    }
  } catch (error) {
    logTest('Node.js dependencies', 'FAIL', error.message);
  }
}

async function testConfiguration() {
  log('\n⚙️ Testing configuration...', 'cyan');
  
  // Check Tailwind config
  try {
    const tailwindConfig = fs.readFileSync('frontend/tailwind.config.ts', 'utf8');
    if (tailwindConfig.includes('darkMode') && tailwindConfig.includes('accent-blue')) {
      logTest('Tailwind configuration', 'PASS');
    } else {
      logTest('Tailwind configuration', 'FAIL', 'Missing required Tailwind settings');
    }
  } catch (error) {
    logTest('Tailwind configuration', 'FAIL', error.message);
  }
  
  // Check global CSS
  try {
    const globalCSS = fs.readFileSync('frontend/src/app/globals.css', 'utf8');
    if (globalCSS.includes('--accent-blue') && globalCSS.includes('@import "tailwindcss"')) {
      logTest('Global CSS', 'PASS');
    } else {
      logTest('Global CSS', 'FAIL', 'Missing required CSS variables');
    }
  } catch (error) {
    logTest('Global CSS', 'FAIL', error.message);
  }
  
  // Check environment file
  if (fs.existsSync('.env')) {
    logTest('Environment file', 'PASS');
  } else {
    logTest('Environment file', 'FAIL', '.env file not found');
  }
}

async function testDirectories() {
  log('\n📂 Testing directory structure...', 'cyan');
  
  const requiredDirs = ['outputs', 'outputs/videos', 'outputs/audio', 'models', 'logs'];
  
  for (const dir of requiredDirs) {
    if (fs.existsSync(dir)) {
      logTest(`Directory exists: ${dir}`, 'PASS');
    } else {
      logTest(`Directory exists: ${dir}`, 'FAIL', 'Directory not found');
    }
  }
}

async function testSetupScripts() {
  log('\n🚀 Testing setup scripts...', 'cyan');
  
  // Check setup.sh
  if (fs.existsSync('setup.sh')) {
    const setupSh = fs.readFileSync('setup.sh', 'utf8');
    if (setupSh.includes('GenAI Media Studio') && setupSh.includes('python3')) {
      logTest('Setup script (Linux/macOS)', 'PASS');
    } else {
      logTest('Setup script (Linux/macOS)', 'FAIL', 'Invalid setup.sh content');
    }
  } else {
    logTest('Setup script (Linux/macOS)', 'FAIL', 'setup.sh not found');
  }
  
  // Check setup.bat
  if (fs.existsSync('setup.bat')) {
    const setupBat = fs.readFileSync('setup.bat', 'utf8');
    if (setupBat.includes('GenAI Media Studio') && setupBat.includes('python')) {
      logTest('Setup script (Windows)', 'PASS');
    } else {
      logTest('Setup script (Windows)', 'FAIL', 'Invalid setup.bat content');
    }
  } else {
    logTest('Setup script (Windows)', 'FAIL', 'setup.bat not found');
  }
}

// Main test function
async function runTests() {
  log('🧪 GenAI Media Studio Test Suite', 'bright');
  log('================================', 'bright');
  
  await testFileStructure();
  await testDependencies();
  await testConfiguration();
  await testDirectories();
  await testSetupScripts();
  await testFrontendBuild();
  
  // Only test API if backend is running
  try {
    await makeRequest(`${config.backendUrl}/`);
    await testBackendAPI();
  } catch (error) {
    logTest('Backend API', 'SKIP', 'Backend not running (this is expected if not started)');
  }
  
  // Print summary
  log('\n📊 Test Summary', 'cyan');
  log('===============', 'cyan');
  log(`Total Tests: ${testResults.total}`, 'blue');
  log(`Passed: ${testResults.passed}`, 'green');
  log(`Failed: ${testResults.failed}`, 'red');
  
  const successRate = ((testResults.passed / testResults.total) * 100).toFixed(1);
  log(`Success Rate: ${successRate}%`, successRate >= 80 ? 'green' : 'yellow');
  
  if (testResults.failed > 0) {
    log('\n❌ Failed Tests:', 'red');
    testResults.details
      .filter(test => test.status === 'FAIL')
      .forEach(test => {
        log(`  • ${test.testName}`, 'red');
        if (test.details) {
          log(`    ${test.details}`, 'yellow');
        }
      });
  }
  
  if (testResults.failed === 0) {
    log('\n🎉 All tests passed! The application is ready to use.', 'green');
  } else {
    log('\n⚠️ Some tests failed. Please check the issues above.', 'yellow');
  }
  
  process.exit(testResults.failed > 0 ? 1 : 0);
}

// Run tests
runTests().catch(error => {
  log(`\n💥 Test suite crashed: ${error.message}`, 'red');
  process.exit(1);
});
