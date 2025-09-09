#!/usr/bin/env node

/**
 * GPU Detection Script for GenAI Media Studio
 * Checks for available GPU support (CUDA, ROCm, MPS)
 */

const { execSync } = require('child_process');
const os = require('os');

console.log('🔍 Checking GPU support for GenAI Media Studio...\n');

function checkNVIDIA() {
  try {
    const output = execSync('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits', { encoding: 'utf8' });
    const gpus = output.trim().split('\n');
    console.log('✅ NVIDIA GPU detected:');
    gpus.forEach((gpu, index) => {
      const [name, memory] = gpu.split(', ');
      console.log(`   GPU ${index + 1}: ${name} (${memory} MB)`);
    });
    return true;
  } catch (error) {
    return false;
  }
}

function checkAMD() {
  try {
    const output = execSync('rocm-smi --showproductname', { encoding: 'utf8' });
    console.log('✅ AMD GPU detected:');
    console.log(`   ${output.trim()}`);
    return true;
  } catch (error) {
    return false;
  }
}

function checkAppleSilicon() {
  const platform = os.platform();
  const arch = os.arch();
  
  if (platform === 'darwin' && arch === 'arm64') {
    console.log('✅ Apple Silicon detected (M1/M2/M3)');
    console.log('   MPS acceleration will be available');
    return true;
  }
  return false;
}

function checkCPU() {
  const cpus = os.cpus();
  const totalMemory = Math.round(os.totalmem() / 1024 / 1024 / 1024);
  
  console.log('💻 CPU Information:');
  console.log(`   Processor: ${cpus[0].model}`);
  console.log(`   Cores: ${cpus.length}`);
  console.log(`   Memory: ${totalMemory} GB`);
  return true;
}

// Main check function
function main() {
  let gpuFound = false;
  
  console.log('Checking for GPU support...\n');
  
  // Check NVIDIA
  if (checkNVIDIA()) {
    gpuFound = true;
  }
  
  // Check AMD
  if (checkAMD()) {
    gpuFound = true;
  }
  
  // Check Apple Silicon
  if (checkAppleSilicon()) {
    gpuFound = true;
  }
  
  // If no GPU found, show CPU info
  if (!gpuFound) {
    console.log('⚠️  No GPU detected');
    checkCPU();
    console.log('\n💡 For better performance, consider installing:');
    console.log('   - NVIDIA CUDA drivers for NVIDIA GPUs');
    console.log('   - AMD ROCm drivers for AMD GPUs');
    console.log('   - The application will run on CPU');
  }
  
  console.log('\n🎯 GPU Support Summary:');
  console.log(`   CUDA: ${checkNVIDIA() ? '✅' : '❌'}`);
  console.log(`   ROCm: ${checkAMD() ? '✅' : '❌'}`);
  console.log(`   MPS:  ${checkAppleSilicon() ? '✅' : '❌'}`);
  console.log(`   CPU:  ✅ (Always available)`);
  
  console.log('\n✨ GenAI Media Studio is ready to run!');
}

main();
