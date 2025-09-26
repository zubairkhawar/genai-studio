#!/usr/bin/env node

/**
 * Dependency Check Script for GenAI Media Studio
 * Verifies all required dependencies are installed
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔍 Checking dependencies for GenAI Media Studio...\n');

const dependencies = {
  python: {
    command: 'python --version',
    fallback: 'python3 --version',
    required: true,
    minVersion: '3.8.0'
  },
  node: {
    command: 'node --version',
    required: true,
    minVersion: '18.0.0'
  },
  npm: {
    command: 'npm --version',
    required: true
  },
  ffmpeg: {
    command: 'ffmpeg -version',
    required: true
  },
  pip: {
    command: 'pip --version',
    fallback: 'pip3 --version',
    required: true
  }
};

function checkCommand(command, fallback = null) {
  try {
    const output = execSync(command, { encoding: 'utf8' });
    return { success: true, output: output.trim() };
  } catch (error) {
    if (fallback) {
      try {
        const output = execSync(fallback, { encoding: 'utf8' });
        return { success: true, output: output.trim() };
      } catch (fallbackError) {
        return { success: false, error: fallbackError.message };
      }
    }
    return { success: false, error: error.message };
  }
}

function compareVersions(version1, version2) {
  const v1parts = version1.split('.').map(Number);
  const v2parts = version2.split('.').map(Number);
  
  for (let i = 0; i < Math.max(v1parts.length, v2parts.length); i++) {
    const v1part = v1parts[i] || 0;
    const v2part = v2parts[i] || 0;
    
    if (v1part > v2part) return 1;
    if (v1part < v2part) return -1;
  }
  
  return 0;
}

function checkDependency(name, config) {
  console.log(`Checking ${name}...`);
  
  const result = checkCommand(config.command, config.fallback);
  
  if (!result.success) {
    console.log(`❌ ${name}: Not found`);
    if (config.required) {
      console.log(`   Error: ${result.error}`);
      return false;
    }
    return true;
  }
  
  console.log(`✅ ${name}: ${result.output}`);
  
  // Check version if specified
  if (config.minVersion) {
    const versionMatch = result.output.match(/(\d+\.\d+\.\d+)/);
    if (versionMatch) {
      const version = versionMatch[1];
      if (compareVersions(version, config.minVersion) < 0) {
        console.log(`⚠️  ${name}: Version ${version} is below minimum required ${config.minVersion}`);
        return false;
      }
    }
  }
  
  return true;
}

function checkProjectFiles() {
  console.log('\nChecking project files...');
  
  const requiredFiles = [
    'backend/main.py',
    'backend/requirements.txt',
    'frontend/package.json',
    'frontend/next.config.ts',
    'setup.sh',
    'setup.bat',
    'README.md'
  ];
  
  let allFilesExist = true;
  
  requiredFiles.forEach(file => {
    if (fs.existsSync(file)) {
      console.log(`✅ ${file}`);
    } else {
      console.log(`❌ ${file}: Missing`);
      allFilesExist = false;
    }
  });
  
  return allFilesExist;
}

function checkDirectories() {
  console.log('\nChecking directories...');
  
  const requiredDirs = [
    'backend',
    'frontend',
    'outputs',
    'models',
    'scripts'
  ];
  
  let allDirsExist = true;
  
  requiredDirs.forEach(dir => {
    if (fs.existsSync(dir) && fs.statSync(dir).isDirectory()) {
      console.log(`✅ ${dir}/`);
    } else {
      console.log(`❌ ${dir}/: Missing`);
      allDirsExist = false;
    }
  });
  
  return allDirsExist;
}

function main() {
  let allDepsOk = true;
  
  // Check system dependencies
  Object.entries(dependencies).forEach(([name, config]) => {
    const isOk = checkDependency(name, config);
    if (!isOk && config.required) {
      allDepsOk = false;
    }
  });
  
  // Check project structure
  const filesOk = checkProjectFiles();
  const dirsOk = checkDirectories();
  
  console.log('\n📊 Dependency Check Summary:');
  console.log(`   System Dependencies: ${allDepsOk ? '✅' : '❌'}`);
  console.log(`   Project Files: ${filesOk ? '✅' : '❌'}`);
  console.log(`   Directories: ${dirsOk ? '✅' : '❌'}`);
  
  if (allDepsOk && filesOk && dirsOk) {
    console.log('\n🎉 All dependencies are satisfied!');
    console.log('✨ GenAI Media Studio is ready to run!');
    console.log('\nTo start the application:');
    console.log('   ./setup.sh (Linux/macOS)');
    console.log('   setup.bat (Windows)');
    console.log('   npm run dev (Development)');
  } else {
    console.log('\n⚠️  Some dependencies are missing or incorrect.');
    console.log('Please install the missing dependencies and try again.');
    process.exit(1);
  }
}

main();
