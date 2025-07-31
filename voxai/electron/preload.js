// preload.js - Secure bridge between main and renderer processes
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Send commands to Python backend
  sendToPython: (message) => ipcRenderer.send('to-python', message),
  
  // Listen for messages from Python backend
  onFromPython: (callback) => ipcRenderer.on('from-python', callback),
  
  // Remove listeners (for cleanup)
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});