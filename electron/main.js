const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let pyProc;

function createWindow() {
  const win = new BrowserWindow({
    width: 800, height: 600,
    webPreferences: { nodeIntegration: true, contextIsolation: false }
  });
  win.loadFile('index.html');

  // Start Python backend
  pyProc = spawn('python3', ['-m', 'backend.core']);

  pyProc.stdout.on('data', data => {
    const lines = data.toString().split('\n').filter(Boolean);
    for (const line of lines) {
      if (line.startsWith('from-python:')) {
        const payload = line.replace('from-python:', '');
        win.webContents.send('from-python', payload);
      }
    }
  });

  ipcMain.on('to-python', (_, msg) => {
    pyProc.stdin.write(msg + '\n');
  });
}

app.whenReady().then(createWindow);
