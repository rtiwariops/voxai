const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let pyProc;

function createWindow() {
  const win = new BrowserWindow({
    width: 800, height: 600,
    webPreferences: { nodeIntegration: true, contextIsolation: false }
  });
  win.loadFile(path.join(__dirname, 'index.html'));

  // spawn backend.core
  pyProc = spawn(process.env.PYTHON_EXECUTABLE || 'python3', ['-m','voxai.core'], {
    cwd: path.join(__dirname, '..')
  });
  pyProc.stdout.on('data', data => {
    data.toString().split('\n').filter(Boolean).forEach(line => {
      if (line.startsWith('from-python:')) {
        win.webContents.send('from-python', line.replace('from-python:', ''));
      }
    });
  });

  ipcMain.on('to-python', (_, msg) => {
    pyProc.stdin.write(msg + '\n');
  });

  app.on('before-quit', () => {
    pyProc.kill();
  });
}

app.whenReady().then(createWindow);
