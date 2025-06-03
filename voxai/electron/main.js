// main.js

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow = null;
let pyProcess = null;

function createWindow() {
  // 1) Create the browser window.
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,      // allow require('electron') in renderer
      contextIsolation: false,    // so ipcRenderer works
    }
  });

  // Load our index.html (renderer) file.
  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  // 2) Spawn the Python backend with unbuffered output (−u).
  pyProcess = spawn('python3', ['-u', '-m', 'voxai.core'], {
    cwd: __dirname,
    env: process.env,
  });

  // If Python exits, log the code & signal
  pyProcess.on('exit', (code, signal) => {
    console.error(`⛔ Python process exited with code ${code} and signal ${signal}`);
    pyProcess = null;
  });

  // 3) Forward every line of Python stdout into the renderer under "from-python".
  pyProcess.stdout.on('data', rawData => {
    // rawData may contain multiple lines, so split them
    const lines = rawData.toString().split('\n').filter(line => line.length > 0);
    for (const line of lines) {
      // Log to the main process console
      console.log(`[Main ← Python] ${line}`);

      // If it starts with "from-python:", strip that prefix and send the remainder
      if (line.startsWith('from-python:')) {
        const payload = line.replace(/^from-python:/, '');

        if (mainWindow && mainWindow.webContents) {
          mainWindow.webContents.send('from-python', payload);
        } else {
          console.warn('[Main] Attempted to send to renderer, but window is null.');
        }
      } else {
        // If Python printed something unexpected (no "from-python:" prefix),
        // still forward it so you can see it in DevTools under "UNPREFIXED::…"
        if (mainWindow && mainWindow.webContents) {
          mainWindow.webContents.send('from-python', `UNPREFIXED::${line}`);
        }
      }
    }
  });

  // 4) Forward Python stderr as well (for tracebacks, permission errors, etc.).
  pyProcess.stderr.on('data', rawData => {
    const text = rawData.toString();
    console.error(`[Main ← Python stderr] ${text}`);
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('from-python', `STDERR::${text}`);
    }
  });

  // 5) Listen for messages from the renderer (e.g. "START", "STOP", "ASK::")
  ipcMain.on('to-python', (_, m) => {
    console.log(`[Main → Python stdin] ${JSON.stringify(m)}`);
    if (pyProcess && !pyProcess.killed) {
      pyProcess.stdin.write(m.trim() + '\n');
    } else {
      console.error('❌ Cannot write to Python stdin: process not running.');
    }
  });

  // 6) When the window is closed, clean up references and kill Python if still running.
  mainWindow.on('closed', () => {
    mainWindow = null;
    if (pyProcess) {
      pyProcess.kill();
      pyProcess = null;
    }
  });
}

// 7) When Electron has finished initialization, create the window.
app.whenReady().then(createWindow);

// 8) Quit the app when all windows are closed (except on macOS).
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
app.on('activate', () => {
  // On macOS, re‐create a window when the dock icon is clicked and no windows are open.
  if (mainWindow === null) {
    createWindow();
  }
});
