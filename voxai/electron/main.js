const {app,BrowserWindow,ipcMain} = require('electron');
const path=require('path');
const {spawn}=require('child_process');
let py;
function createWin(){
  const w=new BrowserWindow({width:800,height:600,webPreferences:{nodeIntegration:true,contextIsolation:false}});
  w.loadFile(path.join(__dirname,'index.html'));
  py=spawn('python3',['-m','voxai.core'],{cwd:path.join(__dirname,'..')});
  py.stdout.on('data',d=>d.toString().split('\n').filter(Boolean).forEach(l=>{
    if(l.startsWith('from-python:')) w.webContents.send('from-python',l.replace('from-python:',''));
  }));
  ipcMain.on('to-python',(_,m)=>py.stdin.write(m+'\n'));
  app.on('before-quit',()=>py.kill());
}
app.whenReady().then(createWin);
