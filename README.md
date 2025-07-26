# VoxAI
[![PyPI version](https://img.shields.io/pypi/v/voxai)](https://pypi.org/project/voxai/)  
[![License MIT](https://img.shields.io/pypi/l/voxai)](https://github.com/rtiwariops/voxai/blob/main/LICENSE)  

![VoxAI Logo](https://raw.githubusercontent.com/rtiwariops/voxai/main/assets/logo.png)

## 🚀 Features

- **🎧 System Audio Capture**  
  Captures clean audio from your computer (meetings, browser, apps) - no external microphone noise or room interference.
- **🎮 Manual Control**  
  Start/Stop buttons let you define exactly the boundaries of your prompt—perfect for long, multi-sentence queries.  
- **🔄 Optimized Transcription**  
  Fast Whisper tiny model with GPU acceleration for near real-time speech-to-text processing.  
- **⚡ Live AI Streaming**  
  ChatGPT-like streaming responses with structured formatting, headings, and bullet points.  
- **🎯 Concise Technical Responses**  
  Structured responses with bullet points optimized for quick technical understanding.
- **🔧 Configurable Model**  
  Swap between Gemini variants via environment (no code changes).  
- **🚀 Zero-Install UI Bootstrapping**  
  On first run `voxai` auto-installs Electron dependencies—included in the PyPI package—so you only ever need `pip install voxai` and `voxai`.
- **🎨 Modern UI**  
  Beautiful gradient interface with real-time status indicators, typing animations, and professional styling.  

---

## 🎯 Quickstart

## ⚙️ Prerequisites

- **Python ≥3.7**  
- **Node.js ≥14** (for Electron UI)

**That's it!** VoxAI automatically installs Electron dependencies on first run.

**Important:** VoxAI captures system audio only (meetings, browser, apps) - no external microphone.

## Install & Run
### 1) Install from PyPI
pip install voxai

### 2) Configure environment
cp .env.example .env

##### Edit `.env`:
GENAI_API_KEY=sk-…

GENAI_MODEL=gemini-1.5-flash

### 3) Launch the app
voxai

On first launch, VoxAI will automatically run npm install inside its bundled electron/ folder, then open a desktop window.

---

## 📖 Usage
Once the UI opens:

Start: Click Start to begin recording.

Speak: Listen to question`s from zoom, teams or any other tool aloud—no length limit.

Stop: Click Stop. Whisper transcribes your entire clip to the Transcript pane.

Ask AI: Click Ask AI. Gemini’s answer streams live into the Answer pane.

Copy & Share: Everything is plain text—copy it, paste it, or feed it into your own RAG/finetuning pipeline.

---

## 🎧 How It Works

- **🎤 Audio Source Detection**  
  VoxAI automatically detects your system audio setup and displays it in the beautiful UI with real-time status.

- **🎬 Start Recording**  
  Click **Start Recording** to capture clean audio from your computer (meetings, browser, apps).

- **✋ Stop & Transcribe**  
  Click **Stop Recording**. Whisper transcribes the entire audio clip and shows it under **Transcript**.

- **🤖 Ask AI**  
  Click **Ask AI** to send the transcript to Gemini. Get ChatGPT-like streaming responses with structured formatting.

- **📋 Copy & Share**  
  Professional responses with headings, bullet points, and technical depth—perfect for documentation or sharing.

## 🎨 Modern UI Features

- **🌈 Beautiful Design**: Gradient backgrounds with glassmorphism effects
- **💫 Real-time Animations**: Typing indicators, loading states, and smooth transitions  
- **📊 Status Indicators**: Color-coded status dots (🟢 ready, 🔴 recording, 🟡 thinking)
- **📱 Responsive Layout**: Works perfectly on different screen sizes
- **🎯 Professional Formatting**: AI responses with headings, bullets, and technical structure

---

## 🛠 Configuration

Set in `.env`:

| Variable        | Description                                | Example             |
|-----------------|--------------------------------------------|---------------------|
| `GENAI_API_KEY` | Google Generative AI (Gemini) API key      | `sk-…`              |
| `GENAI_MODEL`   | Gemini model to use                        | `gemini-1.5-flash`  |

See `.env.example` for reference.

---

## 🛠️ Developer Guide

### Install from source
git clone https://github.com/rtiwariops/voxai.git

cd voxai

pip install -e .

npm install -g electron

voxai

## 🔧 System Audio Setup

VoxAI captures **computer audio only** (no microphone) - perfect for meeting recordings, browser audio, and app sounds without room noise.

### macOS Setup
```bash
# 1. Install BlackHole
brew install blackhole-2ch

# 2. Install Ladiocast
# Download from: https://existential.audio/ladiocast/
```

**Configure Ladiocast:**
1. **Input 1**: Set to your audio source (Built-in Input or system audio)
2. **Main**: Route Input 1 to Main (for speakers)
3. **Aux 1**: Set to BlackHole 2ch
4. **Enable**: Route Input 1 to Aux 1

**System Settings:**
- **Input**: BlackHole 2ch (VoxAI reads from here)
- **Output**: Built-in Output (you hear from here)

### Windows (Built-in)
WASAPI loopback support is built into Windows 10+ - VoxAI will auto-detect.

### Linux (Built-in)
PulseAudio monitor devices are auto-detected - no setup needed.

**Result:** You hear audio through speakers + VoxAI captures clean computer audio.

---

## ⚙️ Features

- Smart Cross-Platform Audio Detection
- Manual Control for long-form Q&A
- One-Shot, Full-Clip Transcription
- Live Token-by-Token AI Streaming
- Configurable Gemini Model (`.env`)
- Electron-Based Desktop UI

---

## 📜 License

MIT License © 2025 Ravi (Robbie) Tiwari

Released under the MIT License.