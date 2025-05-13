# VoxAI
[![PyPI version](https://img.shields.io/pypi/v/voxai)](https://pypi.org/project/voxai/)  
[![License MIT](https://img.shields.io/pypi/l/voxai)](https://github.com/rtiwariops/voxai/blob/main/LICENSE)  

![VoxAI Logo](https://raw.githubusercontent.com/rtiwariops/voxai/main/assets/logo.png)

## 🚀 Features

- **Universal Audio Capture**  
  Record mic, system audio, or any combination via BlackHole (macOS) or equivalent loopback drivers.  
- **Manual Control**  
  Start/Stop buttons let you define exactly the boundaries of your prompt—perfect for long, multi-sentence queries.  
- **One-Shot Transcription**  
  Whisper processes the entire recording in a single call—no fragmented sentences.  
- **Live AI Streaming**  
  Gemini’s response appears token by token, just like in ChatGPT’s streaming interface.  
- **Configurable Model**  
  Swap between Gemini variants via environment (no code changes).  
- **Zero-Install UI Bootstrapping**  
  On first run `voxai` auto-installs the minimal Electron UI source—included in the PyPI package—so you only ever need `pip install voxai` and `voxai`.  

---

## 🎯 Quickstart

## ⚙️ Prerequisites

- **Python ≥3.7**  
- **Node.js ≥14**  
- **Electron** (global)  
  ```bash
  npm install -g electron

- **macOS Audio Loopback (BlackHole 2ch + Ladiocast)**:
    
   1. **Install BlackHole 2ch**::

       brew install blackhole-2ch

  2. **Create a Multi-Output Device**  
     - Open **Audio MIDI Setup**  
     - Click “+” → **Create Multi-Output Device**  
     - Check both **BlackHole 2ch** and **MacBook Pro Speakers**

  3. **Select Multi-Output Device**  
     System Preferences → Sound → Output → **Multi-Output Device**

  4. **Configure Ladiocast**  
     - **Input 1**: **BlackHole 2ch** → route to **Main**  
     - **Main Output**: **MacBook Pro Speakers**  
     - Mute other channels

  This will route all system and microphone audio into VoxAI.

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

- **Start Recording**  
  Click **Start** to capture audio from your input device (e.g., BlackHole).

- **Stop & Transcribe**  
  Click **Stop**. Whisper transcribes and shows the full audio clip under **Transcript**.

- **Ask AI**  
  Click **Ask AI** to send the transcript to Gemini. Answers stream live in the **Answer** panel.

- **Copy & Share**  
  Output is plain text—reuse it in RAG/finetune workflows or anywhere else.

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

## ⚙️ Features

- Universal Audio Capture (BlackHole, etc.)
- Manual Control for long-form Q&A
- One-Shot, Full-Clip Transcription
- Live Token-by-Token AI Streaming
- Configurable Gemini Model (`.env`)
- Electron-Based Desktop UI

---

## 📜 License

MIT License © 2025 Ravi (Robbie) Tiwari

Released under the MIT License.