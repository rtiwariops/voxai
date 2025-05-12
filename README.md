# VoxAI

![VoxAI Logo](https://raw.githubusercontent.com/rtiwariops/voxai/main/assets/logo.png)

Voice-driven AI assistant capturing desktop audio and streaming live, textual answers—ideal for interviews, meetings, and knowledge work.

---

## ⚙️ Prerequisites

- **Python ≥3.7**  
- **Node.js ≥14**  
- **Electron** (global)  
  ```bash
  npm install -g electron

## 🚀 Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/rtiwariops/voxai.git
   cd voxai


2. **Create & activate a Python virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   # .venv\Scripts\activate       # Windows PowerShell
   ```

3. **Install Python dependencies**  
   ```bash
   pip install --upgrade pip
   pip install .
   ```

4.  **Run** VoxAI
   ```bash
   voxai
   ```

5. **Configure environment variables**  
   ```bash
   cp .env.example .env
   ```

6. **Edit `.env` and provide:**
   ```env
   GENAI_API_KEY=sk-...
   GENAI_MODEL=gemini-1.5-flash
   ```

7. **Run VoxAI**  
   ```bash
   VoxAI
   ```

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

## ⚙️ Features

- Universal Audio Capture (BlackHole, etc.)
- Manual Control for long-form Q&A
- One-Shot, Full-Clip Transcription
- Live Token-by-Token AI Streaming
- Configurable Gemini Model (`.env`)
- Electron-Based Desktop UI

---

## 📦 Installation Options

**From source:**
```bash
git clone https://github.com/YourUsername/VoxAI.git
cd VoxAI
pip install -e .
VoxAI
```

**From PyPI:**
```bash
pip install VoxAI
VoxAI
```

---

## 🛠 Configuration

Set in `.env`:

| Variable        | Description                                | Example             |
|-----------------|--------------------------------------------|---------------------|
| `GENAI_API_KEY` | Google Generative AI (Gemini) API key      | `sk-…`              |
| `GENAI_MODEL`   | Gemini model to use                        | `gemini-1.5-flash`  |

See `.env.example` for reference.
