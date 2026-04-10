# 🧠 AI-Powered Agent For Healthcare
### Cognitive Rehabilitation Assistant — Voice-Driven Orientation Therapy

An AI-powered, fully offline voice interaction system for cognitive rehabilitation. It conducts structured orientation therapy sessions, scores patient responses using semantic classification, escalates emergencies automatically, and presents a staff dashboard — all without cloud API dependencies.

---

## Architecture Overview

```
task.py  (entry point)
  ├── speech_Recognition.py   — Whisper / faster-whisper ASR + Silero VAD
  ├── speech_synthesis.py     — Kokoro TTS + WSOLA time-stretch
  ├── classifier.py           — BGE-large + FAISS semantic scoring
  ├── emergency.py            — Alert escalation
  ├── report.py               — Session report generation
  ├── database/db.py          — SQLite patient records
  └── dashboard.py            — PyQt6 staff GUI
```

---

## Requirements

| Item | Requirement |
|------|-------------|
| OS | Windows 10 / 11 (64-bit) |
| Python | **3.9 – 3.11** — [download](https://www.python.org/downloads/) |
| eSpeak NG | Required by Kokoro TTS — [download](https://github.com/espeak-ng/espeak-ng/releases) |
| GPU | NVIDIA GPU with CUDA recommended (CPU mode also supported) |
| CUDA Toolkit | 11.8 or 12.x — must match your PyTorch build |
| Disk space | ≥ 5 GB free (model weights + cache) |

---

## Environment Setup

### Step 0 — Prerequisites

Before running the setup script, install the following manually:

#### Python 3.9 – 3.11

This project requires Python **3.9, 3.10, or 3.11**. Python 3.12+ is not supported due to TensorFlow and TTS compatibility constraints.

Download from: https://www.python.org/downloads/

> During installation on Windows, check **"Add Python to PATH"** and select **"Customize installation" → enable pip**.

Verify after installing:
```bat
python --version
```

#### eSpeak NG (required by the Kokoro TTS engine)

eSpeak NG is a text-to-speech backend that Kokoro depends on at runtime. It must be installed **before** running `setup_env.bat`.

Download the Windows installer (`.msi`) from the official releases page:
**https://github.com/espeak-ng/espeak-ng/releases**

Download the file named `espeak-ng-X.XX.X-x64.msi` (latest release), run it, and follow the installer. The default install path (`C:\Program Files\eSpeak NG\`) is fine.

> After installation, confirm it works:
> ```bat
> espeak-ng --version
> ```
> If the command is not found, add `C:\Program Files\eSpeak NG\` to your system PATH manually via System Properties → Environment Variables.

---

### Step 1 — Clone the repository

```bat
git clone https://github.com/Jerry-6lg6/AI-powered-Agent-For-healthcare.git
cd AI-powered-Agent-For-healthcare
```
Possible solution when execution of scrip is unable on this system => https://stackoverflow.com/questions/4037939/powershell-says-execution-of-scripts-is-disabled-on-this-system
---

### Step 2 — Run the setup script

Open **Command Prompt as Administrator** (right-click Start → "Command Prompt (Admin)"), then run:

```bat
:: If you have an NVIDIA GPU with CUDA 12.1 (recommended):
setup_env.bat

:: CUDA 11.8:
setup_env.bat cu118

:: No GPU / CPU only:
setup_env.bat cpu
```

The script runs the following steps automatically and will stop with a clear `[ERROR]` message if any step fails:

| Step | What it does |
|------|-------------|
| Python version check | Confirms Python 3.9–3.11 is active; exits if not |
| Create `.venv\` | Creates an isolated virtual environment in the project root |
| Upgrade pip | Ensures pip is up to date inside the venv |
| Install PyAudio | Installs PyAudio; skips silently if already present |
| Install PyTorch | Installs the correct CPU or CUDA build based on the argument passed |
| Install dependencies | Runs `pip install -r requirements.txt`; checks the file exists first |

> **Note:** Administrator privileges are required because the `keyboard` library registers a system-wide input hook.

---

### Step 3 — Activate and run

```bat
:: Activate the virtual environment
.venv\Scripts\activate

:: Verify the venv Python is active (prompt should show (.venv))
where python

:: Start the therapy session agent
python task.py

:: Or launch the staff dashboard
python dashboard.py
```

---

### Check your CUDA version (GPU users only)

Before choosing a CUDA build, confirm your driver supports it:

```bat
nvidia-smi
```

The top-right corner shows `CUDA Version: XX.X`. Use the table below to pick the right argument:

| nvidia-smi reports | Use |
|--------------------|-----|
| CUDA 12.1 or higher | `setup_env.bat` (default) |
| CUDA 11.8 – 12.0 | `setup_env.bat cu118` |
| No NVIDIA GPU | `setup_env.bat cpu` |

---

## Running the Application

```bash
# Start the therapy session agent (main entry point)
python task.py

# Launch the staff desktop dashboard (PyQt6 GUI)
python dashboard.py
```

---

## Torch Model Cache

By default, Whisper and Silero VAD model weights are downloaded and cached in `<project_root>/torch_cache/`, which is **created automatically** on first run. No manual directory creation is needed on any platform.

To override the cache location, set the `TORCH_HOME` environment variable before launching:

```bash
# Linux / macOS
export TORCH_HOME=/data/my_torch_cache
python task.py

# Windows CMD
set TORCH_HOME=D:\my_torch_cache
python task.py

# Windows PowerShell
$env:TORCH_HOME = "D:\my_torch_cache"
python task.py
```

---

## Project Structure

```
AI-powered-Agent-For-healthcare/
├── task.py                      # Main entry — therapy session controller
├── speech_Recognition.py        # ASR: Whisper / faster-whisper + Silero VAD
├── speech_synthesis.py          # TTS: Kokoro + WSOLA time-stretch playback
├── classifier.py                # Semantic response scorer (BGE-large + FAISS)
├── tools.py                     # Audio enhancement plugins (SpectralSubtraction, etc.)
├── emergency.py                 # Alert escalation logic
├── report.py                    # Session report generator
├── dashboard.py                 # PyQt6 staff desktop GUI
├── database/
│   ├── db.py                    # DatabaseManager (SQLite)
│   └── models.py                # ORM models
├── audio/
│   └── question/                # Pre-synthesised therapy audio prompts (.wav)
├── bge-large-en-v1.5/           # Local BGE embedding model weights (bundled)
├── requirements.txt             # Python dependencies
├── setup_env.sh                 # Linux/macOS one-shot setup script
└── setup_env.bat                # Windows one-shot setup script
```

---

## Troubleshooting

| Symptom | Solution |
|---------|----------|
| `espeak-ng` not found at runtime | Install from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases) and add install path to system PATH |
| `No module named 'pyaudio'` | Re-run `setup_env.bat cpu` as Administrator; or manually `pip install PyAudio` inside the activated venv |
| `[ERROR] Python X.XX detected` from script | Install Python 3.9–3.11 from python.org; ensure "Add to PATH" was checked during install |
| `此时不应有` error in CMD | Ensure you are using the latest `setup_env.bat` — older versions had encoding issues |
| `[ERROR] requirements.txt not found` | Run the script from the project root folder, not from another directory |
| CUDA out of memory during inference | Switch to `faster_whisper` with `compute_type="int8"` in `task.py` |
| Slow transcription | Reduce `model_size` from `"medium"` to `"small"` in `task.py` |
| `torch.hub` download hangs | Set `TORCH_HOME` to a path with ≥ 5 GB free space; retry with a stable network connection |
| `where python` shows system path after activation | Close and reopen CMD as Administrator, then re-run `.venv\Scripts\activate` |
| `KeyError: 'TORCH_HOME'` | Replace `speech_Recognition.py` with the patched version included in this repository |

---

## Notes

- The `bge-large-en-v1.5/` folder contains model weights committed directly to the repository (~1.3 GB). Cloning requires Git LFS or sufficient bandwidth.
- All audio prompts in `audio/question/` are pre-synthesised `.wav` files. To regenerate them, run `speech_synthesis.py` directly.
- The application is designed for local/on-premise deployment. No internet connection is required after initial model download.
