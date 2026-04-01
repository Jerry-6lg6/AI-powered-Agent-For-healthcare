# ✦ Jennet — AI-Powered Healthcare Cognitive Assessment System

> An autonomous AI agent that conducts real-time cognitive orientation assessments for patients in hospital and care home settings — combining neural speech recognition, semantic AI classification, emergency detection, and automated clinical PDF reporting.

**University of Liverpool · BEng Computer Science and Electronic Engineering · Final Year Project 2026**  
**Authors:** Joiakim & Rami · **Supervisor:** Mario Gianni · **Assessor:** Jeremy Smith

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Installation Guide](#installation-guide)
- [Running the System](#running-the-system)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Technology Stack](#technology-stack)
- [References](#references)

---

## Overview

Jennet is a patient-facing AI assistant that autonomously conducts cognitive orientation assessments — asking patients about the current date, month, year, and weekday. It adapts to each patient's performance, detects emergencies, generates clinical PDF reports, and provides staff with a real-time monitoring dashboard.

The system runs **entirely offline** on local hardware. No patient data leaves the device.

### What Jennet does

1. Greets the patient and checks if they are ready
2. Asks orientation questions (date, month, year, weekday)
3. Classifies responses using a 3-stage ensemble AI classifier
4. Adapts speed and difficulty based on patient performance
5. Gives hints before revealing answers on incorrect responses
6. Detects emergencies and triggers an immediate alarm and staff alert
7. Saves all responses, scores, and timing data to a local SQLite database
8. Auto-generates a 7-section clinical PDF report after every session

---

## System Requirements

| Requirement | Minimum |
|---|---|
| OS | Windows 10 / 11 (64-bit) |
| GPU | NVIDIA GPU with CUDA support |
| VRAM | 4 GB |
| RAM | 16 GB |
| Disk Space | 20 GB free |
| Python | 3.11.x |
| CUDA | 12.1 |
| Microphone | Required |
| Speakers | Required |
| Internet | Required for initial setup only |

---

## Project Structure

```
AI_Powered HCA\
│
├── task.py                  ← Main entry point — run this to start a session
├── dashboard.py             ← Staff desktop monitoring dashboard (PyQt6)
├── classifier.py            ← 3-stage ensemble classifier (BGE + FAISS)
├── speech_Recognition.py    ← VAD-controlled recording + Faster-Whisper ASR
├── speech_synthesis.py      ← Kokoro TTS + WSOLA time-stretching
├── emergency.py             ← Emergency alert system
├── report.py                ← Clinical PDF report generator
├── tools.py                 ← Audio denoising plugins (SpectralSubtraction)
├── personal_task.py         ← Autobiographical memory task (optional)
├── test.py                  ← ASR evaluation scripts
│
├── database\
│   ├── db.py                ← DatabaseManager — all SQLite queries
│   ├── models.py            ← Table schema definitions
│   └── hca.db               ← SQLite database (auto-created on first run)
│
├── audio\
│   ├── question\            ← Generated TTS audio files (auto-populated)
│   └── beep\
│       └── beep.wav         ← Beep sound played before recording
│
├── reports\                 ← Auto-generated PDF reports (auto-created)
│
├── requirements.txt         ← Python dependencies
└── README.md                ← This file
```

---

## Installation Guide

### Step 1 — Install Python 3.11

1. Download from: https://www.python.org/downloads/release/python-3119/
2. Select **Windows installer (64-bit)**
3. Run the installer — **tick "Add Python to PATH"** before installing
4. Verify in Command Prompt:
```
python --version
```
Expected output: `Python 3.11.x`

---

### Step 2 — Install CUDA 12.1

1. Download from: https://developer.nvidia.com/cuda-12-1-0-download-archive
2. Select: Windows → x86_64 → 11 → exe (local)
3. Run installer → choose **Express installation**
4. Restart the PC
5. Verify:
```
nvcc --version
```
Expected output: `release 12.1`

---

### Step 3 — Place the project files

Copy the project folder to your machine, for example:
```
C:\Users\[username]\Desktop\AI_Powered HCA
```

---

### Step 4 — Update the torch cache path

Open `speech_Recognition.py` and find line 17:
```python
os.environ['TORCH_HOME'] = r'C:\Users\joiak\torch_cache'
```
Change `joiak` to your Windows username:
```python
os.environ['TORCH_HOME'] = r'C:\Users\[your-username]\torch_cache'
```
Then create the folder:
```
mkdir C:\Users\[your-username]\torch_cache
```

---

### Step 5 — Create a virtual environment

Open Command Prompt and run:
```
cd "C:\Users\[username]\Desktop\AI_Powered HCA"
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` at the start of the prompt.

---

### Step 6 — Install PyTorch with CUDA

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
This takes 5–10 minutes. Verify CUDA works:
```
python -c "import torch; print(torch.cuda.is_available())"
```
**Must print `True`.** If it prints `False`, CUDA is not set up correctly — reinstall CUDA from Step 2.

---

### Step 7 — Install Python dependencies

```
pip install faster-whisper
pip install kokoro
pip install sounddevice
pip install soundfile
pip install pyaudio
pip install PyQt6
pip install reportlab
pip install sentence-transformers
pip install faiss-gpu
pip install transformers
pip install librosa
pip install scipy
pip install pygame
pip install numpy
pip install keyboard
```

> **If pyaudio fails to install**, run:
> ```
> pip install pipwin
> pipwin install pyaudio
> ```

---

### Step 8 — Download AI models (one-time)

These commands download the AI models on first use. Run each one and wait for it to complete.

**BGE-large embedding model (~1.3 GB):**
```
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5'); AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')"
```

**Kokoro TTS model:**
```
python -c "from kokoro import KPipeline; KPipeline(lang_code='a', device='cuda')"
```

**Faster-Whisper medium model (~1.5 GB):**
```
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cuda', compute_type='int8')"
```

**Silero VAD** downloads automatically on first run.

---

### Step 9 — Initialise the database

```
python -c "from database.db import DatabaseManager; DatabaseManager(); print('Database ready')"
```
Expected output: `Database ready`

---

### Step 10 — Verify the installation

```
python -c "
import torch
import faster_whisper
import kokoro
import sounddevice
import PyQt6
import reportlab
import faiss
import librosa
import pygame
print('All imports successful')
print('CUDA available:', torch.cuda.is_available())
"
```
Expected output:
```
All imports successful
CUDA available: True
```

---

## Running the System

### Start a patient session

```
cd "C:\Users\[username]\Desktop\AI_Powered HCA"
venv\Scripts\activate
python task.py
```

You will be prompted to enter:
```
Enter patient name: [e.g. Margaret Thompson]
Enter patient location (e.g. Ward 3, Room 5): [e.g. Ward 2, Room 4]
```

The system will then load all AI models (~30–60 seconds on first run) and begin the session.

---

### Start the staff dashboard

```
venv\Scripts\activate
python dashboard.py
```

The dashboard shows:
- Live session status
- All patients and their session history
- Score trend charts
- Active emergency alerts
- PDF report downloads

---

## How It Works

### Conversation flow

```
INIT → READY_CHECK → INTERACT_MAIN → INTERACT_BREAKDOWN → NEXT → FINISH / EXIT
```

1. **INIT** — Jennet introduces herself and plays instructions
2. **READY_CHECK** — Waits for patient to say "yes" to begin (3 silences = exit)
3. **INTERACT_MAIN** — Asks the main date question
4. **INTERACT_BREAKDOWN** — If wrong, asks sub-questions: year → month → day → weekday
5. **NEXT** — Moves to next question
6. **FINISH** — Plays closing message, saves session, generates PDF
7. **EXIT** — Triggered by emergency, stop request, or prolonged silence

### Classification pipeline

Each patient response goes through three stages:

| Stage | Method | Cost |
|---|---|---|
| Stage 1 | Exact whole-word string match | Zero — instant |
| Stage 2 | Normalised keyword match (handles "fourteenth", "twenty twenty six", "14th") | Zero — instant |
| Stage 3 | BGE-large semantic embedding + FAISS similarity search | GPU — only fires if stages 1 and 2 fail |

Special states (emergency, stop, don't know, repeat, require) always go through FAISS regardless of stage.

### Detection thresholds

| State | Threshold | Minimum words |
|---|---|---|
| Emergency | 0.75 cosine similarity | 2 words |
| Don't know | 0.85 cosine similarity | 1 word |
| Stop | 0.90 cosine similarity | 2 words |
| Repeat | 0.80 cosine similarity | 1 word |
| Require (food/drink) | 0.82 cosine similarity | 2 words |

---

## Key Features

### Adaptive conversation
- Automatically slows speech speed if patient scored low in their last session
- Increases encouragement after consecutive wrong answers
- After 3 wrong answers in a row, slows down further and tells the patient

### Hint system
- Before revealing the answer, Jennet gives a one-word clue
- Short answers (e.g. "March") → *"It starts with the letter M."*
- The patient gets one more attempt after the hint

### Silence handling
- 3 consecutive silences anywhere in the session ends it gracefully
- Jennet says: *"I will end the orientation now as you have not responded for a while."*

### Emergency detection
- Triggered by 40+ emergency phrases ("help me", "I'm in pain", "call a nurse")
- Single-word responses ("now", "help") cannot trigger emergency to prevent false alarms
- On trigger: continuous two-tone alarm, TTS announcement of patient name and location, database log, waits for staff to press Enter

### Clinical PDF reports
Auto-generated after every session with 7 sections:
1. Patient information
2. Session overview
3. Score summary
4. Question and response detail (with confidence scores, thinking time, last session comparison)
5. Emergency events
6. Session history trend
7. Clinical notes and recommendations

### Confidence scoring
Every response is scored 0.0–1.0:
- `1.0` = exact match
- `0.95` = normalised keyword match
- `0.55–0.94` = semantic similarity match
- Below `0.55` = incorrect

### Session comparison
The PDF report shows a ▼ symbol when a patient answered a question correctly last session but not this session — a potential sign of cognitive decline.

---

## Configuration

Key constants in `task.py`:

| Constant | Default | Description |
|---|---|---|
| `SPEED` | `1` | Normal playback speed |
| `SPEED_SLOW` | `0.85` | Slow playback speed for struggling patients |
| `WAIT_TIME` | `1` | Minutes to wait when patient says not ready |
| `CHANGEABLE` | `True` | Allow TTS re-synthesis |
| `UNCHANGEABLE` | `True` | Lock voice parameters for consistency |

Confidence filter threshold in `speech_Recognition.py`:

| Constant | Default | Description |
|---|---|---|
| `LOGPROB_THRESHOLD` | `-1.0` | Reject transcriptions below this avg_logprob |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA available: False` | Reinstall CUDA 12.1, restart PC |
| `No module named X` | Run `pip install X` with venv active |
| Torch cache errors | Update `TORCH_HOME` path in `speech_Recognition.py` |
| Microphone not detected | Check Windows sound settings, set mic as default recording device |
| PyAudio install fails | Use `pipwin install pyaudio` |
| `hca.db not found` | Run `python -c "from database.db import DatabaseManager; DatabaseManager()"` |
| Models downloading every run | Check `TORCH_HOME` path exists and is writable |
| Voice sounds distorted | Ensure `UNCHANGEABLE = True` in `task.py` |
| Emergency false triggers | Check word count guard is active in `classifier.py` |

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Speech detection | Silero VAD | Real-time voice activity detection |
| Speech recognition | Faster-Whisper (int8 CUDA) | Audio to text transcription |
| Text embeddings | BGE-large-en-v1.5 | Semantic text representation |
| Similarity search | FAISS IndexFlatIP | Fast nearest-neighbour search |
| Speech synthesis | Kokoro TTS (af_bella) | Natural text to speech |
| Time stretching | WSOLA | Speed adjustment without pitch change |
| Audio denoising | Spectral Subtraction | Background noise reduction |
| Conversation control | Custom FSM (task.py) | Session flow and state management |
| Database | SQLite | Local patient data persistence |
| PDF generation | ReportLab | Clinical report creation |
| Staff dashboard | PyQt6 | Windows desktop monitoring app |
| Deep learning | PyTorch 2.5 + CUDA 12.1 | GPU-accelerated inference |

---

## References

1. Radford, A. et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML 2023.*
2. Xiao, S. et al. (2023). C-Pack: BGE-large-en-v1.5 General Text Embeddings. *arXiv:2309.07597.*
3. Silero Team (2021). Silero VAD: Pre-trained Enterprise-Grade Voice Activity Detector. *GitHub.*
4. Johnson, J. et al. (2019). Billion-scale similarity search with GPUs. *IEEE Trans. Big Data — FAISS.*
5. Folstein, M.F. et al. (1975). Mini-mental state examination. *J. Psychiatric Research, 12(3).*
6. Kokoro TTS (2024). Lightweight Neural TTS. *HuggingFace hexgrad/Kokoro-82M.*
7. NHS England (2023). NHS Long Term Workforce Plan. *NHSE Publications.*

---

## Acknowledgements

Thanks to supervisor Mario Gianni for guidance throughout this project. All patient data used during testing was entirely synthetic — no real patient data was collected or stored at any point. The system runs fully offline on local hardware, ensuring data security suitable for clinical environments.

---

*Jennet — AI Healthcare Assistant · University of Liverpool · 2026*
