# Cognitive Rehabilitation Assistant

An AI-powered voice interaction system for cognitive rehabilitation tasks (e.g., orientation therapy).

---

## Requirements

- **Python**: 3.9 – 3.11
- **GPU**: NVIDIA GPU with CUDA recommended
- **CUDA Toolkit**: 11.8 or 12.x (match your PyTorch build)

---

## Environment Setup

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install PyTorch with CUDA

Choose the command matching your CUDA version from https://pytorch.org/get-started/locally/. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies via requirements.txt

All dependencies (including `torch~=2.5.1+cu121`) are pinned in `requirements.txt`. Install everything with:

```bash
pip install -r requirements.txt
```

> **`pyaudio` on Windows**: `pyaudio` cannot be installed directly via pip on Windows. Install it separately using `pipwin` **before** running the above command:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```
> On Ubuntu/Debian, install the system dependency first:
> ```bash
> sudo apt-get install portaudio19-dev
> ```
> `PyAudio` is already included in `requirements.txt` and will install correctly on Linux/macOS once the system dependency is present.

### 4. Set the Torch cache directory (Windows)

`speech_Recognition.py` sets the torch hub cache to `C:\torch_cache`. Create this directory before running:

```bash
mkdir C:\torch_cache
```

Or update the path at line 14 of `speech_Recognition.py` to a directory of your choice:

```python
os.environ['TORCH_HOME'] = r'C:\torch_cache'
```

---

## Running the Application

```bash
python task.py
```

