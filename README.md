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

---

## Google Colab Setup

Google Colab provides free GPU access and is a convenient way to run this project without a local CUDA-capable machine.

### Requirements

- A Google account with access to [Google Colab](https://colab.research.google.com/)
- A GPU runtime (T4 or better recommended) — go to **Runtime → Change runtime type → GPU**

> **Note on audio I/O**: Colab does not support real-time microphone input or audio playback via `sounddevice` or `pygame` out of the box. You will need to use file-based audio input/output or browser-based recording workarounds (see the notes below each step where relevant).

---

### Step 1: Enable GPU Runtime

In Colab, before running any cells:

1. Click **Runtime** in the top menu.
2. Select **Change runtime type**.
3. Set **Hardware accelerator** to **GPU** (T4 recommended).
4. Click **Save**.

---

### Step 2: Clone the Repository

```python
!git clone https://github.com/Jerry-6lg6/AI-powered-Agent-For-healthcare.git
%cd AI-powered-Agent-For-healthcare
```

### Step 3: Install PyTorch with CUDA

Colab typically comes with a compatible PyTorch version pre-installed. Verify first:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

If CUDA is not available or you need a specific version, reinstall:

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 4: Install System Dependencies

`PyAudio` requires `portaudio` at the system level, which must be installed before `requirements.txt`:

```python
!apt-get install -y portaudio19-dev
```

---

### Step 5: Install Python Dependencies

All dependencies are pinned in `requirements.txt`. Install them in one command:

```python
!pip install -r requirements.txt
```

> **Note on `torch` in `requirements.txt`**: The file pins `torch~=2.5.1+cu121`. Colab may already have a compatible PyTorch version — if the install raises a conflict, you can comment out the `torch` line in `requirements.txt` and rely on the pre-installed version, provided CUDA is available (verified in Step 3).

---

### Step 6: Configure the Torch Cache Directory

`speech_Recognition.py` sets the torch cache to `C:\torch_cache` by default, which is a Windows path. Override it for Colab:

```python
import os
os.environ['TORCH_HOME'] = '/content/torch_cache'
!mkdir -p /content/torch_cache
```

Run this cell **before** importing or running any model code.

---

### Step 7: Upload or Mount Input Files

If your application requires audio input files, either upload them directly or mount your Google Drive:

```python
# Option A: Upload files manually
from google.colab import files
uploaded = files.upload()  # Opens a file picker in the browser

# Option B: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Your files will be accessible at /content/drive/MyDrive/
```

---

### Step 8: Run the Application

```python
!python task.py
```

> **Audio playback**: If the application attempts to play audio via `pygame` or `sounddevice`, it may fail silently in Colab. To hear output, save audio to a `.wav` file and play it back in the notebook:
> ```python
> from IPython.display import Audio
> Audio('/content/output.wav')
> ```

---

### Colab-Specific Notes

| Feature | Local | Google Colab |
|---|---|---|
| GPU support | Requires NVIDIA GPU + CUDA | Built-in (T4 by default) |
| Microphone input | Supported via `sounddevice` | Requires browser workaround |
| Audio playback | Supported via `pygame` | Use `IPython.display.Audio` |
| File persistence | Local filesystem | Files reset each session; use Drive to persist |
| Torch cache | `C:\torch_cache` (Windows) | Set to `/content/torch_cache` |

