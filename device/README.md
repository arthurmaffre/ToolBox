# Device Module — README

## 📌 Overview
This module provides a **clean, professional, and frictionless way** to select the best available device for your PyTorch projects. It detects and prioritizes CUDA, MPS (Apple Silicon), or CPU depending on your environment.

## ⚙️ Features
- **Automatic device detection** with multiple strategies.
- **Flexible selection modes**:
  - `auto`: CUDA > MPS > CPU
  - `cuda`: CUDA > CPU
  - `cpu`: CPU only (forces CPU usage)
- **Plug-and-play**: Easily integrate into any PyTorch project.

## 🚀 Installation
Place `device.py` in your **ToolBox** repository under the appropriate module folder.

## 🛠️ Usage
```python
from device import get_device

device = get_device("auto")  # Options: "auto", "cuda", "cpu"
print(f"Using device: {device}")
```

### 🔄 Modes Explained
- **`auto`** → Uses CUDA if available, else MPS (Mac), else CPU.
- **`cuda`** → Uses CUDA if available, otherwise falls back to CPU.
- **`cpu`** → Forces CPU regardless of hardware.

## ✅ Best Practices
- Call `get_device()` **once** at the start of your script.
- Pass the returned `device` to your models and tensors using `.to(device)`.
- Combine with your training loop for seamless execution.

## 📂 Example Integration
```python
model = MyModel().to(device)
inputs = inputs.to(device)
outputs = model(inputs)
```

## 🧠 Why this module?
This small utility is part of your **ToolBox** to:
- **Avoid repeating boilerplate code**.
- **Simplify device handling across projects**.
- **Ensure maximum performance** by using the best hardware available automatically.