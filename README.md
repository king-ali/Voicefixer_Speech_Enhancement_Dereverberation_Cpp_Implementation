# Voicefixer_speech_enhancement_dereverberation_cpp_implementation

VoiceFixer is an advanced tool designed to restore degraded human speech signals. It excels in handling various issues such as noise, reverberation, low resolution, and clipping effects, all within a single model.

## Features

- **Robust Restoration**: Capable of restoring speech signals regardless of the severity of degradation.
- **Versatile**: Handles noise, reverberation, low resolution (2kHz to 44.1kHz), and clipping effects (threshold from 0.1 to 1.0) within one comprehensive model.
- **Easy Installation**: Quick setup with Git clone and pip install.

## Installation

You can easily get started with VoiceFixer by following these simple steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/haoheliu/voicefixer.git
    ```

2. Install using pip:
    ```bash
    pip install git+https://github.com/haoheliu/voicefixer.git
    ```

## Usage

Please note that this repository is intended primarily for testing purposes, as the discriminator model is not included. However, you can perform inference in Python by running the `test.py` file.

For more efficient inference times, we've implemented support for inference in C++. Below are the steps to set it up:

1. **Download Torch C++ Installation**: Visit [pytorch.org](https://pytorch.org/get-started/locally/) to download the Torch C++ installation.

2. **Extract Torch**: Unzip the downloaded package into a directory, e.g., `libtorch-shared-with-deps-2.2.1+cpu`.

3. **Create Necessary Files**: Create the following files in your project directory:
   - `main.cpp`
   - `CMakeLists.txt`

4. **Build and Run**:
   - Navigate to your project directory.
   - Use sudo if necessary.
   ```bash
   mkdir build
   cd build
   cmake ..
   make 
   ./debugfunc
