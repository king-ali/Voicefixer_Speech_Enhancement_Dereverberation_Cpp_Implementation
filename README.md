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

## Cpp inference and Usage

Please note that this repository is intended primarily for testing purposes, as the discriminator model is not included. However, you can perform inference in Python by running the `test.py` file.

For more efficient inference times, we've implemented support for inference in C++. Below are the steps to set it up:

1. **Download Torch C++ Installation**: Visit [pytorch.org](https://pytorch.org/get-started/locally/) to download the Torch C++ installation.

2. **Extract Torch**: Unzip the downloaded package into a directory, e.g., `libtorch-shared-with-deps-2.2.1+cpu`.

3. **Create Necessary Files**: Create the following files in your project directory:
   - `main.cpp`
   - `CMakeLists.txt`

4. **main.cpp**:
    ```bash
    // import libraries
    #include <iostream> 
    #include <torch/torch.h> 
    int main()
    { torch::manual_seed(0); // set manual seed
     torch::Tensor x = torch::randn({2,3}); // create torch random tensor
     std::cout << x;} // print tensor
    ```

5. **CMakeLists.txt**:
    ```bash
    cmake_minimum_required(VERSION 3.0)
    # project name
    project(debugfunc) 
    
    # define path to the libtorch extracted folder 
    set(CMAKE_PREFIX_PATH /mnt/d/RIR_Estimation/voicefixer/voicefixer/test/inference/) # ADD YOUR PATH HERE
    
    # find torch library and all necessary files
    find_package(Torch REQUIRED) 
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}") 
    
    # executable to add that we want to compile and run 
    add_executable(debugfunc main.cpp) # link torch libraries to our executable target_link_libraries(debugfunc "${TORCH_LIBRARIES}") 
    
    set_property(TARGET debugfunc PROPERTY CXX_STANDARD 17)

    ```

6. **Build and Run**:
   - Navigate to your project directory.
   - Use sudo if necessary.
   ```bash
   mkdir build
   cd build
   cmake ..
   make 
   ./debugfunc

7. **Code for reading mel from folder**:
     ```bash
    #include <iostream>
    #include <fstream>
    #include <filesystem>
    #include <torch/torch.h>    
    namespace fs = std::filesystem;
    using namespace std;
    int main() {
        string folder_path = "mel"; // Path to the folder
        int total_size = 0;
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.path().extension() == ".pt") {
                try {
                    // Open the file for reading
                    ifstream file(entry.path(), ios::binary);
                    if (!file.is_open()) {
                        cerr << "Failed to open file: " << entry.path() << endl;
                        continue;
                    }
                    // Read the file size
                    file.seekg(0, ios::end);
                    streamsize size = file.tellg();
                    file.seekg(0, ios::beg);
   
                    // Increment total size
                    total_size += size;
                    // Output file name and size
                    cout << "File: " << entry.path().filename() << ", Size: " << size << " bytes" << endl;
                    // Close the file
                    file.close();
                } catch (const std::exception& e) {
                    cerr << "Failed to read file: " << entry.path() << ", Error: " << e.what() << endl;
                }
            }
        }
    
        cout << "Total size of .pt files: " << total_size << " bytes" << endl;
        return 0;
    }

    ```

9. **Converting vocoder model to torchscript**:

     ```bash
    example_input = torch.randn(1, 128, 106) 
    traced_script_module = torch.jit.trace(self.model, example_input) traced_script_module.save("torchscript_model.pt") 
    print("torchscript_model saved") 
    for p in self.model.parameters():
    p.requires_grad = False
    ```
10. **Load tensor from C++**:
    
     ```bash
    #include <iostream>
    #include <fstream>
    #include <vector>
    #include <torch/torch.h>
    std::vector<char> get_the_bytes(const std::string& filename) {
        std::ifstream input(filename, std::ios::binary);
        std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));
        input.close();
        return bytes;
    }    
    int main()
    {
        std::vector<char> f = get_the_bytes("/mnt/d/RIR_Estimation/voicefixer/voicefixer/test/inference/build/mel/denoised_mel_44100.pt");
        torch::IValue x = torch::pickle_load(f);
        torch::Tensor tensor = x.toTensor();
       
        std::cout << "Tensor shape: ";
        for (int64_t dim : tensor.sizes()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "Tensor size: " << tensor.numel() << std::endl;   
        std::cout << tensor << std::endl;
    }
    ```

   11.  **Cpp code to process each mel**:
       The following code passes all the mels from the model and calculates the time taken by the model to process each mel. Here, one mel is one second long
     ```bash

    #include <iostream>
    #include <vector>
    #include <torch/torch.h>
    #include <torch/script.h>
    #include <fstream>
    #include <string>
    #include <cmath>
    #include <cstdint>
    #include <cstring>
    #include <cassert>
    #include <cstdio>
    #include <filesystem>
    #include <chrono>
    #include <algorithm>
    #include <future>
    #include <onnxruntime_cxx_api.h>
    namespace fs = std::filesystem;
    using namespace std::chrono;
    // Function to read the contents of a file and return them as a vector of bytes
    std::vector<char> get_the_bytes(const std::string& filename) {
        std::ifstream input(filename, std::ios::binary); // Open the file specified by 'filename' in binary mode
        std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>())); // Read the contents of the file into the vector
        input.close(); // Close the file stream
        return bytes; // Return the vector containing the file's bytes
    }
    // Function to write audio data to a WAV file
    void write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate) {
        FILE* file = fopen(filename.c_str(), "wb");
        assert(file);    
        // Write WAV header
        const int16_t fmt_chunk_size = 16;
        const int16_t audio_format = 1; // PCM
        const int16_t num_channels = 1;
        const int32_t byte_rate = sample_rate * sizeof(int16_t);
        const int16_t block_align = sizeof(int16_t);
        const int16_t bits_per_sample = 16;
        const int32_t data_size = samples.size() * sizeof(int16_t);
        const int32_t total_size = 36 + data_size;
    
        fwrite("RIFF", 1, 4, file);
        fwrite(&total_size, 4, 1, file);
        fwrite("WAVE", 1, 4, file);
        fwrite("fmt ", 1, 4, file);
        fwrite(&fmt_chunk_size, 2, 1, file);
        fwrite(&audio_format, 2, 1, file);
        fwrite(&num_channels, 2, 1, file);
        fwrite(&sample_rate, 4, 1, file);
        fwrite(&byte_rate, 4, 1, file);
        fwrite(&block_align, 2, 1, file);
        fwrite(&bits_per_sample, 2, 1, file);
        fwrite("data", 1, 4, file);
        fwrite(&data_size, 4, 1, file);
    
    
        // Write audio data
        for (size_t i = 0; i < samples.size(); ++i) {
            int16_t sample = static_cast<int16_t>(samples[i] * 32767);
            fwrite(&sample, sizeof(int16_t), 1, file);
        }
    
    
        fclose(file);
    }    
    int main() {
        // Load the TorchScript model
        torch::jit::script::Module module = torch::jit::load("torchscript_model.pt");
    
    
        // Create a vector to hold combined audio samples
        std::vector<float> combined_audio;
    
    
        // Create a vector to hold file paths
        std::vector<std::string> file_paths;
    
    
        // Iterate over all files in the "mel" folder and store their paths
        for (const auto& entry : fs::directory_iterator("mel")) {
            file_paths.push_back(entry.path().string());
        }
    
    
        // Sort file paths to ensure deterministic order
        std::sort(file_paths.begin(), file_paths.end());
    
    
        // Iterate over sorted file paths
        for (const auto& file_path : file_paths) {
    
    
            // Read the tensor from the file
            std::vector<char> f = get_the_bytes(file_path);
    
    
            // Deserialize the tensor
            torch::IValue x = torch::pickle_load(f);
            torch::Tensor tensor = x.toTensor();
    
    
            // Perform inference using the loaded tensor
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(tensor);
    
    
            // Start the timer for each inference
            auto inference_start = high_resolution_clock::now();
    
    
            // Disable gradient computation during inference
            {
                torch::NoGradGuard no_grad;
    
    
                at::Tensor output_tensor = module.forward(inputs).toTensor();
                // Convert the output tensor to a vector of floats
                std::vector<float> output_audio(output_tensor.data<float>(), output_tensor.data<float>() + output_tensor.numel());
    
    
                // Append the output audio to the combined audio vector
                combined_audio.insert(combined_audio.end(), output_audio.begin(), output_audio.end());
            }
    
    
            // Stop the timer for each inference
            auto inference_end = high_resolution_clock::now();
            auto inference_duration = duration_cast<milliseconds>(inference_end - inference_start);
    
    
            // Print the time taken for each inference
            std::cout << "Inference time for " << file_path << ": " << inference_duration.count() << " milliseconds" << std::endl;
        }
    
    
        // Save the combined output audio to a WAV file
        write_wav("output_audio.wav", combined_audio, 44100);
    
    
        return 0;
    }

    ```
