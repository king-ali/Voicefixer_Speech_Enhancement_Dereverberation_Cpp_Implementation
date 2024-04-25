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


