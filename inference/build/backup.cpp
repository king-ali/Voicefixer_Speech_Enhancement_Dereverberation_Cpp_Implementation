// #include <iostream>
// #include <fstream>
// #include <filesystem>
// #include <torch/torch.h>

// namespace fs = std::filesystem;
// using namespace std;

// int main() {
//     string folder_path = "mel"; // Path to the folder
//     int total_size = 0;
//     int count = 0;

//     for (const auto& entry : fs::directory_iterator(folder_path)) {
//         if (entry.path().extension() == ".pt") {
//             try {
//                 // Open the file for reading
//                 ifstream file(entry.path(), ios::binary);
//                 if (!file.is_open()) {
//                     cerr << "Failed to open file: " << entry.path() << endl;
//                     continue;
//                 }

//                 // Read the file size
//                 file.seekg(0, ios::end);
//                 streamsize size = file.tellg();
//                 file.seekg(0, ios::beg);

//                 // Increment total size
//                 total_size += size;

//                 // Output file name and size
//                 cout << "File: " << entry.path().filename() << ", Size: " << size << " bytes" << endl;
//                 count = count + 1;

//                 // Close the file
//                 file.close();
//             } catch (const std::exception& e) {
//                 cerr << "Failed to read file: " << entry.path() << ", Error: " << e.what() << endl;
//             }
//         }
//     }

//     cout << "Total size of .pt files: " << total_size << " bytes" << endl;
//     cout << "Total count is: " << count << " files" << endl;
    


//     return 0;
// }

// // // https://freedium.cfd/https://towardsdatascience.com/torch-and-torchvision-c-installation-and-debugging-on-linux-263676c38fa2


// #include <iostream>
// #include <torch/script.h>

// int main() {
//     // Load the TorchScript model
//     torch::jit::script::Module module = torch::jit::load("torchscript_model.pt");

//     // Create input tensor
//     std::vector<float> input_data(128 * 101);  // Adjusted input size according to (1, 128, 101)
//     torch::Tensor input_tensor = torch::from_blob(input_data.data(), {1, 128, 101});

//     // Perform inference
//     std::vector<torch::jit::IValue> inputs;
//     inputs.push_back(input_tensor);
//     at::Tensor output = module.forward(inputs).toTensor();

//     // Access output tensor
//     std::cout << "Output tensor shape: " << output.sizes() << std::endl;

//     return 0;
// }








// #include <iostream>
// #include <filesystem>
// #include <torch/torch.h>
// #include <torch/script.h>
// #include <librosa/io.h>
// // #include <boost/python.hpp>


// namespace fs = std::filesystem;

// // Function to load Mel spectrogram from .pt file
// torch::Tensor load_mel_spectrogram(const std::string& file_path) {
//     return torch::load(file_path);
// }

// // Function to convert Mel spectrogram to audio using the model
// torch::Tensor mel_to_audio(const torch::jit::script::Module& model, const torch::Tensor& mel_spectrogram) {
//     // Perform inference using the model
//     torch::Tensor audio;
//     try {
//         // Assuming your model has a method named 'forward' for inference
//         auto output = model.forward({mel_spectrogram}).toTuple();
//         audio = output->elements()[0].toTensor();
//     } catch (const c10::Error& e) {
//         std::cerr << "Error during model inference: " << e.what() << std::endl;
//     }
//     return audio;
// }

// // Function to save audio as .wav file
// void save_audio(const torch::Tensor& audio, const std::string& output_path) {
//     // Convert tensor to numpy array and save as .wav using librosa
//     auto audio_np = audio.cpu().squeeze().contiguous().data<float>();
//     librosa::output::write_wav(output_path, audio_np, 22050); // Adjust sample rate if needed
// }

// int main() {
//     std::string mel_folder = "mel"; // Path to the folder containing .pt files
//     std::string output_folder = "output"; // Output folder for .wav files
//     int count = 0;

//     // Load the model
//     torch::jit::script::Module model;
//     try {
//         model = torch::jit::load("torchscript_model.pt"); // Load your TorchScript model
//     } catch (const c10::Error& e) {
//         std::cerr << "Error loading the model: " << e.what() << std::endl;
//         return 1;
//     }

//     // Iterate through .pt files in the folder
//     for (const auto& entry : fs::directory_iterator(mel_folder)) {
//         if (entry.path().extension() == ".pt") {
//             // Load Mel spectrogram
//             torch::Tensor mel_spectrogram = load_mel_spectrogram(entry.path());

//             // Perform inference
//             torch::Tensor audio = mel_to_audio(model, mel_spectrogram);

//             // Save audio as .wav file
//             std::string output_file_path = output_folder + "/" + entry.path().stem().string() + ".wav";
//             save_audio(audio, output_file_path);

//             ++count;
//         }
//     }

//     std::cout << "Total .wav files generated: " << count << std::endl;

//     return 0;
// }







// #include <torch/script.h>
// #include <iostream>
// #include <filesystem>
// #include <vector>
// #include <fstream>

// namespace fs = std::filesystem;

// // Function to read mel spectrograms from .pt files in a directory
// std::vector<torch::Tensor> readMelSpectFromFolder(const std::string& folderPath) {
//     std::vector<torch::Tensor> melSpectrograms;
//     for (const auto& entry : fs::directory_iterator(folderPath)) {
//         if (entry.path().extension() == ".pt") {
//             try {
//                 torch::Tensor melSpectrogram = torch::jit::load(entry.path().string()).toTensor();
//                 melSpectrograms.push_back(melSpectrogram);
//             } catch (const c10::Error& e) {
//                 std::cerr << "Error loading mel spectrogram from file: " << entry.path() << " - " << e.what() << std::endl;
//             }
//         }
//     }
//     return melSpectrograms;
// }

// int main() {
//     // Directory containing mel spectrograms
//     std::string melFolderPath = "mel";

//     // Load vocoder model
//     std::string modelPath = "torchscript_model.pt";
//     torch::jit::script::Module vocoderModel;
//     try {
//         vocoderModel = torch::jit::load(modelPath);
//     } catch (const c10::Error& e) {
//         std::cerr << "Error loading vocoder model from file: " << modelPath << " - " << e.what() << std::endl;
//         return 1;
//     }

//     // Read mel spectrograms
//     std::vector<torch::Tensor> melSpectrograms = readMelSpectFromFolder(melFolderPath);
//     if (melSpectrograms.empty()) {
//         std::cerr << "No mel spectrograms found in directory: " << melFolderPath << std::endl;
//         return 1;
//     }

//     // Perform inference and combine audio clips
//     std::vector<float> combinedAudio;
//     for (const auto& melSpectrogram : melSpectrograms) {
//         // Perform inference
//         std::vector<torch::jit::IValue> inputs;
//         inputs.push_back(melSpectrogram.unsqueeze(0)); // unsqueeze to add batch dimension
//         at::Tensor generatedAudioTensor = vocoderModel.forward(inputs).toTensor();

//         // Convert tensor to float array
//         std::vector<float> generatedAudio(generatedAudioTensor.data_ptr<float>(), generatedAudioTensor.data_ptr<float>() + generatedAudioTensor.numel());
        
//         // Append generated audio to combined audio
//         combinedAudio.insert(combinedAudio.end(), generatedAudio.begin(), generatedAudio.end());
//     }

//     // Save combined audio as .wav file
//     std::ofstream outputFile("combined_audio.wav", std::ios::binary);
//     if (!outputFile) {
//         std::cerr << "Error creating output file: combined_audio.wav" << std::endl;
//         return 1;
//     }
//     outputFile.write(reinterpret_cast<const char*>(&combinedAudio[0]), combinedAudio.size() * sizeof(float));
//     outputFile.close();

//     std::cout << "Combined audio saved as combined_audio.wav" << std::endl;

//     return 0;
// }






// #include <torch/script.h> // One-stop header for including PyTorch
// #include <iostream>
// #include <filesystem>

// namespace fs = std::filesystem;

// int main() {
//     // Path to the directory containing .pt files
//     std::string path_to_mels = "./mel";

//     // Iterate through the directory
//     for (const auto& entry : fs::directory_iterator(path_to_mels)) {
//         if (entry.path().extension() == ".pt") { // Check if the file has a .pt extension
//             try {
//                 // Load the PyTorch tensor from the file
//                 torch::jit::script::Module module = torch::jit::load(entry.path().string());
//                 torch::Tensor tensor = module.forward({}).toTensor();
//                 // Print out the size and shape of the tensor
//                 // std::cout << "File: " << entry.path().filename() << std::endl;
//                 std::cout << "Size: " << tensor.numel() << std::endl; // Total number of elements
//                 std::cout << "Shape: ";
//                 for (size_t i = 0; i < tensor.sizes().size(); ++i) {
//                     std::cout << tensor.sizes()[i] << (i < tensor.sizes().size() - 1 ? "x" : "");
//                 }
//                 std::cout << std::endl << std::endl;
//             } catch (const c10::Error& e) {
//                 std::cerr << "Error loading file " << entry.path().filename() << ": " << e.what() << std::endl;
//             }
//         }
//     }

//     return 0;
// }







// #include <torch/script.h>
// #include <iostream>
// #include <filesystem>
// #include <vector>
// #include <fstream>

// namespace fs = std::filesystem;

// // Function to read mel spectrograms from .pt files in a directory
// std::vector<torch::Tensor> readMelSpectFromFolder(const std::string& folderPath) {
//     std::vector<torch::Tensor> melSpectrograms;
//     for (const auto& entry : fs::directory_iterator(folderPath)) {
//         if (entry.path().extension() == ".pt") {
//             try {
//                 torch::jit::script::Module module = torch::jit::load(entry.path().string());
//                 torch::Tensor melSpectrogram = module.attr("mel_spectrogram").toTensor();
//                 melSpectrograms.push_back(melSpectrogram);
//             } catch (const c10::Error& e) {
//                 std::cerr << "Error loading mel spectrogram from file: " << entry.path() << " - " << e.what() << std::endl;
//             }
//         }
//     }
//     return melSpectrograms;
// }

// int main() {
//     // Directory containing mel spectrograms
//     std::string melFolderPath = "mel";

//     // Load vocoder model
//     std::string modelPath = "torchscript_model.pt";
//     torch::jit::script::Module vocoderModel;
//     try {
//         vocoderModel = torch::jit::load(modelPath);
//     } catch (const c10::Error& e) {
//         std::cerr << "Error loading vocoder model from file: " << modelPath << " - " << e.what() << std::endl;
//         return 1;
//     }

//     // Read mel spectrograms
//     std::vector<torch::Tensor> melSpectrograms = readMelSpectFromFolder(melFolderPath);
//     if (melSpectrograms.empty()) {
//         std::cerr << "No mel spectrograms found in directory: " << melFolderPath << std::endl;
//         return 1;
//     }

//     // Perform inference and combine audio clips
//     std::vector<float> combinedAudio;
//     for (const auto& melSpectrogram : melSpectrograms) {
//         // Perform inference
//         std::vector<torch::jit::IValue> inputs;
//         inputs.push_back(melSpectrogram.unsqueeze(0)); // Add batch dimension
//         at::Tensor generatedAudioTensor = vocoderModel.forward(inputs).toTensor();

//         // Convert tensor to float array
//         std::vector<float> generatedAudio(generatedAudioTensor.data_ptr<float>(), generatedAudioTensor.data_ptr<float>() + generatedAudioTensor.numel());
        
//         // Append generated audio to combined audio
//         combinedAudio.insert(combinedAudio.end(), generatedAudio.begin(), generatedAudio.end());
//     }

//     // Save combined audio as .wav file
//     std::ofstream outputFile("combined_audio.wav", std::ios::binary);
//     if (!outputFile) {
//         std::cerr << "Error creating output file: combined_audio.wav" << std::endl;
//         return 1;
//     }
//     outputFile.write(reinterpret_cast<const char*>(&combinedAudio[0]), combinedAudio.size() * sizeof(float));
//     outputFile.close();

//     std::cout << "Combined audio saved as combined_audio.wav" << std::endl;

//     return 0;
// }








// #include <torch/torch.h>
// #include <iostream>
// #include <filesystem>
// #include <vector>
// #include <fstream>
// #include <torch/script.h>

// namespace fs = std::filesystem;

// // Function to load bytes from file
// std::vector<char> loadBytesFromFile(const std::string& filePath) {
//     std::ifstream file(filePath, std::ios::binary | std::ios::ate);
//     if (!file) {
//         std::cerr << "Error opening file: " << filePath << std::endl;
//         return {};
//     }

//     std::streamsize size = file.tellg();
//     file.seekg(0, std::ios::beg);
//     std::vector<char> bytes(size);
//     if (!file.read(bytes.data(), size)) {
//         std::cerr << "Error reading file: " << filePath << std::endl;
//         return {};
//     }

//     return bytes;
// }

// // Function to read mel spectrograms from bytes files
// std::vector<torch::Tensor> readMelSpectFromFolder(const std::string& folderPath) {
//     std::vector<torch::Tensor> melSpectrograms;
//     for (const auto& entry : fs::directory_iterator(folderPath)) {
//         if (entry.path().extension() == ".pt") {
//             try {
//                 // Load bytes from file
//                 std::vector<char> bytes = loadBytesFromFile(entry.path().string());
//                 if (bytes.empty()) {
//                     continue;
//                 }

//                 // Deserialize bytes into a Torch tensor
//                 torch::IValue melSpectrogramValue = torch::pickle_load(bytes);
//                 if (!melSpectrogramValue.isTensor()) {
//                     std::cerr << "Error: Deserialized value is not a tensor." << std::endl;
//                     continue;
//                 }
//                 torch::Tensor melSpectrogram = melSpectrogramValue.toTensor();

//                 // Adjust tensor shape
//                 melSpectrogram = melSpectrogram.squeeze(0).permute({1, 0}); // Adjust shape

//                 melSpectrograms.push_back(melSpectrogram);
//             } catch (const c10::Error& e) {
//                 std::cerr << "Error loading mel spectrogram from file: " << entry.path() << " - " << e.what() << std::endl;
//             }
//         }
//     }
//     return melSpectrograms;
// }

// int main() {
//     // Directory containing mel spectrograms
//     std::string melFolderPath = "mel";

//     // Load vocoder model
//     std::string modelPath = "torchscript_model.pt";
//     torch::jit::script::Module vocoderModel;
//     try {
//         vocoderModel = torch::jit::load(modelPath);
//     } catch (const c10::Error& e) {
//         std::cerr << "Error loading vocoder model from file: " << modelPath << " - " << e.what() << std::endl;
//         return 1;
//     }

//     // Read mel spectrograms
//     std::vector<torch::Tensor> melSpectrograms = readMelSpectFromFolder(melFolderPath);
//     if (melSpectrograms.empty()) {
//         std::cerr << "No mel spectrograms found in directory: " << melFolderPath << std::endl;
//         return 1;
//     }

//     // Perform inference and combine audio clips
//     std::vector<float> combinedAudio;
//     for (const auto& melSpectrogram : melSpectrograms) {
//         // Perform inference
//         std::vector<torch::jit::IValue> inputs;
//         inputs.push_back(melSpectrogram.unsqueeze(0).unsqueeze(2)); // Add batch dimension and channel dimension
//         at::Tensor generatedAudioTensor = vocoderModel.forward(inputs).toTensor();

//         // Convert tensor to float array
//         std::vector<float> generatedAudio(generatedAudioTensor.data_ptr<float>(), generatedAudioTensor.data_ptr<float>() + generatedAudioTensor.numel());
        
//         // Append generated audio to combined audio
//         combinedAudio.insert(combinedAudio.end(), generatedAudio.begin(), generatedAudio.end());
//     }

//     // Save combined audio as .wav file
//     std::ofstream outputFile("combined_audio.wav", std::ios::binary);
//     if (!outputFile) {
//         std::cerr << "Error creating output file: combined_audio.wav" << std::endl;
//         return 1;
//     }
//     outputFile.write(reinterpret_cast<const char*>(&combinedAudio[0]), combinedAudio.size() * sizeof(float));
//     outputFile.close();

//     std::cout << "Combined audio saved as combined_audio.wav" << std::endl;

//     return 0;
// }










// #include <iostream>
// #include <vector>
// #include <torch/torch.h>
// #include <torch/script.h>
// #include <fstream>
// #include <string>
// #include <cmath>
// #include <cstdint>
// #include <cstring>
// #include <cassert>
// #include <cstdio>

// // Function to read the contents of a file and return them as a vector of bytes
// std::vector<char> get_the_bytes(const std::string& filename) {
//     std::ifstream input(filename, std::ios::binary); // Open the file specified by 'filename' in binary mode
//     std::vector<char> bytes(
//         (std::istreambuf_iterator<char>(input)),
//         (std::istreambuf_iterator<char>())); // Read the contents of the file into the vector
//     input.close(); // Close the file stream
//     return bytes; // Return the vector containing the file's bytes
// }

// // Function to write audio data to a WAV file
// void write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate) {
//     FILE* file = fopen(filename.c_str(), "wb");
//     assert(file);

//     // Write WAV header
//     const int16_t fmt_chunk_size = 16;
//     const int16_t audio_format = 1; // PCM
//     const int16_t num_channels = 1;
//     const int32_t byte_rate = sample_rate * sizeof(int16_t);
//     const int16_t block_align = sizeof(int16_t);
//     const int16_t bits_per_sample = 16;
//     const int32_t data_size = samples.size() * sizeof(int16_t);
//     const int32_t total_size = 36 + data_size;

//     fwrite("RIFF", 1, 4, file);
//     fwrite(&total_size, 4, 1, file);
//     fwrite("WAVE", 1, 4, file);
//     fwrite("fmt ", 1, 4, file);
//     fwrite(&fmt_chunk_size, 2, 1, file);
//     fwrite(&audio_format, 2, 1, file);
//     fwrite(&num_channels, 2, 1, file);
//     fwrite(&sample_rate, 4, 1, file);
//     fwrite(&byte_rate, 4, 1, file);
//     fwrite(&block_align, 2, 1, file);
//     fwrite(&bits_per_sample, 2, 1, file);
//     fwrite("data", 1, 4, file);
//     fwrite(&data_size, 4, 1, file);

//     // Write audio data
//     for (size_t i = 0; i < samples.size(); ++i) {
//         int16_t sample = static_cast<int16_t>(samples[i] * 32767);
//         fwrite(&sample, sizeof(int16_t), 1, file);
//     }

//     fclose(file);
// }

// int main() {
//     // Load the TorchScript model
//     torch::jit::script::Module module = torch::jit::load("torchscript_model.pt");

//     // Read the tensor from the specified file
//     std::vector<char> f = get_the_bytes("/mnt/d/RIR_Estimation/voicefixer/voicefixer/test/inference/build/mel/denoised_mel_44100.pt");

//     // Deserialize the tensor
//     torch::IValue x = torch::pickle_load(f);
//     torch::Tensor tensor = x.toTensor();

//     // Reshape the tensor to [1, 101, 128]
//     tensor = tensor.reshape({1, 101, 128});

//     // Permute the dimensions of the tensor from [1, 101, 128] to [1, 128, 101]
//     // tensor = tensor.permute({0, 2, 1});

//     // Perform inference using the loaded tensor
//     std::vector<torch::jit::IValue> inputs;
//     inputs.push_back(tensor);
//     at::Tensor output_tensor = module.forward(inputs).toTensor();

//     // Convert the output tensor to a vector of floats
//     std::vector<float> output_audio(output_tensor.data<float>(), output_tensor.data<float>() + output_tensor.numel());

//     // Save the output audio to a WAV file
//     write_wav("output_audio.wav", output_audio, 44100);

//     return 0;
// }






// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <torch/torch.h>
// #include <torch/script.h>

// // Function to read the contents of a file and return them as a vector of bytes
// std::vector<char> get_the_bytes(const std::string& filename) {
//     std::ifstream input(filename, std::ios::binary); // Open the file specified by 'filename' in binary mode
//     std::vector<char> bytes(
//         (std::istreambuf_iterator<char>(input)),
//         (std::istreambuf_iterator<char>())); // Read the contents of the file into the vector
//     input.close(); // Close the file stream
//     return bytes; // Return the vector containing the file's bytes
// }

// int main() {
//     // Load the TorchScript model
//     torch::jit::script::Module module = torch::jit::load("torchscript_model.pt");

//     // Read the tensor from the specified file
//     std::vector<char> f = get_the_bytes("/mnt/d/RIR_Estimation/voicefixer/voicefixer/test/inference/build/mel/denoised_mel_44100.pt");
    
//     // Deserialize the tensor
//     torch::IValue x = torch::pickle_load(f);
//     torch::Tensor tensor = x.toTensor();
    
//     // Reshape the tensor to [1, 101, 128]
//     tensor = tensor.reshape({1, 101, 128});
    
//     // Permute the dimensions of the tensor from [1, 101, 128] to [1, 128, 101]
//     tensor = tensor.permute({0, 2, 1});
    
//     // Print the shape of the tensor
//     std::cout << "Tensor shape: ";
//     for (int64_t dim : tensor.sizes()) {
//         std::cout << dim << " ";
//     }
//     std::cout << std::endl;

//     // Print the total number of elements in the tensor
//     std::cout << "Tensor size: " << tensor.numel() << std::endl;
    
//     // Perform inference using the loaded tensor
//     std::vector<torch::jit::IValue> inputs;
//     inputs.push_back(tensor);
//     at::Tensor output_tensor = module.forward(inputs).toTensor();

//     // Print the output tensor shape
//     std::cout << "Output tensor shape: " << output_tensor.sizes() << std::endl;

//     return 0;
// }





// #include <torch/torch.h>

// class Config {
// public:
//     static torch::Tensor x_orig_torch; // Define x_orig_torch appropriately
//     static torch::Tensor get_mel_weight_torch(float percent = 1.0, float a = 18.8927416350036, float b = 0.0269863588184314) {
//         b *= percent;

//         auto func = [](float a, float b, torch::Tensor x) {
//             return a * torch::exp(b * x);
//         };

//         return func(a, b, x_orig_torch);
//     }
// };

// torch::Tensor Config::x_orig_torch = torch::tensor({/* initialize x_orig_torch appropriately */});

// class Model {
// public:
//     torch::Tensor operator()(torch::Tensor mel) {
//         // Define your model logic here
//         return torch::tensor(0); // Return a dummy tensor
//     }
// };

// class MelModel {
// private:
//     Model model;
//     torch::Tensor weight_torch;

// public:
//     MelModel() {
//         // Initialize weight_torch
//         weight_torch = Config::get_mel_weight_torch(1.0)[torch::indexing::None][torch::indexing::None][torch::indexing::None];
//     }

//     torch::Tensor forward(torch::Tensor mel, bool cuda = false) {
//         // Check if last dimension is 128
//         assert(mel.size(-1) == 128);

//         // Check CUDA availability and move tensors if necessary
//         if (cuda) {
//             if (!torch::cuda::is_available()) {
//                 throw std::runtime_error("CUDA is not available.");
//             }
//             mel = mel.cuda();
//         }

//         // Normalize mel spectrogram
//         mel = mel / weight_torch;
//         mel = torch::clamp(torch::abs(mel), 1e-5) - 20.0;

//         // Apply pre-processing
//         mel = mel.index({torch::indexing::Slice(), 0, torch::indexing::Ellipsis()});
        
//         // Pass mel spectrogram through model
//         torch::Tensor wav_re = model(mel);

//         return wav_re;
//     }
// };

// int main() {
//     MelModel melModel;
//     torch::Tensor mel = torch::randn({1, 1, 101, 128}); // Replace /* t-steps */ with actual number of time steps
//     bool cuda = false; // Set to true if using CUDA

//     torch::Tensor wav_re = melModel.forward(mel, cuda);
//     std::cout << "Output shape: " << wav_re.sizes() << std::endl;

//     return 0;
// }









// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <string>
// #include <cmath>
// #include <cstdint>
// #include <cstring>
// #include <cassert>
// #include <cstdio>
// #include <filesystem>
// #include <chrono>
// #include <algorithm>
// #include <future>
// #include <sstream>
// #include <numeric>
// #include <filesystem>
// #include <onnxruntime_cxx_api.h>



// namespace fs = std::filesystem;
// using namespace std::chrono;

// // Function to read the contents of a file and return them as a vector of bytes
// std::vector<char> get_the_bytes(const std::string& filename) {
//     std::ifstream input(filename, std::ios::binary);
//     std::vector<char> bytes(
//         (std::istreambuf_iterator<char>(input)),
//         (std::istreambuf_iterator<char>()));
//     input.close();
//     return bytes;
// }

// // Function to write audio data to a WAV file
// void write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate) {
//     FILE* file = fopen(filename.c_str(), "wb");
//     assert(file);
//     // Write WAV header
//     const int16_t fmt_chunk_size = 16;
//     const int16_t audio_format = 1; // PCM
//     const int16_t num_channels = 1;
//     const int32_t byte_rate = sample_rate * sizeof(int16_t);
//     const int16_t block_align = sizeof(int16_t);
//     const int16_t bits_per_sample = 16;
//     const int32_t data_size = samples.size() * sizeof(int16_t);
//     const int32_t total_size = 36 + data_size;
//     fwrite("RIFF", 1, 4, file);
//     fwrite(&total_size, 4, 1, file);
//     fwrite("WAVE", 1, 4, file);
//     fwrite("fmt ", 1, 4, file);
//     fwrite(&fmt_chunk_size, 2, 1, file);
//     fwrite(&audio_format, 2, 1, file);
//     fwrite(&num_channels, 2, 1, file);
//     fwrite(&sample_rate, 4, 1, file);
//     fwrite(&byte_rate, 4, 1, file);
//     fwrite(&block_align, 2, 1, file);
//     fwrite(&bits_per_sample, 2, 1, file);
//     fwrite("data", 1, 4, file);
//     fwrite(&data_size, 4, 1, file);
//     // Write audio data
//     for (size_t i = 0; i < samples.size(); ++i) {
//         int16_t sample = static_cast<int16_t>(samples[i] * 32767);
//         fwrite(&sample, sizeof(int16_t), 1, file);
//     }
//     fclose(file);
// }


// int main() {
//     // Load the ONNX model
//     Ort::Env env;
//     Ort::SessionOptions session_options;
//     Ort::Session session(env, "new_vocoder.onnx", session_options);

//     // Create a vector to hold combined audio samples
//     std::vector<float> combined_audio;
//     // Create a vector to hold file paths
//     std::vector<std::string> file_paths;
//     // Iterate over all files in the "mel" folder and store their paths
//     for (const auto& entry : fs::directory_iterator("mel")) {
//         file_paths.push_back(entry.path().string());
//     }
//     // Sort file paths to ensure deterministic order
//     std::sort(file_paths.begin(), file_paths.end());
//     // Iterate over sorted file paths
//     for (const auto& file_path : file_paths) {
//         // Read the tensor from the file
//         std::vector<char> f = get_the_bytes(file_path);



//         // Deserialize the tensor
//         std::istringstream iss(std::string(f.begin(), f.end()));
//         Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

//         // Get the tensor shape from the input stream
//         int64_t num_dims;
//         iss.read(reinterpret_cast<char*>(&num_dims), sizeof(int64_t));
//         std::vector<int64_t> input_shape(num_dims);
//         iss.read(reinterpret_cast<char*>(input_shape.data()), num_dims * sizeof(int64_t));

//         // Get the tensor data from the input stream
//         size_t tensor_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
//         std::vector<float> tensor_data(tensor_size);
//         iss.read(reinterpret_cast<char*>(tensor_data.data()), tensor_size * sizeof(float));

//         // Create the ONNX input tensor
//         Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, tensor_data.data(), tensor_size, input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);


//         // Perform inference using the loaded tensor
//         std::vector<Ort::Value> inputs;
//         inputs.push_back(std::move(input_tensor));

//         // Start the timer for each inference
//         auto inference_start = high_resolution_clock::now();

//         const char* input_names[] = {"input"};
//         const char* output_names[] = {"output"};
//         auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);

//         // Stop the timer for each inference
//         auto inference_end = high_resolution_clock::now();
//         auto inference_duration = duration_cast<milliseconds>(inference_end - inference_start);

//         // Get the output tensor
//         auto& output_tensor = output_tensors.front();
//         auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
//         size_t num_samples = output_shape[1];

//         float* output_data = output_tensor.GetTensorMutableData<float>();
//         std::vector<float> output_audio(output_data, output_data + num_samples);

//         // Append the output audio to the combined audio vector
//         combined_audio.insert(combined_audio.end(), output_audio.begin(), output_audio.end());

//         // Print the time taken for each inference
//         std::cout << "Inference time for " << file_path << ": " << inference_duration.count() << " milliseconds" << std::endl;
//     }
//     // Save the combined output audio to a WAV file
//     write_wav("output_audio.wav", combined_audio, 44100);
//     return 0;
// }