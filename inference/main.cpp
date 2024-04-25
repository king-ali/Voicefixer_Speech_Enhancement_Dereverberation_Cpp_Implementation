#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <onnxruntime/include/onnxruntime_cxx_api.h>
#include <onnxruntime/include/cpu_provider_factory.h>
#include <torch/torch.h> 
#include <torch/script.h>
namespace fs = std::filesystem;
using namespace std::chrono;
std::vector<char> get_the_bytes(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();
    return bytes;
}
void write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate) {
    FILE* file = fopen(filename.c_str(), "wb");
    assert(file);
    const int16_t fmt_chunk_size = 16, audio_format = 1, num_channels = 1, block_align = 2, bits_per_sample = 16;
    const int32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
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
    for (auto sample : samples) {
        int16_t int_sample = static_cast<int16_t>(sample * 32767);
        fwrite(&int_sample, sizeof(int16_t), 1, file);
    }
    fclose(file);
}
int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(6);
    session_options.SetInterOpNumThreads(4);
    Ort::Session session(env, "/mnt/d/RIR_Estimation/voicefixer/voicefixer/test/inference/build/new_vocoder.onnx", session_options);
    std::vector<float> combined_audio;
    std::vector<std::string> file_paths;
    for (const auto& entry : fs::directory_iterator("/mnt/d/RIR_Estimation/voicefixer/voicefixer/test/inference/build/mel")) {
        file_paths.push_back(entry.path().string());
    }
    std::sort(file_paths.begin(), file_paths.end());
    for (const auto& file_path : file_paths) {
        std::vector<char> file_data = get_the_bytes(file_path);
        auto tensor_ivalue = torch::jit::pickle_load(file_data);
        auto tensor = tensor_ivalue.toTensor().squeeze().to(torch::kFloat32);
     
        std::vector<float> tensor_data(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
        
        int64_t input_tensor_shape[] = {1, 128, 106}; // This should match the expected input shape of the model
        std::vector<const char*> input_names = {"input.1"}; // Ensure these match model's input names
        std::vector<const char*> output_names = {"694"}; // Ensure these match model's output names
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, tensor_data.data(), tensor.numel(), input_tensor_shape, 3);
        auto inference_start = high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        auto inference_end = high_resolution_clock::now();
        auto inference_duration = duration_cast<milliseconds>(inference_end - inference_start);
        std::cout << "Inference time for " << file_path << ": " << inference_duration.count() << " milliseconds" << std::endl;
        float* output_ptr = output_tensors.front().GetTensorMutableData<float>();
        size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        combined_audio.insert(combined_audio.end(), output_ptr, output_ptr + output_size);
    }
    write_wav("output_audio.wav", combined_audio, 44100);
    return 0;
}
