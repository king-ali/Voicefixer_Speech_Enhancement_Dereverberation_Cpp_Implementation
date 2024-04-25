cmake_minimum_required(VERSION 3.0)
project(debugfunc)

# Set the paths to LibTorch and Onnx Runtime
set(CMAKE_PREFIX_PATH /mnt/d/RIR_Estimation/voicefixer/voicefixer/test/inference/libtorch)
set(ONNXRUNTIME_DIR /mnt/d/RIR_Estimation/voicefixer/voicefixer/test/inference/onnxruntime)

# Find Torch library and all necessary files
find_package(Torch REQUIRED)

# Find Onnx Runtime
find_package(onnxruntime REQUIRED)

# Include the directories for Onnx Runtime
include_directories(${ONNXRUNTIME_DIR}/include)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${ONNXRUNTIME_CXX_FLAGS}")

# Define the executable
add_executable(debugfunc main.cpp)

# Link Torch and Onnx Runtime libraries to the executable
target_link_libraries(debugfunc "${TORCH_LIBRARIES}" "${onnxruntime_LIBRARY}" pthread)

# Set C++ standard to C++17
set_property(TARGET debugfunc PROPERTY CXX_STANDARD 17)
