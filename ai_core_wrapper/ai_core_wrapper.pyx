# ai_core_wrapper.pyx (Self-Contained Version)
# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.string cimport memcpy
import numpy as np
cimport numpy as np

np.import_array()

# --- C++ Code is now embedded directly inside the Cython file ---
cdef extern from *:
    """
    #include <torch/script.h>
    #include <torch/csrc/api/include/torch/cuda.h>
    #include <vector>
    #include <string>
    #include <iostream>

    namespace AICore {

        // Forward declare the class so the wrapper can see it
        class FastLSTMEngine;

        // Implementation of the C++ Engine
        class FastLSTMEngine {
        private:
            torch::jit::script::Module module;
            torch::Device device;

            torch::Device get_device() {
                if (torch::cuda::is_available()) {
                    std::cout << "CUDA is available! C++ engine will use GPU." << std::endl;
                    return torch::kCUDA;
                }
                std::cout << "CUDA not available. C++ engine will use CPU." << std::endl;
                return torch::kCPU;
            }

        public:
            FastLSTMEngine(const std::string& model_path) : device(get_device()) {
                try {
                    module = torch::jit::load(model_path);
                    module.to(device);
                    module.eval();
                }
                catch (const c10::Error& e) {
                    std::cerr << "FATAL: Error loading TorchScript model: " << e.what() << std::endl;
                    throw;
                }
            }

            std::vector<float> forward(const std::vector<float>& input_features) {
                torch::Tensor input_tensor = torch::from_blob((void*)input_features.data(), {(long)input_features.size()}, torch::kFloat32)
                                                .clone()
                                                .view({1, 1, (long)input_features.size()})
                                                .to(device);

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);

                at::Tensor output_tensor = module.forward(inputs).toTensor().to(torch::kCPU);

                std::vector<float> output_vec(output_tensor.data_ptr<float>(), output_tensor.data_ptr<float>() + output_tensor.numel());
                return output_vec;
            }
        };
    }
    """
    cdef cppclass FastLSTMEngine "AICore::FastLSTMEngine":
        FastLSTMEngine(const string& model_path) except +
        vector[float] forward(const vector[float]& input_features)

# --- Python Wrapper (no changes needed here) ---
cdef class PyFastLSTMEngine:
    cdef FastLSTMEngine* thisptr
    cdef bint is_initialized

    def __cinit__(self, model_path: str):
        cdef string model_path_cpp = model_path.encode('utf-8')
        try:
            self.thisptr = new FastLSTMEngine(model_path_cpp)
            self.is_initialized = True
        except Exception as e:
            self.is_initialized = False
            print(f"Critical Error initializing C++ FastLSTMEngine: {e}")
            raise

    def __dealloc__(self):
        if self.is_initialized:
            del self.thisptr

    def forward(self, np.ndarray[float, ndim=1, mode="c"] input_features):
        if not self.is_initialized:
            raise RuntimeError("FastLSTMEngine was not initialized correctly.")

        cdef float* data_ptr = <float*>np.PyArray_DATA(input_features)
        cdef int size = input_features.shape[0]

        cdef vector[float] input_vec
        input_vec.assign(data_ptr, data_ptr + size)
        cdef vector[float] result_vec = self.thisptr.forward(input_vec)

        cdef np.ndarray[float, ndim=1] result_array = np.empty(result_vec.size(), dtype=np.float32)
        cdef float* result_ptr = <float*>np.PyArray_DATA(result_array)
        
        if result_vec.size() > 0:
            memcpy(result_ptr, &result_vec[0], result_vec.size() * sizeof(float))

        return result_array