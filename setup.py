# setup.py (Simplified Version)
import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

LIBTORCH_PATH = os.path.abspath("C:/libtorch")

if not os.path.isdir(LIBTORCH_PATH):
    raise FileNotFoundError(
        f"LibTorch path not found at '{LIBTORCH_PATH}'. "
        "Please download LibTorch and update the LIBTORCH_PATH variable."
    )

ai_core_extension = Extension(
    name="ai_core_wrapper",
    sources=["ai_core_wrapper/ai_core_wrapper.pyx"],
    language="c++",
    include_dirs=[
        numpy.get_include(),
        os.path.join(LIBTORCH_PATH, "include"),
        os.path.join(LIBTORCH_PATH, "include", "torch", "csrc", "api", "include"),
    ],
    library_dirs=[os.path.join(LIBTORCH_PATH, "lib")],
    libraries=["torch", "torch_cpu", "c10"],
    extra_compile_args=["/std:c++17", "/EHsc", "/O2"],
)

setup(
    name="ai_core_wrapper",
    ext_modules=cythonize(
        [ai_core_extension],
        compiler_directives={'language_level': "3"}
    ),
    zip_safe=False,
)