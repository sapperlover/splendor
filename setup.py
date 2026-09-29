from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[Pybind11Extension(
        "splendor._native_search", ["src/splendor/native_search.cpp"],
        cxx_std=17, extra_compile_args=["-O3", "-DNDEBUG"],
    )],
    cmdclass={"build_ext": build_ext},
)
