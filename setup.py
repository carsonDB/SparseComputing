from os import path
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


root = path.dirname(path.abspath(__file__))
cpp_dir = path.join(root, 'src/cpp')
sources = sorted(glob(path.join(cpp_dir, "*.cpp")))
cpp_module = CppExtension(
    'sparseOps_cpp', 
    sources,
    # include_dirs=[cpp_dir],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp']
)

setup(
    name='sparseOps_cpp',
    ext_modules=[cpp_module],
    cmdclass={ 'build_ext': BuildExtension })
