from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension
import datetime
import os

setup(name='fast_sweeping',
      version='0.0.1',
      description='',
      url='',
      author='',
      author_email='',
      license='MIT',
      packages=find_packages(),
      rust_extensions=[
          RustExtension('fast_sweeping_capi',
'../capi/Cargo.toml', binding=Binding.NoBinding, debug=False)],
      install_requires=['numpy'],
      zip_safe=False)
