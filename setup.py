# setup.py
from setuptools import setup, find_packages

setup(
    name="qft_simulation",
    version="0.1.0",
    description="A package for simulating Quantum Field Theory, including phi^4 theory",
    author="Zahed Riyaz",
    author_email="your.email@example.com",
    url="https://github.com/zahed-riyaz/qft_simulation", 
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "sympy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change if using another license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
