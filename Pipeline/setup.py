#!/usr/bin/env python3
"""
Setup script for kirby_pipeline package.
Install with: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="kirby_pipeline",
    version="0.1.0",
    description="RL training pipeline for Kirby's Dream Land",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pyboy>=2.4.1",
        "gymnasium>=0.29.1",
        "stable-baselines3>=2.3.2",
        "sb3-contrib>=2.3.0",
        "torch>=2.0.0",
        "tensorboard",
        "pyyaml",
        "numpy",
    ],
)
