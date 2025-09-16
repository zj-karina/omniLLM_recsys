#!/usr/bin/env python3
"""
Setup script for Fashion Recommendations LLM.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fashion-recommendations-llm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multimodal LLM for Amazon Fashion Recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/fashion-recommendations-llm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fashion-prepare=prepare_fashion_multitask:main",
            "fashion-train=train_fashion:main",
        ],
    },
)
