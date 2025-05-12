from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="voxai",
    version="0.1.4",
    description="Voice-driven AI assistant for real-time transcription and Gemini integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ravi (Robbie) Tiwari",
    author_email="rtiwariops@gmail.com",
    url="https://github.com/rtiwariops/voxai",
    packages=find_packages(include=["voxai", "voxai.*"]),
    include_package_data=True,
    install_requires=[
        "sounddevice",
        "faster_whisper",
        "google-generativeai",
        "python-dotenv"
    ],
    entry_points={
        "console_scripts": [
            "voxai = voxai.cli:main",
        ],
    },
    package_data={
        "voxai": ["electron/**/*"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
    ],
    python_requires='>=3.7',
)
