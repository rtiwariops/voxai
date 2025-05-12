from setuptools import setup, find_packages
from pathlib import Path

# read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="voxai",
    version="0.1.1",
    description="Voice-driven AI assistant for real-time transcription and Gemini integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ravi (Robbie) Tiwari",
    author_email="rtiwariops@gmail.com",
    url="https://github.com/rtiwariops/voxai",
    packages=find_packages(include=["backend", "backend.*"]),
    include_package_data=True,
    install_requires=[
        "sounddevice",
        "faster_whisper",
        "google-generativeai",
        "python-dotenv"
    ],
    entry_points={
        "console_scripts": [
            "voxai = backend.cli:main",
        ],
    },
    package_data={
        "": ["electron/**/*"],
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
