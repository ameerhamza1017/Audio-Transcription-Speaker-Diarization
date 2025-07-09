# 🎙️ Audio Transcription & Speaker Diarization

This project combines **OpenAI Whisper** and **PyAnnote Audio** to perform end-to-end **speech-to-text transcription** with **speaker diarization**. It identifies “who spoke when” in an audio file and produces a clean JSON output mapping each segment to a speaker.

🚀 Designed to run in a local environment (VS Code or any Python IDE) with GPU support for faster processing.

---

## ✨ Features

- ✅ **Speaker Diarization**: Distinguishes between multiple speakers in audio.
- ✅ **Speech-to-Text Transcription**: Converts spoken words to text using OpenAI Whisper.
- ✅ **Speaker Labeling**: Maps speakers to friendly names like `Speaker 1`, `Speaker 2`, etc.
- ✅ **JSON Output**: Saves a detailed transcript with timestamps and speaker IDs.
- ✅ **GPU Acceleration**: Automatically uses CUDA if available.
