from pyannote.audio import Pipeline
import torch
import whisper
import warnings
import json
import ffmpeg  
import os

# Hugging Face Token
HUGGINGFACE_TOKEN = "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"

warnings.filterwarnings("ignore")

# Path to audio file
audio_file = "audio/input1.mp3"
wav_file = "audio5.wav"

# Convert MP3 to WAV using ffmpeg-python
print("Converting MP3 to WAV...")
try:
    (
        ffmpeg
        .input(audio_file)
        .output(wav_file)
        .run(overwrite_output=True, quiet=True)
    )
    print(f"Converted: {wav_file}")
except ffmpeg.Error as e:
    print("FFmpeg error:", e)
    exit(1)

# PyAnnote speaker diarization pipeline
print("Loading speaker diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)

# Move pipeline to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

# Load Whisper model for transcription
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

# Run speaker diarization
print("Processing audio for diarization...")
diarization = pipeline(wav_file)

# Run Whisper transcription
print("Transcribing audio...")
transcription = whisper_model.transcribe(wav_file)

# Map PyAnnote speakers to friendly names
speaker_map = {}
speaker_count = 1

# Create a list to hold JSON entries
json_output = []

for segment in transcription['segments']:
    seg_start = segment['start']
    seg_end = segment['end']
    text = segment['text']

    # Find matching speaker
    speaker_overlap = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(seg_start, turn.start)
        overlap_end = min(seg_end, turn.end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > 0:
            if speaker not in speaker_overlap:
                speaker_overlap[speaker] = 0
            speaker_overlap[speaker] += overlap

    if speaker_overlap:
        matched_speaker = max(speaker_overlap, key=speaker_overlap.get)
        if matched_speaker not in speaker_map:
            speaker_map[matched_speaker] = f"Speaker {speaker_count}"
            speaker_count += 1
        speaker_name = speaker_map[matched_speaker]
    else:
        speaker_name = "Speaker ?"

    # Add entry to JSON output
    json_output.append({
        "speaker": speaker_name,
        "text": text.strip(),
        "start": round(seg_start, 1),
        "end": round(seg_end, 1)
    })

# Write JSON output to file
output_json_file = "transcript.json"
with open(output_json_file, "w") as json_file:
    json.dump(json_output, json_file, indent=4)

print(f"\nTranscript saved as: {output_json_file}")

# Print JSON to console
print(json.dumps(json_output, indent=4))
