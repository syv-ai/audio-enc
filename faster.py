
import whisperx
import gc
from diarizers import SegmentationModel
from pyannote.audio import Pipeline
import torch

device = "cpu"
audio_file = "conversations_1196.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("pluttodk/hviske-tiske", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
print(audio)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment


# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# load the pre-trained pyannote pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
model = SegmentationModel().from_pretrained("syvai/speaker-segmentation")
model = model.to_pyannote_model()
pipeline._segmentation.model = model.to(device)

new_audio = {}
new_audio["waveform"] = torch.from_numpy(audio).unsqueeze(0)  # Convert to tensor and add channel dimension
new_audio["sample_rate"] = 16000
# add min/max number of speakers if known
diarize_segments = pipeline(new_audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

# Convert pyannote annotation to DataFrame format expected by whisperx
import pandas as pd
diarize_df = pd.DataFrame(columns=['start', 'end', 'speaker'])
for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
    diarize_df = pd.concat([diarize_df, pd.DataFrame({
        'start': [segment.start],
        'end': [segment.end], 
        'speaker': [speaker]
    })], ignore_index=True)

result = whisperx.assign_word_speakers(diarize_df, result)
print(diarize_df)
print(result["segments"]) # segments are now assigned speaker IDs

for segment in result["segments"]:
    print(f"Speaker {segment['speaker']}: {segment['text']}")