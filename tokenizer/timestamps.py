import whisper_timestamped as whisper
import json
import torch
import torchaudio.transforms as T
import librosa
import numpy as np

# Load model once at module level
model = whisper.load_model("CoRal-project/roest-whisper-large-v1", device="cuda")

def transcribe_full_audio(audio_file_path, language="da"):
    """
    Transcribe the full audio file and return the complete transcription result.
    
    Args:
        audio_file_path (str): Path to the audio file
        language (str): Language of the audio file
    
    Returns:
        dict: Complete whisper transcription result with timestamps
    """
    # Load audio and get original sample rate
    waveform, ds_sample_rate = librosa.load(audio_file_path, sr=None)
    
    # Check if sample rate is 24000 Hz, if not convert it
    target_sample_rate = 24000
    if ds_sample_rate != target_sample_rate:
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
        waveform = waveform.squeeze(0).numpy()
    
    # Transcribe the full audio with timestamps using the pre-loaded model
    result = whisper.transcribe(model, waveform, language=language)
    
    return result, waveform, target_sample_rate

def transcribe_audio_array(audio_array, sample_rate=24000, language="da"):
    """
    Transcribe an audio array directly.
    
    Args:
        audio_array (numpy.ndarray): Audio array
        sample_rate (int): Sample rate of the audio
        language (str): Language of the audio
    
    Returns:
        dict: Complete whisper transcription result with timestamps
    """
    # Check if sample rate is 24000 Hz, if not convert it
    target_sample_rate = 24000
    if sample_rate != target_sample_rate:
        waveform = torch.from_numpy(audio_array).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
        audio_array = waveform.squeeze(0).numpy()
    
    # Transcribe the audio with timestamps using the pre-loaded model
    result = whisper.transcribe(model, audio_array, language=language)
    
    return result

def split_audio_into_parts(audio_array, sample_rate=24000):
    """
    Split audio array into cumulative parts and return 3 parts.
    
    Args:
        audio_array (numpy.ndarray): Audio array
        sample_rate (int): Sample rate of the audio
    
    Returns:
        list: [part1, part2, full_audio] where each is a numpy array
        - part1: first third (0% to 33%)
        - part2: first two thirds (0% to 66%)
        - full: complete original audio (0% to 100%)
    """
    audio_length = len(audio_array)
    part_length = audio_length // 3
    
    part1 = audio_array[:part_length]                    # 0% to 33%
    part2 = audio_array[:2*part_length]                  # 0% to 66%
    full_audio = audio_array                             # 0% to 100%
    
    return [part1, part2, full_audio]

def get_text_from_transcription(transcription_result):
    """
    Extract text from transcription result.
    
    Args:
        transcription_result (dict): Whisper transcription result
    
    Returns:
        str: Extracted text
    """
    text = ""
    for segment in transcription_result.get("segments", []):
        text += segment.get("text", "") + " "
    return text.strip()

def get_words_before_timestamp(audio_file_path, timestamp_seconds, language="da", time_offset=0.5):
    """
    Get all words from an audio file that end before a specified timestamp,
    and additional words in the next time_offset seconds.
    
    Args:
        audio_file_path (str): Path to the audio file
        timestamp_seconds (float): Timestamp in seconds
        language (str): Language of the audio file
        time_offset (float): Time window after timestamp to include additional words
    
    Returns:
        tuple: (words_before_timestamp, words_after_timestamp_within_offset)
            - words_before_timestamp: String of words that end before the timestamp
            - words_after_timestamp_within_offset: String of words that end after timestamp but within timestamp + time_offset
    """
    # Load audio and get original sample rate
    waveform, ds_sample_rate = librosa.load(audio_file_path, sr=None)
    
    # Check if sample rate is 24000 Hz, if not convert it
    target_sample_rate = 24000
    if ds_sample_rate != target_sample_rate:
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
        waveform = waveform.squeeze(0).numpy()
    
    # Cut off audio at timestamp + 2 seconds for efficiency
    cutoff_samples = int((timestamp_seconds + 2) * target_sample_rate)
    audio_trimmed = waveform[:cutoff_samples]
    
    # Transcribe only the trimmed audio with timestamps using the pre-loaded model
    result = whisper.transcribe(model, audio_trimmed, language=language)
    
    # Extract word texts that end before the specified timestamp
    words_before_timestamp = ""
    words_after_timestamp_within_offset = ""
    
    # Iterate through segments
    for segment in result.get("segments", []):
        # Iterate through words in each segment
        for word in segment.get("words", []):
            word_end = word.get("end", 0)
            word_text = word.get("text", "") + " "
            
            # Check if word ends before the specified timestamp
            if word_end < timestamp_seconds:
                words_before_timestamp += word_text
            # Check if word ends after timestamp but before timestamp + time_offset seconds
            elif word_end < timestamp_seconds + time_offset:
                words_after_timestamp_within_offset += word_text
            else:
                # Since words are in chronological order, we can break early
                # when we find a word that ends after our timestamp + time_offset
                break
    
    return words_before_timestamp.strip(), words_after_timestamp_within_offset.strip()

# Original code for testing
if __name__ == "__main__":
    words_before_80_seconds, words_after_80_seconds = get_words_before_timestamp("audio.wav", 80.0)
    print(words_before_80_seconds)
    print(words_after_80_seconds)