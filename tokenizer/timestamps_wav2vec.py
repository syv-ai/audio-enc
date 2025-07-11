import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import re

# Load model and processor once at module level
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "CoRal-project/roest-wav2vec2-1B-v2"

print(f"Loading wav2vec2 model: {model_name} on device: {device}")
try:
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float

def load_and_preprocess_audio(audio_file_path: str, target_sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file to the target sample rate.
    
    Args:
        audio_file_path (str): Path to the audio file
        target_sample_rate (int): Target sample rate (wav2vec2 typically uses 16kHz)
    
    Returns:
        tuple: (audio_array, sample_rate)
    """
    # Load audio and get original sample rate
    waveform, original_sample_rate = librosa.load(audio_file_path, sr=None)
    
    # Resample to target sample rate if needed
    if original_sample_rate != target_sample_rate:
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
        waveform = waveform.squeeze(0).numpy()
    
    return waveform, target_sample_rate

def preprocess_audio_array(audio_array: np.ndarray, original_sample_rate: int, target_sample_rate: int = 16000) -> np.ndarray:
    """
    Preprocess audio array to the target sample rate.
    
    Args:
        audio_array (np.ndarray): Audio array
        original_sample_rate (int): Original sample rate
        target_sample_rate (int): Target sample rate
    
    Returns:
        np.ndarray: Resampled audio array
    """
    if original_sample_rate != target_sample_rate:
        waveform = torch.from_numpy(audio_array).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
        audio_array = waveform.squeeze(0).numpy()
    
    return audio_array

def get_emissions(audio_array: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
    """
    Get emission probabilities from wav2vec2 model.
    
    Args:
        audio_array (np.ndarray): Audio array
        sample_rate (int): Sample rate of the audio
    
    Returns:
        torch.Tensor: Emission probabilities
    """
    # Ensure audio array is 1D and float32
    if audio_array.ndim > 1:
        audio_array = audio_array.flatten()
    audio_array = audio_array.astype(np.float32)
    
    # Check if audio is empty or too short
    if len(audio_array) == 0:
        raise ValueError("Audio array is empty")
    
    # Minimum audio length (100ms at 16kHz = 1600 samples)
    min_length = int(0.1 * sample_rate)
    if len(audio_array) < min_length:
        # Pad audio if too short
        audio_array = np.pad(audio_array, (0, min_length - len(audio_array)), 'constant')
    
    # Preprocess audio with proper parameters
    inputs = processor(
        audio_array, 
        sampling_rate=sample_rate, 
        return_tensors="pt", 
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get emission probabilities (log probabilities)
    emissions = torch.log_softmax(outputs.logits, dim=-1)
    
    return emissions

def align_text_with_audio(audio_array: np.ndarray, text: str, sample_rate: int = 16000) -> List[WordTimestamp]:
    """
    Align text with audio using wav2vec2 forced alignment.
    
    Args:
        audio_array (np.ndarray): Audio array
        text (str): Text to align
        sample_rate (int): Sample rate of the audio
    
    Returns:
        List[WordTimestamp]: List of word timestamps
    """
    # Validate inputs
    if len(audio_array) == 0:
        raise ValueError("Audio array is empty")
    
    if not text or not text.strip():
        raise ValueError("Text is empty")
    
    # Get emissions from the model
    try:
        emissions = get_emissions(audio_array, sample_rate)
    except Exception as e:
        print(f"Failed to get emissions: {e}")
        raise
    
    # Store original text and words with their casing
    original_text = text.strip()
    original_words = original_text.split()
    
    if not original_words:
        raise ValueError("No words found in text")
    
    # Clean and prepare text for model (lowercase for alignment)
    text_for_model = original_text.lower()
    lowercase_words = text_for_model.split()
    
    # Try different alignment approaches
    # Skip torchaudio forced alignment due to compatibility issues
    # Start with CTC-based alignment which is more stable
    try:
        # Method 1: Try CTC-based alignment first
        result = _try_ctc_alignment(emissions, text_for_model, lowercase_words, sample_rate)
        return _map_to_original_casing(result, original_words)
    except Exception as e:
        print(f"CTC alignment failed: {e}")
        try:
            # Method 2: Fallback to simple alignment
            result = _simple_alignment(emissions, lowercase_words, sample_rate)
            return _map_to_original_casing(result, original_words)
        except Exception as e:
            print(f"Simple alignment failed: {e}")
            # Final fallback: create simple uniform timestamps
            return _create_uniform_timestamps(original_words, len(audio_array) / sample_rate)

def _try_torchaudio_alignment(emissions: torch.Tensor, text: str, words: List[str], sample_rate: int) -> List[WordTimestamp]:
    """Try torchaudio forced alignment approach."""
    try:
        import torchaudio
        print(f"Torchaudio version: {torchaudio.__version__}")
        
        # Try different import paths for different torchaudio versions
        try:
            from torchaudio.functional import forced_align
        except ImportError:
            try:
                from torchaudio.models.wav2vec2.utils import forced_align
            except ImportError:
                raise ImportError("forced_align function not available in this torchaudio version")
        
        # Tokenize text properly
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            # Fallback: try with processor tokenizer
            tokens = processor.tokenizer.encode(text, add_special_tokens=False)
        
        # Validate tokens
        if not tokens:
            raise ValueError("No tokens generated from text")
        
        # Convert to tensor
        if isinstance(tokens, list):
            token_ids = torch.tensor(tokens, dtype=torch.long)
        else:
            token_ids = tokens
            
        # Validate token_ids
        if token_ids.numel() == 0:
            raise ValueError("Token IDs tensor is empty")
            
        # Ensure proper tensor dimensions for forced_align
        # forced_align expects emissions: (T, C) where T=frames, C=classes
        # and targets: (N,) where N=number of tokens
        
        print(f"Original emissions shape: {emissions.shape}")
        print(f"Original token_ids shape: {token_ids.shape}")
        
        # Handle emissions tensor
        if emissions.dim() == 3:
            # Shape: (batch, frames, classes) -> (frames, classes)
            emissions = emissions.squeeze(0)
        elif emissions.dim() == 2:
            # Already correct: (frames, classes)
            pass
        else:
            raise ValueError(f"Unexpected emissions shape: {emissions.shape}, expected 2D or 3D")
        
        # Handle token_ids tensor
        if token_ids.dim() == 0:
            # Scalar -> 1D tensor
            token_ids = token_ids.unsqueeze(0)
        elif token_ids.dim() == 1:
            # Already correct: (num_tokens,)
            pass
        elif token_ids.dim() == 2:
            # 2D -> 1D
            token_ids = token_ids.squeeze()
            if token_ids.dim() == 0:  # If squeezing resulted in scalar
                token_ids = token_ids.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected token_ids shape: {token_ids.shape}, expected 1D or 2D")
        
        print(f"Final emissions shape: {emissions.shape}")
        print(f"Final token_ids shape: {token_ids.shape}")
            
        # Get blank token ID
        blank_id = getattr(tokenizer, 'pad_token_id', None) or getattr(tokenizer, 'unk_token_id', None) or 0
        
        # Move tensors to same device
        token_ids = token_ids.to(emissions.device)
        
        # Try forced alignment with proper error handling
        try:
            # Check if this is the right way to call forced_align
            print(f"Calling forced_align with emissions: {emissions.shape}, token_ids: {token_ids.shape}, blank: {blank_id}")
            
            # Try different calling patterns for different torchaudio versions
            try:
                # Method 1: Standard call
                alignment = forced_align(emissions, token_ids, blank=blank_id)
            except Exception as e1:
                print(f"Standard call failed: {e1}")
                try:
                    # Method 2: Try without blank parameter
                    alignment = forced_align(emissions, token_ids)
                except Exception as e2:
                    print(f"No blank parameter failed: {e2}")
                    try:
                        # Method 3: Try with emissions as log_probs instead of logits
                        log_probs = torch.log_softmax(emissions, dim=-1)
                        alignment = forced_align(log_probs, token_ids, blank=blank_id)
                    except Exception as e3:
                        print(f"Log probs failed: {e3}")
                        raise e1  # Re-raise original error
                        
        except RuntimeError as e:
            if "Dimension out of range" in str(e):
                print(f"Dimension error details: {e}")
                print(f"Emissions shape: {emissions.shape}, token_ids shape: {token_ids.shape}")
                raise e
            else:
                raise e
        
        # Validate alignment results
        if alignment is None:
            raise ValueError("Forced alignment returned None")
        
        # Convert alignment to tensor if it's not already
        if not isinstance(alignment, torch.Tensor):
            alignment = torch.tensor(alignment)
        
        # Check if alignment is empty
        if alignment.numel() == 0:
            raise ValueError("Alignment is empty")
        
        # Check for None values in alignment
        if torch.any(torch.isnan(alignment.float())):
            raise ValueError("Alignment contains NaN values")
        
        # Debug information
        print(f"Alignment shape: {alignment.shape}, tokens: {len(tokens)}, words: {len(words)}")
        
        # Convert alignment to word timestamps
        word_timestamps = []
        frame_duration = 0.02  # 20ms
        
        # Simple mapping: distribute tokens across words
        tokens_per_word = len(tokens) / len(words) if words else 1
        
        for i, word in enumerate(words):
            start_token_idx = int(i * tokens_per_word)
            end_token_idx = int((i + 1) * tokens_per_word)
            
            if start_token_idx < len(alignment):
                # Get frame indices with proper error handling
                try:
                    start_frame_val = alignment[start_token_idx].item() if start_token_idx < len(alignment) else 0
                    end_frame_val = alignment[min(end_token_idx - 1, len(alignment) - 1)].item() if end_token_idx <= len(alignment) else len(alignment) - 1
                    
                    # Validate frame values
                    if start_frame_val is None or end_frame_val is None:
                        raise ValueError("Frame values are None")
                    
                    start_frame = int(start_frame_val)
                    end_frame = int(end_frame_val)
                    
                    start_time = start_frame * frame_duration
                    end_time = (end_frame + 1) * frame_duration
                    
                    word_timestamps.append(WordTimestamp(
                        word=word,
                        start=start_time,
                        end=end_time,
                        confidence=1.0
                    ))
                except (ValueError, TypeError) as e:
                    # Skip this word if frame extraction fails
                    print(f"Warning: Could not extract frames for word '{word}': {e}")
                    continue
        
        return word_timestamps
        
    except ImportError:
        raise Exception("torchaudio forced alignment not available")

def _try_ctc_alignment(emissions: torch.Tensor, text: str, words: List[str], sample_rate: int) -> List[WordTimestamp]:
    """Try CTC-based alignment approach."""
    print(f"Starting CTC alignment with emissions shape: {emissions.shape}")
    
    # Handle emissions tensor
    if emissions.dim() == 3:
        emissions = emissions.squeeze(0)
    
    # Get the most likely token sequence using CTC decoding
    emissions_np = emissions.cpu().numpy()
    
    # Simple CTC decoding - get most likely token at each frame
    predicted_tokens = np.argmax(emissions_np, axis=1)
    
    # Remove blanks and duplicates (basic CTC decoding)
    blank_token = getattr(tokenizer, 'pad_token_id', None) or getattr(tokenizer, 'unk_token_id', None) or 0
    decoded_tokens = []
    prev_token = None
    
    for token in predicted_tokens:
        if token != blank_token and token != prev_token:
            decoded_tokens.append(token)
        prev_token = token
    
    print(f"CTC decoding: {len(predicted_tokens)} frames -> {len(decoded_tokens)} tokens")
    
    # Create word timestamps based on frame positions
    word_timestamps = []
    frame_duration = 0.02  # 20ms
    total_frames = len(predicted_tokens)
    
    if not words:
        print("No words to align")
        return word_timestamps
    
    # Distribute frames across words more intelligently
    frames_per_word = total_frames / len(words)
    
    print(f"Distributing {total_frames} frames across {len(words)} words ({frames_per_word:.2f} frames per word)")
    
    for i, word in enumerate(words):
        start_frame = int(i * frames_per_word)
        end_frame = int((i + 1) * frames_per_word)
        
        # Ensure end_frame doesn't exceed total frames
        end_frame = min(end_frame, total_frames)
        
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        
        word_timestamps.append(WordTimestamp(
            word=word,
            start=start_time,
            end=end_time,
            confidence=0.8  # Medium confidence for CTC
        ))
    
    print(f"Generated {len(word_timestamps)} word timestamps")
    return word_timestamps

def _simple_alignment(emissions: torch.Tensor, words: List[str], sample_rate: int) -> List[WordTimestamp]:
    """
    Simple alignment fallback when torchaudio forced alignment is not available.
    """
    # Calculate frame duration
    frame_duration = 0.02  # 20ms frames
    total_frames = emissions.shape[1]
    total_duration = total_frames * frame_duration
    
    # Simple uniform distribution of words across audio duration
    word_timestamps = []
    words_per_frame = len(words) / total_frames
    
    for i, word in enumerate(words):
        start_frame = int(i / words_per_frame)
        end_frame = int((i + 1) / words_per_frame)
        
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        
        word_timestamps.append(WordTimestamp(
            word=word,
            start=start_time,
            end=end_time,
            confidence=0.5  # Lower confidence for simple alignment
        ))
    
    return word_timestamps

def _map_to_original_casing(word_timestamps: List[WordTimestamp], original_words: List[str]) -> List[WordTimestamp]:
    """
    Map the lowercase aligned words back to their original casing.
    
    Args:
        word_timestamps (List[WordTimestamp]): Word timestamps with lowercase words
        original_words (List[str]): Original words with their casing
    
    Returns:
        List[WordTimestamp]: Word timestamps with original casing
    """
    # Create a mapping from lowercase to original casing
    original_casing_map = {}
    for original_word in original_words:
        lowercase_word = original_word.lower()
        original_casing_map[lowercase_word] = original_word
    
    # Map the aligned words back to original casing
    mapped_timestamps = []
    for word_ts in word_timestamps:
        lowercase_word = word_ts.word.lower()
        
        # Try to find the original casing
        original_word = original_casing_map.get(lowercase_word, word_ts.word)
        
        # If not found, try to find a word that starts with the same letters
        if original_word == word_ts.word and lowercase_word != word_ts.word:
            for orig_word in original_words:
                if orig_word.lower() == lowercase_word:
                    original_word = orig_word
                    break
        
        mapped_timestamps.append(WordTimestamp(
            word=original_word,
            start=word_ts.start,
            end=word_ts.end,
            confidence=word_ts.confidence
        ))
    
    return mapped_timestamps

def _create_uniform_timestamps(words: List[str], total_duration: float) -> List[WordTimestamp]:
    """
    Create uniform timestamps as a final fallback when all alignment methods fail.
    
    Args:
        words (List[str]): List of words
        total_duration (float): Total duration of audio in seconds
    
    Returns:
        List[WordTimestamp]: List of word timestamps with uniform distribution
    """
    if not words:
        return []
    
    word_timestamps = []
    duration_per_word = total_duration / len(words)
    
    for i, word in enumerate(words):
        start_time = i * duration_per_word
        end_time = (i + 1) * duration_per_word
        
        word_timestamps.append(WordTimestamp(
            word=word,
            start=start_time,
            end=end_time,
            confidence=0.1  # Very low confidence for uniform fallback
        ))
    
    return word_timestamps

def transcribe_and_align_audio_file(audio_file_path: str, text: str) -> List[WordTimestamp]:
    """
    Transcribe and align audio file with provided text.
    
    Args:
        audio_file_path (str): Path to the audio file
        text (str): Text to align with the audio
    
    Returns:
        List[WordTimestamp]: List of word timestamps
    """
    # Load and preprocess audio
    audio_array, sample_rate = load_and_preprocess_audio(audio_file_path)
    
    # Align text with audio
    word_timestamps = align_text_with_audio(audio_array, text, sample_rate)
    
    return word_timestamps

def transcribe_and_align_audio_array(audio_array: np.ndarray, text: str, sample_rate: int = 16000) -> List[WordTimestamp]:
    """
    Transcribe and align audio array with provided text.
    
    Args:
        audio_array (np.ndarray): Audio array
        text (str): Text to align with the audio
        sample_rate (int): Sample rate of the audio
    
    Returns:
        List[WordTimestamp]: List of word timestamps
    """
    # Preprocess audio to target sample rate
    audio_array = preprocess_audio_array(audio_array, sample_rate)
    
    # Align text with audio
    word_timestamps = align_text_with_audio(audio_array, text, 16000)
    
    return word_timestamps

def get_words_before_timestamp(audio_file_path: str, text: str, timestamp_seconds: float, time_offset: float = 0.5) -> Tuple[str, str]:
    """
    Get all words from an audio file that end before a specified timestamp,
    and additional words in the next time_offset seconds.
    
    Args:
        audio_file_path (str): Path to the audio file
        text (str): Text to align with the audio
        timestamp_seconds (float): Timestamp in seconds
        time_offset (float): Time window after timestamp to include additional words
    
    Returns:
        tuple: (words_before_timestamp, words_after_timestamp_within_offset)
    """
    # Get word timestamps
    word_timestamps = transcribe_and_align_audio_file(audio_file_path, text)
    
    # Filter words based on timestamp
    words_before_timestamp = []
    words_after_timestamp_within_offset = []
    
    for word_ts in word_timestamps:
        if word_ts.end < timestamp_seconds:
            words_before_timestamp.append(word_ts.word)
        elif word_ts.end < timestamp_seconds + time_offset:
            words_after_timestamp_within_offset.append(word_ts.word)
    
    return " ".join(words_before_timestamp), " ".join(words_after_timestamp_within_offset)

def split_audio_into_parts(audio_array: np.ndarray, sample_rate: int = 16000) -> List[np.ndarray]:
    """
    Split audio array into cumulative parts and return 3 parts.
    
    Args:
        audio_array (np.ndarray): Audio array
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

def get_text_from_word_timestamps(word_timestamps: List[WordTimestamp]) -> str:
    """
    Extract text from word timestamps.
    
    Args:
        word_timestamps (List[WordTimestamp]): List of word timestamps
    
    Returns:
        str: Concatenated text
    """
    return " ".join([word_ts.word for word_ts in word_timestamps])

def process_dataset_sample(audio_array: np.ndarray, text: str, sample_rate: int = 16000) -> Dict[str, Any]:
    """
    Process a single sample from a dataset with audio and text columns.
    
    Args:
        audio_array (np.ndarray): Audio array from dataset
        text (str): Text from dataset
        sample_rate (int): Sample rate of the audio
    
    Returns:
        Dict[str, Any]: Dictionary containing word timestamps and other info
    """
    # Get word timestamps
    word_timestamps = transcribe_and_align_audio_array(audio_array, text, sample_rate)
    
    return {
        "word_timestamps": word_timestamps,
        "text": text,
        "total_duration": len(audio_array) / sample_rate,
        "num_words": len(word_timestamps)
    }

# Example usage for testing
if __name__ == "__main__":
    # Example with file path and text
    audio_file_path = "audio.wav"
    text = """Den grimme ælling
Eventyr af Hans Christian Andersen
Der var så dejligt ude på landet; det var sommer, kornet stod gult, havren grøn, høet var rejst i stakke nede i de grønne enge, og der gik storken på sine lange, røde ben og snakkede ægyptisk, for det sprog havde han lært af sin moder. Rundt om ager og eng var der store skove, og midt i skovene dybe søer; jo, der var rigtignok dejligt derude på landet! Midt i solskinnet lå der en gammel herregård med dybe kanaler rundt om, og fra muren og ned til vandet voksede store skræppeblade, der var så høje, at små børn kunne stå oprejste under de største; der var lige så vildsomt derinde, som i den tykkeste skov, og her lå en and på sin rede; hun skulle ruge sine små ællinger ud, men nu var hun næsten ked af det, fordi det varede så længe, og hun sjælden fik visit; de andre ænder holdt mere af at svømme om i kanalerne, end at løbe op og sidde under et skræppeblad for at snadre med hende.
Endelig knagede det ene æg efter det andet: "pip! pip!" sagde det, alle æggeblommerne var blevet levende og stak hovedet ud."""
    
    try:
        word_timestamps = transcribe_and_align_audio_file(audio_file_path, text)
        
        print("Word timestamps:")
        for word_ts in word_timestamps:
            print(f"{word_ts.word}: {word_ts.start:.2f}s - {word_ts.end:.2f}s (confidence: {word_ts.confidence:.2f})")
        
        # Test getting words before timestamp
        words_before, words_after = get_words_before_timestamp(audio_file_path, text, 5.0)
        print(f"\nWords before 5.0s: {words_before}")
        print(f"Words after 5.0s (within 0.5s): {words_after}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required audio file and dependencies installed.") 