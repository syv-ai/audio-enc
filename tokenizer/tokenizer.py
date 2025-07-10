import torch
import yaml
from snac import SNAC
from datasets import load_dataset
from datasets import Dataset
import torchaudio.transforms as T
from transformers import AutoTokenizer
import os
import argparse
from timestamps_wav2vec import transcribe_and_align_audio_array, split_audio_into_parts, get_text_from_word_timestamps

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Global model setup
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"SNAC model loaded on device: {device}")

# Load Tokenizer
tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10

tokenizer_name = "canopylabs/orpheus-3b-0.1-pretrained"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
num_proc = os.cpu_count() - 2

def tokenise_audio(waveform, original_sample_rate):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=original_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to(device)

    # generate the codes from snac
    with torch.inference_mode():
        codes = model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item()+128266)
        all_codes.append(codes[1][0][2*i].item()+128266+4096)
        all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))

    return all_codes

def resample_audio_for_snac(audio_array, original_sample_rate):
    """
    Resample audio array from original sample rate to 24kHz for SNAC tokenization.
    """
    if original_sample_rate == 24000:
        return audio_array
    
    # Convert to tensor for resampling
    waveform = torch.from_numpy(audio_array).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    
    # Resample to 24kHz
    resample_transform = T.Resample(orig_freq=original_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)
    
    # Convert back to numpy array
    return waveform.squeeze(0).numpy()

def process_audio_splits(example, audio_column="audio", text_column="text", original_sample_rate=None):
    """
    Process one row and create 3 new rows (2 cumulative splits + full audio).
    Use wav2vec2 alignment with dataset text and audio to determine text_before and text_after.
    """
    processed_examples = []
    
    try:
        answer_audio = example.get(audio_column)
        if not answer_audio or "array" not in answer_audio:
            return processed_examples
            
        audio_array = answer_audio["array"]
        sample_rate = original_sample_rate if original_sample_rate is not None else answer_audio["sampling_rate"]
        
        # Get the text from the dataset
        dataset_text = example.get(text_column, "")
        if not dataset_text:
            print(f"Skipping example: no text found in column '{text_column}'")
            return processed_examples
        
        # Use wav2vec2 to align the full audio with the dataset text
        word_timestamps = transcribe_and_align_audio_array(audio_array, dataset_text, sample_rate)
        
        # Split audio into 3 parts (2 cumulative parts + full)
        audio_parts = split_audio_into_parts(audio_array, sample_rate)
        part_names = ["part1", "part2", "full"]
        
        # Calculate the duration of each part in seconds
        # Note: audio_parts are still at the original sample rate, not 16kHz
        part_durations = []
        for audio_part in audio_parts:
            duration = len(audio_part) / sample_rate  # Use original sample rate
            part_durations.append(duration)
        
        for i, (audio_part, part_name, part_duration) in enumerate(zip(audio_parts, part_names, part_durations)):
            # Get words before and after the end of this audio part
            words_before_timestamp = ""
            words_after_timestamp_within_offset = ""
            
            # For the full audio, all words are "before" and none are "after"
            if part_name == "full":
                words_before_timestamp = get_text_from_word_timestamps(word_timestamps)
                words_after_timestamp_within_offset = ""
            else:
                # Extract words based on timestamp logic using WordTimestamp objects
                time_offset = 0.5  # 0.5 seconds offset for "after" words
                
                words_before = []
                words_after = []
                
                for word_ts in word_timestamps:
                    if word_ts.end < part_duration:
                        words_before.append(word_ts.word)
                    elif word_ts.end < part_duration + time_offset:
                        words_after.append(word_ts.word)
                    else:
                        # Break early for efficiency since timestamps are chronological
                        break
                
                words_before_timestamp = " ".join(words_before)
                words_after_timestamp_within_offset = " ".join(words_after)
            
            # Tokenize the audio part (resample to 24kHz for SNAC)
            try:
                # Resample audio part to 24kHz for SNAC tokenization
                audio_part_24k = resample_audio_for_snac(audio_part, sample_rate)
                codes_list = tokenise_audio(audio_part_24k, 24000)  # Already resampled to 24kHz
            except Exception as e:
                print(f"Skipping {part_name} due to tokenization error: {e}")
                continue
            
            # Create new example for this part
            new_example = {
                "text_before": words_before_timestamp.strip(),
                "text_after": words_after_timestamp_within_offset.strip(),
                "codes_list": codes_list,
                "part_type": part_name,
                "original_text": dataset_text,
            }

            print(f"Text before: {new_example['text_before']}, Text after: {new_example['text_after']}")
            
            processed_examples.append(new_example)
            
    except Exception as e:
        print(f"Skipping row due to error: {e}")
    
    return processed_examples

# Remove duplicate frames
def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]
    removed_frames = 0

    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i:i+7])
        else:
            removed_frames += 1

    example["codes_list"] = result
    return example

# Create input IDs
def create_input_ids(example):
    text_ids_before = tokenizer.encode(example['text_before'], add_special_tokens=True)
    text_ids_before.append(end_of_text)
    text_ids_after = tokenizer.encode(example['text_after'], add_special_tokens=True)
    text_ids_after.append(end_of_text)
    
    example["text_tokens_before"] = text_ids_before
    example["text_tokens_after"] = text_ids_after
    
    input_ids = (
        [start_of_human]
        + text_ids_before
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
        + [start_of_human]
        + text_ids_after
        + [end_of_human]
    )
    
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)

    return example

def process_dataset(dataset_config):
    """Process a single dataset based on its configuration."""
    print(f"Processing dataset: {dataset_config['name']}")
    
    # Load dataset
    ds = load_dataset(
        dataset_config["name"],
        dataset_config.get("section"),
        split=dataset_config["split"],
        streaming=True
    )
    
    # Get sample rate from first item
    first_item = next(iter(ds))
    ds_sample_rate = first_item[dataset_config["audio_column"]]["sampling_rate"]
    
    # Process the streaming dataset
    all_processed_examples = []
    count = 0
    
    for example in ds:
        processed_examples = process_audio_splits(
            example,
            audio_column=dataset_config["audio_column"],
            text_column=dataset_config["text_column"],
            original_sample_rate=ds_sample_rate
        )
        all_processed_examples.extend(processed_examples)
        count += 1
        
        if count % 100 == 0:
            print(f"Processed {count} examples, created {len(all_processed_examples)} rows")
    
    # Convert to dataset
    ds_processed = Dataset.from_list(all_processed_examples)
    
    # Filter out None codes
    ds_processed = ds_processed.filter(lambda x: x["codes_list"] is not None)
    ds_processed = ds_processed.filter(lambda x: len(x["codes_list"]) > 0)
    
    # Remove duplicate frames
    ds_processed = ds_processed.map(remove_duplicate_frames, num_proc=num_proc)
    
    # Create input IDs
    ds_processed = ds_processed.map(create_input_ids, num_proc=num_proc)
    
    # Remove unnecessary columns
    columns_to_keep = ["input_ids", "labels", "attention_mask", "part_type", "text_before", "text_after", "original_text"]
    columns_to_remove = [col for col in ds_processed.column_names if col not in columns_to_keep]
    
    ds_processed = ds_processed.remove_columns(columns_to_remove)
    
    print(f"Final dataset size: {len(ds_processed)}")
    print(f"Sample: {ds_processed[0]}")
    
    # Push to hub
    ds_processed.push_to_hub(dataset_config["output_name"])
    print(f"Dataset pushed to: {dataset_config['output_name']}")

def main():
    parser = argparse.ArgumentParser(description="Process audio datasets with YAML configuration")
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process each dataset
    for dataset_config in config["datasets"]:
        try:
            process_dataset(dataset_config)
        except Exception as e:
            print(f"Error processing dataset {dataset_config['name']}: {e}")
            continue

if __name__ == "__main__":
    main()


