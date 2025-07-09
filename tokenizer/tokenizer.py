import torch
from snac import SNAC
from datasets import load_dataset
from datasets import Dataset
import torchaudio.transforms as T
from transformers import AutoTokenizer
import os
from timestamps import transcribe_audio_array, split_audio_into_parts, get_text_from_transcription

my_original_dataset_name = "CoRal-project/coral-tts"
name_to_push_dataset_to = "syvai/coral-tts-zac-splits"

dsn = my_original_dataset_name

# Stream the dataset instead of downloading
ds = load_dataset(dsn, split="train", streaming=True)

# Take first item to get sample rate
first_item = next(iter(ds))
ds_sample_rate = first_item["audio"]["sampling_rate"]

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
model = model.to("cuda")

def tokenise_audio(waveform):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to("cuda")  # Changed from "cuda" to "mps"

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

def process_audio_splits(example):
    """
    Process one row and create 3 new rows (2 cumulative splits + full audio).
    Transcribe full audio once, then use timestamps to determine text_before and text_after.
    """
    processed_examples = []
    
    try:
        answer_audio = example.get("audio")
        if not answer_audio or "array" not in answer_audio:
            return processed_examples
            
        audio_array = answer_audio["array"]
        sample_rate = answer_audio["sampling_rate"]
        
        # Transcribe the full audio once to get timestamps
        full_transcription = transcribe_audio_array(audio_array, sample_rate)
        
        # Split audio into 3 parts (2 cumulative parts + full)
        audio_parts = split_audio_into_parts(audio_array, sample_rate)
        part_names = ["part1", "part2", "full"]
        
        # Calculate the duration of each part in seconds
        target_sample_rate = 24000
        part_durations = []
        for audio_part in audio_parts:
            duration = len(audio_part) / target_sample_rate
            part_durations.append(duration)
        
        for i, (audio_part, part_name, part_duration) in enumerate(zip(audio_parts, part_names, part_durations)):
            # Get words before and after the end of this audio part
            words_before_timestamp = ""
            words_after_timestamp_within_offset = ""
            
            # For the full audio, all words are "before" and none are "after"
            if part_name == "full":
                words_before_timestamp = get_text_from_transcription(full_transcription)
                words_after_timestamp_within_offset = ""
            else:
                # Extract words based on timestamp logic
                time_offset = 0.5  # 0.5 seconds offset for "after" words
                
                # Iterate through segments to find words before/after the part end time
                for segment in full_transcription.get("segments", []):
                    for word in segment.get("words", []):
                        word_end = word.get("end", 0)
                        word_text = word.get("text", "") + " "
                        
                        # Check if word ends before the part end time
                        if word_end < part_duration:
                            words_before_timestamp += word_text
                        # Check if word ends after part end but within offset
                        elif word_end < part_duration + time_offset:
                            words_after_timestamp_within_offset += word_text
                        else:
                            # Break early for efficiency
                            break
            
            # Tokenize the audio part
            try:
                codes_list = tokenise_audio(audio_part)
            except Exception as e:
                print(f"Skipping {part_name} due to tokenization error: {e}")
                continue
            
            # Create new example for this part
            new_example = {
                "text_before": words_before_timestamp.strip(),
                "text_after": words_after_timestamp_within_offset.strip(),
                "codes_list": codes_list,
                "part_type": part_name,
                "original_text": example.get("text", ""),
            }

            print(new_example)
            
            processed_examples.append(new_example)
            
    except Exception as e:
        print(f"Skipping row due to error: {e}")
    
    return processed_examples

# Process the streaming dataset
all_processed_examples = []
count = 0
max_examples = 10  # Limit for testing, remove this for full dataset

for example in ds:
    if count >= max_examples:
        break
        
    processed_examples = process_audio_splits(example)
    all_processed_examples.extend(processed_examples)
    count += 1
    
    if count % 10 == 0:
        print(f"Processed {count} examples, created {len(all_processed_examples)} rows")

# Convert to dataset
ds_processed = Dataset.from_list(all_processed_examples)

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

# Filter out None codes
ds_processed = ds_processed.filter(lambda x: x["codes_list"] is not None)
ds_processed = ds_processed.filter(lambda x: len(x["codes_list"]) > 0)

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

ds_processed = ds_processed.map(remove_duplicate_frames, num_proc=num_proc)

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

ds_processed = ds_processed.map(create_input_ids, num_proc=num_proc)

# Remove unnecessary columns
columns_to_keep = ["input_ids", "labels", "attention_mask", "part_type", "text_before", "text_after", "original_text"]
columns_to_remove = [col for col in ds_processed.column_names if col not in columns_to_keep]

ds_processed = ds_processed.remove_columns(columns_to_remove)

print(f"Final dataset size: {len(ds_processed)}")
print(f"Sample: {ds_processed[0]}")

ds_processed.push_to_hub(name_to_push_dataset_to)


