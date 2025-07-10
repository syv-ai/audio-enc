# Audio Dataset Tokenizer

This tool processes audio datasets by splitting them into parts and tokenizing them using the SNAC model.

## Usage

Run the tokenizer with a YAML configuration file:

```bash
python tokenizer/tokenizer.py config.yaml
```

## Configuration Format

The configuration file uses YAML format and supports multiple datasets:

```yaml
datasets:
  - name: "dataset-owner/dataset-name"
    section: "section_name"  # Optional, use null if not needed
    split: "train"           # train, validation, test, etc.
    audio_column: "audio"    # Column name containing audio data
    text_column: "text"      # Column name containing text data
    output_name: "your-username/output-dataset-name"
```

## Configuration Parameters

- `name`: The Hugging Face dataset name (e.g., "CoRal-project/coral-v2")
- `section`: Dataset section/subset (optional, set to `null` if not needed)
- `split`: Which split to process ("train", "validation", "test", etc.)
- `audio_column`: Name of the column containing audio data
- `text_column`: Name of the column containing text/transcript data
- `output_name`: Name for the processed dataset to upload to Hugging Face Hub

## Example

See `config.yaml` for a complete example configuration.

## Dependencies

Make sure to install the required dependencies:

```bash
pip install torch datasets transformers torchaudio pyyaml
```

You'll also need the SNAC model and wav2vec2 dependencies for audio processing.
