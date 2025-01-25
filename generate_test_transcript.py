import tomli
import os
import pandas as pd
import torch
from transformers import AutoProcessor, pipeline, AutoModelForSpeechSeq2Seq
from display import *

corpus = "MSPPODCAST"   
model_name = "whisper-large-v3"

# Load config file
config = tomli.load(open("config.toml", "rb"))
# Load related file path
model_path = config["PATH_TO_PRETRAINED_MODELS"]
audio_path = config[corpus]['PATH_TO_AUDIO']
transcript_path = config[corpus]['PATH_TO_TRANSCRIPT']

# Load test file
test_df = pd.read_csv(config[corpus]["PATH_TO_TEST"])

if __name__ == "__main__":
    # Load audio feature extractor
    print('='*30 + f' Loading {model_name} ' + '='*30)
    model_id = os.path.join(model_path, model_name)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.cuda()

    # Load ASR pipeline
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=0,
    )
    total_length = len(test_df)
    # Generate transcript
    for idx, row in test_df.iterrows():
        audio = os.path.join(audio_path, row["FileName"] + ".wav")
        transcript = asr(audio)["text"]
        # Write transcript to file
        output_file = os.path.join(transcript_path, row["FileName"] + ".txt")
        with open(output_file, "w") as f:
            f.write(transcript)
        progress_bar(idx+1, total_length, msg=f'Processed {row["FileName"]}.wav')