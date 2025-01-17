import os
import tomli
import pandas as pd
import argparse
import numpy as np
import glob
import time
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModel

with open("config.toml", "rb") as f:
    config = tomli.load(f)

def load_model(model_name):
    model_path = config["PATH_TO_PRETRAINED_MODELS"] 
    model_path = os.path.join(model_path, model_name)
    assert os.path.exists(model_path), f"Model path {model_path} does not exist."
    if model_name in config["WHISPER"]:
        feature_extractor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        model = AutoModel.from_pretrained(model_path)
    elif model_name in config["AM"]:
        feature_extractor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        model = AutoModel.from_pretrained(model_path)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    model = model.to('cuda')

    return feature_extractor, model

def extract(model_name, audio_files, save_dir, feature_level):
    print('='*30 + f"Extracting features with {model_name}" + '='*30)
    start_time = time.time()
    print('Loading pre-trained tokenizer and model...')
    feature_extractor, model = load_model(model_name)

    for idx, audio_file in enumerate(audio_files,1):
        name = os.path.basename(audio_file).split('.')[0]
        samples, sr = sf.read(audio_file)
        assert sr == 16000, "Sample rate must be 16kHz."
        print(f'Processing {idx}/{len(audio_files)}: {name}')

        with torch.no_grad():
            if model_name in config["WHISPER"]:
                input_features = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_features
                decoder_input_ids = torch.tensor([[1,1]]) * model.config.decoder_start_token_id
                input_features = input_features.to('cuda')
                decoder_input_ids = decoder_input_ids.to('cuda')
                last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
                assert last_hidden_state.shape[0] == 1
                feature = last_hidden_state.squeeze(0).cpu().numpy() #(2, D)
            else:
                layer_ids = [-4, -3, -2, -1]
                input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values
                # none split into batch
                input_values = input_values.to('cuda')
                hidden_states = model(input_values, output_hidden_states=True).hidden_states
                feature = torch.stack(hidden_states)[layer_ids].sum(dim=0)
                bsize, segnum, featdim = feature.shape
                feature = feature.view(-1, featdim).detach().squeeze().cpu().numpy() # (T, D)
        audio_feature = os.path.join(save_dir, f"{name}.npy")
        if feature_level == 'FRAME':
            np.save(audio_feature, feature)
        else:
            feature = np.array(feature).squeeze()
            if len(feature.shape) != 1:
                feature = feature.mean(axis=0)
            np.save(audio_feature, feature)
    
    end_time = time.time()
    print(f'Total {len(audio_files)} audio files are processed in {end_time-start_time:.1f}s.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to load")
    parser.add_argument('--feature_level', type=str, default='FRAME', help='FRAME or UTT(UTTERANCE)')
    parser.add_argument('--dataset', type=str, help='input dataset')
    args = parser.parse_args()
    audio_dir = config[args.dataset]["PATH_TO_AUDIO"]
    save_dir = config[args.dataset]["PATH_TO_FEATURE"]

    assert torch.cuda.is_available(), "CUDA is currently not available."
    torch.cuda.empty_cache()
    assert args.feature_level in ['FRAME', 'UTT'], "Feature level must be either FRAME or UTT(UTTERANCE)."

    audio_files = glob.glob(audio_dir + "/*.wav")
    print(f'Find total "{len(audio_files)}" audio files.')

    dir_name = f"{args.model}-{args.feature_level[:3]}"
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    extract(args.model, audio_files, save_dir, args.feature_level)
