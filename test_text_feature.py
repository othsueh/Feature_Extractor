"""
Modified from text_feature.py, which is used to extract features from text data generated by generate_test_transcript.py.
"""
import os
import tomli
import pandas as pd
import argparse
import numpy as np
import glob
import time
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM

with open("config.toml", "rb") as f:
    config = tomli.load(f)

def load_model(model_name):
    model_path = config["PATH_TO_PRETRAINED_MODELS"] 
    model_path = os.path.join(model_path, model_name)
    assert os.path.exists(model_path), f"Model path {model_path} does not exist."
    if model_name in config["LLM"]:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = model.half()
    elif model_name in config["LM"]:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForMaskedLM.from_pretrained(model_path) # Temp change to fix roberta
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    model = model.to('cuda')

    return tokenizer, model

def find_start_end_pos(tokenizer):
    sentence = '今天天氣真好' # No space in sentence
    input_ids = tokenizer(sentence, return_tensors='pt')['input_ids'][0]
    start, end = None, None

    # find start, must in range [0, 1, 2]
    for start in range(0, 3, 1):
        # When decoding, there are some problems with the space, so we need to remove the space
        outputs = tokenizer.decode(input_ids[start:]).replace(' ', '')
        if outputs == sentence:
            print (f'start: {start};  end: {end}')
            return start, None

        if outputs.startswith(sentence):
            break
   
    # find end, must in range [-1, -2]
    for end in range(-1, -3, -1):
        outputs = tokenizer.decode(input_ids[start:end]).replace(' ', '')
        if outputs == sentence:
            break
    
    assert tokenizer.decode(input_ids[start:end]).replace(' ', '') == sentence
    print (f'start: {start};  end: {end}')
    return start, end


# 找到 batch_pos and feature_dim
def find_batchpos_embdim(tokenizer, model):
    sentence = '今天天氣真好'
    inputs = tokenizer(sentence, return_tensors='pt')
    inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
        outputs = torch.stack(outputs)[[-1]].sum(dim=0) # sum => [batch, T, D=768]
        outputs = outputs.cpu().numpy() # (B, T, D) or (T, B, D)
        batch_pos = None
        if outputs.shape[0] == 1:
            batch_pos = 0
        if outputs.shape[1] == 1:
            batch_pos = 1
        assert batch_pos in [0, 1]
        feature_dim = outputs.shape[2]
    print (f'batch_pos:{batch_pos}, feature_dim:{feature_dim}')
    return batch_pos, feature_dim

def extract(model_name, text_files, save_dir, feature_level):
    print('='*30 + f"Extracting features with {model_name}" + '='*30)
    start_time = time.time()
    layer_ids = [-4,-3,-2,-1]
    print('Loading pre-trained tokenizer and model...')
    tokenizer, model = load_model(model_name)

    start, end = find_start_end_pos(tokenizer)
    batch_pos, feature_dim = find_batchpos_embdim(tokenizer, model)
    print(f"Start: {start}; End: {end}; Batch_pos: {batch_pos}; Feature_dim: {feature_dim}")

    for idx, text_file in enumerate(text_files,1):
        embeddings = []
        name = os.path.basename(text_file).split('.')[0]
        sentence = open(text_file, 'r').read()
        print(f'Processing {idx}/{len(text_files)}: {name}')
        if len(sentence) == 0:
            print(f"Empty sentence in {text_file}")
            continue
        inputs = tokenizer(sentence, return_tensors='pt')
        inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True).hidden_states
            outputs = torch.stack(outputs)[layer_ids].sum(dim=0) # sum => [batch, T, D=768]
            outputs = outputs.cpu().numpy() # (B, T, D) or (T, B, D)
            if batch_pos == 1:
                embeddings = outputs[start:end,0]
            elif batch_pos == 0:
                embeddings = outputs[0,start:end]
        text_feature = os.path.join(save_dir, f"{name}.npy")
        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, feature_dim))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            np.save(text_feature, embeddings)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((feature_dim,))
            elif len(embeddings.shape) == 2:
                embeddings = embeddings.mean(axis=0)
            np.save(text_feature, embeddings)
    end_time = time.time()
    print(f'Total {len(text_files)} text files are processed in {end_time-start_time:.1f}s.')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name to load")
    parser.add_argument('--feature_level', type=str, default='FRAME', help='FRAME or UTT(UTTERANCE)')
    parser.add_argument('--dataset', type=str, help='input dataset')
    args = parser.parse_args()
    test_dir = config[args.dataset]["PATH_TO_TEST"]
    text_dfi = config[args.dataset]["PATH_TO_TRANSCRIPT"]
    save_dir = config[args.dataset]["PATH_TO_FEATURE"]

    assert torch.cuda.is_available(), "CUDA is currently not available."
    torch.cuda.empty_cache()
    assert args.feature_level in ['FRAME', 'UTT'], "Feature level must be either FRAME or UTT(UTTERANCE)."

    
    # text_files
    test_df = pd.read_csv(test_dir)
    text_files = []
    for idx, row in test_df.iterrows():
        text_file = os.path.join(text_dfi, row["FileName"] + ".txt")
        if not os.path.exists(text_file):
            print(f"File {text_file} does not exist.")
            raise FileNotFoundError
        text_files.append(text_file)
    print(f'Find total "{len(text_files)}" text files.')
    dir_name = f"{args.model}-{args.feature_level[:3]}"
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    extract(args.model, text_files, save_dir, args.feature_level)
