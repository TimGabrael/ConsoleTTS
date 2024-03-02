import argparse
import os
import sys
import tempfile

import numpy as np

import os
import torch
import torchaudio
import traceback
from xtts.formatter import format_audio_list
from xtts.gpt_train import train_gpt

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

if not torch.cuda.is_available():
    torch.set_num_interop_threads(8)
    torch.set_num_threads(8)

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    xtts_model = None
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        print("You need to run the previous steps or manually set the `XTTS checkpoint path`, `XTTS config path`, and `XTTS vocab path` fields !!")
        return None
    config = XttsConfig()
    config.load_json(xtts_config)
    xtts_model = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    xtts_model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    if torch.cuda.is_available():
        xtts_model.cuda()

    print("Model Loaded!")
    return xtts_model

def run_tts(model, lang, tts_text, speaker_audio_file):
    if model is None or not speaker_audio_file:
        print("You need to run the previous step to load the model !!")
        return "You need to run the previous step to load the model !!", "", ""

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=model.config.gpt_cond_len, max_ref_length=model.config.max_ref_len, sound_norm_refs=model.config.sound_norm_refs)
    out = model.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=model.config.temperature, # Add custom parameters here
        length_penalty=model.config.length_penalty,
        repetition_penalty=model.config.repetition_penalty,
        top_k=model.config.top_k,
        top_p=model.config.top_p,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path, speaker_audio_file

def preprocess_dataset(audio_path_list, language, out_path):
    clear_gpu_cache()
    os.makedirs(out_path, exist_ok=True)
    if audio_path_list is None:
        print("You should provide one or multiple audio files! If you provided it, probably the upload of the files is not finished yet!")
        return False
    else:
        try:
            train_meta, eval_meta, audio_total_size = format_audio_list(audio_path_list, target_language=language, out_path=out_path)
        except:
            traceback.print_exc()
            error = traceback.format_exc()
            print(f"The data processing was interrupted due an error !! Please check the console to verify the full error message! \n Error summary: {error}")
            return False

    clear_gpu_cache()

    # if audio total len is less than 2 minutes raise an error
    if audio_total_size < 120:
        message = "The sum of the duration of the audios that you provided should be at least 2 minutes!"
        print(message)
        return False

    print("Dataset Processed!")
    return True

def train_model(language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    clear_gpu_cache()
    if not train_csv or not eval_csv:
        print("no train/eval csv provided")
        return False
    try:
        max_audio_length = int(max_audio_length * 22050)
        config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
    except:
        traceback.print_exc()
        error = traceback.format_exc()
        print(f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}")
        return False

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    print("Model training done!")
    clear_gpu_cache()
    return True



