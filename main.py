from xtts.xtts_utils import train_model, load_model, run_tts, preprocess_dataset
import os
import winsound
import sys

def train_model_default(directory, language, num_epochs=10, batch_size=4, grad_accum=1, max_audio_length=11):
    train_csv = os.path.join(directory, "metadata_train.csv")
    eval_csv = os.path.join(directory, "metadata_eval.csv")
    out_path = os.path.join(directory, "models/")
    train_model(language, os.path.abspath(train_csv), os.path.abspath(eval_csv), num_epochs, batch_size, grad_accum, out_path, max_audio_length)

def load_model_default(directory):
    models_dir = os.path.join(directory, "models")
    used_dir = ""
    for file in os.listdir(models_dir):
        filename = os.fsdecode(file)
        full_path = os.path.join(models_dir, filename)
        if os.path.isdir(full_path):
            for sub_file in os.listdir(full_path):
                subfilename = os.fsdecode(sub_file)
                if subfilename == "best_model.pth":
                    used_dir = full_path
                    break

    if used_dir.__len__() != 0:
        print("using model in folder: ", used_dir)
        best_model_path = os.path.join(used_dir, "best_model.pth")
        config_path = os.path.join(used_dir, "config.json")
        return load_model(os.path.abspath(best_model_path), os.path.abspath(config_path), os.path.abspath("data/XTTS_v2.0_original_model_files/vocab.json"))
    else:
        print("no model found")
        return None

# put the wav file in data/
# the rest will be handled by this function
def quick_train(wav_filename_without_ending, language):
    out_path = "data/" + wav_filename_without_ending
    wav_file = out_path + ".wav"
    preprocess_dataset([wav_file], language, out_path)
    train_model_default(out_path, language)

def run_tts_on_voice(voice, text, output_filepath=""):
    return run_tts(voice['model'], voice['language'], text, os.path.abspath(voice['audio_file']), output_filepath)

available_voices = [
        {"name": "MyVoice", "language": "de", "folder": "data/MyVoice/", "audio_file": "data/MyVoice/wavs/my_voice_0.wav", "model": None},
]


if __name__ == "__main__":
    language = "de"
    # preprocess_dataset(["data/example.wav"], language, "data/example")
    # train_model_default("data/example/", language)

    # model = load_model_default("data/example/")
    # log_info, generated_file, reference_file = run_tts(model, language, "Dies ist ein Beispiel.", os.path.abspath("data/example/wavs/example_00000000.wav"))
    # if generated_file.__len__() != 0:
    #      winsound.PlaySound(generated_file, winsound.SND_FILENAME)

    for voice in available_voices:
        voice['model'] = load_model_default(voice['folder'])

    while True:
        for line in sys.stdin:
            log_info, generated_file, reference_file = run_tts_on_voice(available_voices[0], line, "output/generated.wav")
            if generated_file.__len__() != 0:
                 winsound.PlaySound(generated_file, winsound.SND_FILENAME)

