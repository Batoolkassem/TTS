import os
import gc
import torchaudio
import pandas as pd
from faster_whisper import WhisperModel
from glob import glob
from tqdm import tqdm
import torch
from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

torch.set_num_threads(16)

audio_types = (".wav", ".mp3", ".flac")

def list_audios(basePath, contains=None):
    return list_files(basePath, validExts=audio_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    for rootDir, dirNames, filenames in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.rfind("."):].lower()
            if validExts is None or ext.endswith(validExts):
                audioPath = os.path.join(rootDir, filename)
                yield audioPath

def format_audio_list(audio_files, target_language="ar", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", gradio_progress=None):
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Whisper Model!")
    asr_model = WhisperModel("large-v2", device=device, compute_type="float16")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
        segments = list(segments)

        # Debug output
        print(f"Transcription for {audio_path}: {[segment.text for segment in segments]}")

        i = 0
        sentence = ""
        sentence_start = None
        first_word = True
        words_list = []

        for _, segment in enumerate(segments):
            words = list(segment.words)
            words_list.extend(words)

        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start = word.start
                if word_idx == 0:
                    sentence_start = max(sentence_start - buffer, 0)
                else:
                    previous_word_end = words_list[word_idx - 1].end
                    sentence_start = max(sentence_start - buffer, (previous_word_end + sentence_start) / 2)
                sentence = word.word
                first_word = False
            else:
                sentence += " " + word.word

            if word.word[-1] in ["!", ".", "؟", "؟"] or word_idx == len(words_list) - 1:
                sentence = multilingual_cleaners(sentence.strip(), target_language)
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
                audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                if word_idx + 1 < len(words_list):
                    next_word_start = words_list[word_idx + 1].start
                else:
                    next_word_start = (wav.shape[0] - 1) / sr

                word_end = min((word.end + next_word_start) / 2, word.end + buffer)
                
                absoulte_path = os.path.join(out_path, audio_file)
                os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
                i += 1
                first_word = True

                audio = wav[int(sr * sentence_start):int(sr * word_end)].unsqueeze(0)
                if audio.size(-1) >= sr / 3:
                    torchaudio.save(absoulte_path, audio, sr)
                else:
                    continue

                metadata["audio_file"].append(audio_file)
                metadata["text"].append(sentence.strip())
                metadata["speaker_name"].append(speaker_name)

    df = pd.DataFrame(metadata)
    df = df.sample(frac=1).reset_index(drop=True)
    num_val_samples = int(len(df) * eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    df_train = df_train.sort_values('audio_file')
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False, encoding='utf-8-sig')

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval = df_eval.sort_values('audio_file')
    df_eval.to_csv(eval_metadata_path, sep="|", index=False, encoding='utf-8-sig')

    del asr_model, df_train, df_eval, df, metadata
    gc.collect()

    return train_metadata_path, eval_metadata_path, audio_total_size
