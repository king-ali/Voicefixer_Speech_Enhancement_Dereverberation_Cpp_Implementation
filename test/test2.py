import git
import os
import sys
import librosa
import numpy as np
import torch
import time
import librosa.display
from voicefixer.tools.pytorch_util import *
from voicefixer.tools.wav import *
from voicefixer.restorer.model import VoiceFixer as voicefixer_fe
import os


git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)
print("git root", git_root)
from voicefixer import VoiceFixer, Vocoder


def load_wav(path, sample_rate, threshold=0.95):
    wav_10k, _ = librosa.load(path, sr=sample_rate)
    return wav_10k
# os.makedirs(os.path.join(git_root, "test/new_output/output"), exist_ok=True)

print("Initializing VoiceFixer...")
voicefixer = VoiceFixer()

mode = 0
inputpath = os.path.join(git_root, "test/new_output/audios/4.wav")
outputpath = os.path.join(git_root, "test/new_output/output_audios/result_4" + ".wav")

wav_10k = load_wav(inputpath, sample_rate=44100)

print("wav_10k", wav_10k.shape)
# print("type", type(wav_10k.shape))
start_time = time.time()

out_np_wav = voicefixer.new_restore(
        input=wav_10k,  # low quality .wav/.flac file
    cuda=False,  # GPU acceleration
        mode=mode,
)

end_time = time.time()

print("time taken", end_time-start_time)
print(out_np_wav.shape)
print(type(out_np_wav))

save_wave(out_np_wav, fname=outputpath, sample_rate=44100)