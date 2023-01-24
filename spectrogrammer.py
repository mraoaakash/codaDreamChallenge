
import librosa
import librosa.display
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import multiprocessing as mp
from multiprocessing import Pool
from PIL import Image


warnings.filterwarnings("ignore")

output_path = "/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/raw_data/spect"
basepath = "/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/raw_data/solicited_data"


def spectrogram():
    for filename in os.listdir(basepath):
        if filename.endswith(".wav"):
            print(filename)
            path = os.path.join(basepath, filename)
            x, sr = librosa.load(path, sr=44100)
            if librosa.get_duration(x, sr) == 0.5:
                x = np.pad(x, (0, 22050), 'constant')
            # print(librosa.get_duration(x, sr))
            S = librosa.feature.melspectrogram(x, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(2.24, 2.24))
            librosa.display.specshow(log_S, sr=sr,fmax=8000)
            plt.clim(-80,0)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"{output_path}/{filename[:-4]}.png")
            # resizing the plot to 224x224
            img = Image.open(f"{output_path}/{filename[:-4]}.png")
            img = img.resize((224, 224))
            img.save(f"{output_path}/{filename[:-4]}.png")
            plt.clf()

if __name__ == "__main__":
    spectrogram()  