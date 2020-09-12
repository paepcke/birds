import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os


def create_spectrogram(file, n_mels=128):
    spectrogramfile = 'home/data/birds/Birdsong_Spectrograms_Augmented/' + file[:len(file) - 4] + '.png'
    audio, sr = librosa.load(file)
    mel = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(6.07, 2.02))
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', cmap='gray_r')
    plt.tight_layout()
    plt.axis('off')

    plt.savefig(spectrogramfile, dpi=100, bbox_inches='tight', pad_inches=0, format='png', facecolor='none')
    plt.close()


if __name__ == "__main__":
    wav_dir = "home/data/birds/Birdsong_Filtered_Augmented/"
    for wav in os.listdir(wav_dir):
        create_spectrogram(wav)

