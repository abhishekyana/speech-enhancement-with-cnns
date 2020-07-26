import torch
import torch.nn as nn
import numpy as np
import scipy
import librosa
import os
import soundfile as sf
from tqdm import tqdm 
import argparse

# CONSTANTS
SAMPLE_RATE = 8000
WINDOW_LENGTH = 256
OVERLAP = 64
FFT_LENGTH = WINDOW_LENGTH

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block1 = [
            nn.Conv2d(1, 18, kernel_size=(9, 8), padding=(4, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[5, 1], padding=((5-1)//2, 0), bias=False),
        ]
        block2 = [
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 18, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
        ]
        block3 = [
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 18, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 18, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
        ]
        block4 = [
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 18, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
        ]
        block5 = [
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(8, 1, kernel_size=[129, 1], padding=((129-1)//2, 0), bias=False),
        ]
        self.block1 = torch.nn.Sequential(*block1)
        self.block2 = torch.nn.Sequential(*block2)
        self.block3 = torch.nn.Sequential(*block3)
        self.block4 = torch.nn.Sequential(*block4)
        self.block5 = torch.nn.Sequential(*block5)
    
    def forward(self, X):
        skip0 = self.block1(X)
        skip1 = self.block2(skip0)
        out = self.block3(skip1)
        out = self.block4(out + skip1)
        out = self.block5(out + skip0)
        return out


def get_stft(audio):
    return librosa.stft(audio, 
                        n_fft=FFT_LENGTH, 
                        win_length=WINDOW_LENGTH, 
                        hop_length=OVERLAP, 
                        window=scipy.signal.hamming(256, sym=False),
                        center=True)


def make_input_windows(stft_features, num_segments=8, num_features=129):
    noisy_stft = np.concatenate([stft_features[:, 0:num_segments - 1], stft_features], axis=1)
    stft_segments = np.zeros((num_features, num_segments, noisy_stft.shape[1] - num_segments + 1))

    for i in range(noisy_stft.shape[1] - num_segments + 1):
        stft_segments[:, :, i] = noisy_stft[:, i:i + num_segments]
    return stft_segments

def get_stft(audio):
    return librosa.stft(audio, 
                        n_fft=FFT_LENGTH, 
                        win_length=WINDOW_LENGTH, 
                        hop_length=OVERLAP, 
                        window=scipy.signal.hamming(256, sym=False),
                        center=True)


def make_input_windows(stft_features, num_segments=8, num_features=129):
    noisy_stft = np.concatenate([stft_features[:, 0:num_segments - 1], stft_features], axis=1)
    stft_segments = np.zeros((num_features, num_segments, noisy_stft.shape[1] - num_segments + 1))

    for i in range(noisy_stft.shape[1] - num_segments + 1):
        stft_segments[:, :, i] = noisy_stft[:, i:i + num_segments]
    return stft_segments

def stft_to_audio(features, phase, window_length, overlap):
    features = np.squeeze(features)
    features = features * np.exp(1j * phase)
    features = features.transpose(1, 0)
    return librosa.istft(features, win_length=window_length, hop_length=overlap)


def clean_audio_waveform(testing_audio, mymodel, cuda=False, msize=2**9):
    testing_audio_stft = get_stft(testing_audio)
    testing_audio_mag, testing_audio_phase = np.abs(testing_audio_stft), np.angle(testing_audio_stft)
    testing_audio_input_windows = make_input_windows(testing_audio_mag)
    fs, ss, m = testing_audio_input_windows.shape
    Tmp = []
    for i in tqdm(range(0, m, msize)):
        testing_tensor = torch.Tensor(testing_audio_input_windows[:, :, i:i+msize]).permute(2, 0, 1)
        if cuda and torch.cuda.is_available():
            testing_tensor = testing_tensor.cuda()
        testing_prediction = mymodel(testing_tensor.unsqueeze(1))
        clean_testing = testing_prediction.squeeze().cpu().detach().numpy()
        clean_testing_audio = stft_to_audio(clean_testing, testing_audio_phase[:, i:i+msize].T, WINDOW_LENGTH, OVERLAP)
        Tmp.append(clean_testing_audio)
    return np.concatenate(Tmp)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Speech Enhancement w/ CNNs by Abhishek Yanamandra")
    parser.add_argument('-in', action='store', dest='infile')
    parser.add_argument('-model', action='store', dest='modelpath')
    parser.add_argument('-cuda', action='store_true', default=False)
    parser.add_argument('-msize', action='store', type=int, default=2**9)
    args = parser.parse_args()
    print("Infile is ", args.infile)
    print("ModelPath is ", args.modelpath)
    print("CUDA: ", args.cuda)
    print("msize: ", args.msize)
    print(f"Loading the torch model from {args.modelpath}")
    mymodel = MyModel()
    mymodel.load_state_dict(torch.load(args.modelpath))
    if args.cuda:
        mymodel.cuda()
    print(mymodel)
    testing_audio, sr = librosa.load(args.infile, sr=SAMPLE_RATE)
    clean_audio = clean_audio_waveform(testing_audio, mymodel, args.cuda, args.msize)
    sf.write(f'AudioOuts/clean_{args.infile}', clean_audio, samplerate=SAMPLE_RATE)
    print("Done")