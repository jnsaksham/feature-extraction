## libraries ##

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile as sp
import glob, os
import IPython.display as ipd
import time
import math

## Blocking and stft computation ##

def  block_audio(x,blockSize,hopSize,fs):    
    # allocate memory    
    numBlocks = math.ceil(x.size / hopSize)    
    xb = np.zeros([numBlocks, blockSize])    
    # compute time stamps    
    t = (np.arange(0, numBlocks) * hopSize) / fs    
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)    
    for n in range(0, numBlocks):        
        i_start = n * hopSize        
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])        
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]    
    return (xb,t)


def fft_mag(x):
    # Analysis of a signal using the discrete Fourier transform
    # x: input signal, w: analysis window, N: FFT size 
    # returns spectrum of the block
    
    # Size of positive side of fft
    blockSize = len(x)
    
    # Define window
    w = np.hanning(blockSize)
    w = w/np.sum(w)
    
    x = x*w
    
    relevant = (blockSize//2)+1
    h1 = (blockSize+1)//2
    h2 = blockSize//2
    
    # Arrange audio to center the fft around zero
    x_arranged = np.zeros(blockSize)                         
    x_arranged[:h1] = x[h2:]                              
    x_arranged[-h2:] = x[:h2]
    
    # compute fft and keep the relevant part
    X = np.fft.fft(x_arranged)[:relevant]
    
    # compute magnitude spectrum in dB
    magX = abs(X)
    
    return magX
def compute_stft(xb):
    # Generate spectrogram
    # returns magnitude spectrogram

    blockSize = xb.shape[1]
    hopSize = int(blockSize/2)
    
    mag_spectrogram = []
    for block in xb:
        magX = fft_mag(block)
        mag_spectrogram.append(np.array(magX))

    mag_spectrogram = np.array(mag_spectrogram)
    return mag_spectrogram

def plot_spectrogram(spectrogram, rate, hopSize):
    
    t = hopSize*np.arange(spectrogram.shape[0])/fs
    f = np.arange(0,fs/2, fs/2/spectrogram.shape[1])

    plt.figure(figsize = (15, 7))
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')
    plt.pcolormesh(t, f, spectrogram.T)
    plt.show()

def plot_spectrogram(spectrogram, fs, hopSize):
    
    t = hopSize*np.arange(spectrogram.shape[0])/fs
    f = np.arange(0,fs/2, fs/2/spectrogram.shape[1])

    plt.figure(figsize = (15, 7))
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')
    plt.pcolormesh(t, f, spectrogram.T)
    plt.show()
    
## Features ##
    
def extract_rms(xb):    
    # xb is a matrix of blocked audio data (dimension NumOfBlocks X blockSize)

    rms_array = np.array([])
    for block in xb:
        rms = np.sqrt(np.dot(block, block)/len(block))
        # Replace rms<-100dB with -100dB. -100dB = 10^(-5)
        if rms <= 10**(-5):
            rms = 10**(-5)
        rms_array = np.append(rms_array, rms)
        
    rms_array = 20*np.log10(rms_array)
    return rms_array

def extract_spectral_crest(xb):
    
    # Calculate STFT
    X = compute_stft(xb)
    
    spectral_crest = np.array([])
    for frame in X:
        crest_factor = np.max(abs(frame))/np.sum(abs(frame))
        spectral_crest = np.append(spectral_crest, crest_factor)
    
    return spectral_crest

def extract_spectral_flux(xb):    
    # Returns an array (NumOfBlocks X k) of spectral flux for all the audio blocks: k = frequency bins
    # xb is a matrix of blocked audio data (dimension NumOfBlocks X blockSize)
    
    X = compute_stft(xb)
    
    # Compute spectral flux
    # Initialise blockNum and freqIndex
    n = 0
    k = 0

    spectral_flux = []

    for n in np.arange(X.shape[0]-1):
        flux_frame = 0
        for k in np.arange(X.shape[1]):
            flux = (abs(X[n+1, k]) - abs(X[n, k]))**2
            flux_frame += flux
        flux_frame = np.sqrt(flux_frame)/(xb.shape[1]+1)
        spectral_flux.append(flux_frame)
    spectral_flux = np.array(spectral_flux)

    return spectral_flux

def extract_spectral_centroid(xb, fs):
    xb_afterFFT = np.zeros((np.shape(xb)[0], np.shape(xb)[1]))
    for flag, i in enumerate(xb):
        iw = i * np.hanning(np.shape(xb)[1])
        iw = scipy.fftpack.fft(iw)
        xb_afterFFT[flag] = iw
    xb_afterFFT = np.absolute(xb_afterFFT)
    vsc = np.zeros(np.shape(xb_afterFFT)[0])
    half_k = np.shape(xb_afterFFT)[1]/2
    for flag_centroid, block in enumerate(xb_afterFFT):
        left = math.log(np.arange(np.shape(xb_afterFFT)[1]/2)*flag_centroid/fs, 2)
        vsc[flag_centroid] = np.sum(left * xb_afterFFT[flag_centroid, 0:half_k]) / np.sum(xb_afterFFT[flag_centroid, 0:half_k])
    return vsc


def extract_zerocrossingrate(xb):
    vzc = np.zeros(np.shape(xb)[0])
    for n, block in enumerate(xb):
        vzc[n] = 0.5 * np.mean(np.abs(np.diff(np.sign[block])))
    return vzc

## Wrapper functions ##

def extract_features(x, blockSize, hopSize, fs):
    block, timeInSec = block_audio(x, blockSize, hopSize, fs)
    Vsc = extract_spectral_centroid(block, fs)
    Vzc = extract_zerocrossingrate(block)
    features = np.vstack(Vsc, Vzc)
    return features


def aggregate_feature_per_file(features):
    feature_matrix = np.zeros(10)
    for n, element in enumerate(features):
        feature_matrix[n] = np.mean(element)
        feature_matrix[n+5] = np.std(element)
    return feature_matrix


def get_feature_data(path, blockSize, hopSize):
    all_feature_matrix = np.zeros((0, 10))
    folder = glob.glob(path + "\\*.wav")
    for count, wavFile in enumerate(folder):
        fs, x = scipy.io.wavfile.read(wavFile)
        all_feature_matrix = np.vstack((all_feature_matrix, aggregate_feature_per_file(extract_features(x, blockSize, hopSize, fs))))
    return all_feature_matrix
