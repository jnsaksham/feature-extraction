def block_audio(x,blockSize,hopSize,fs, w):
    
    # returns a matrix xb (dimension NumOfBlocks X blockSize) and a vector timeInSec (dimension NumOfBlocks) 
    # for blocking the input audio signal into overlapping blocks. 
    # timeInSec will refer to the start time of each block
    
    t = 0
    timeInSec = np.array([])
    xb = []
    samples = len(x)
    
    while t < samples:#-blockSize:
        if t <= samples-blockSize:
            block = x[t:t+blockSize]
        if t>samples-blockSize and t<samples:
            block = np.append(x[t:], np.zeros(blockSize-len(x[t:])))
        
        # Window the audio
        w = w/np.sum(w)
        block = block*w
        timeInSec = np.append(timeInSec, t/fs)
        xb.append(block)
        t += hopSize
    xb = np.array(xb)
    
    return xb, timeInSec


def fft_mag(x):
    # Analysis of a signal using the discrete Fourier transform
    # x: input signal, w: analysis window, N: FFT size 
    # returns spectrum of the block

    # Size of positive side of fft
    blockSize = len(x)
    relevant = (blockSize//2)+1                                   
    h1 = (blockSize+1)//2                                     
    h2 = blockSize//2                                         

    # Arrange audio to center the fft around zero
    x_arranged = np.zeros(blockSize)                         
    x_arranged[:h1] = x[h2:]                              
    x_arranged[-h2:] = x[:h2]        
    
    # compute fft
    X = scipy.fft(x_arranged)
    
    # compute magnitude spectrum in dB
    magX = abs(X[:relevant])                                 
    magX = 20 * np.log10(magX)
    
    return magX

def compute_stft(xb):
    # Generate spectrogram
    # returns magnitude spectrogram

#     if (H <= 0):                                   # raise error if hop size 0 or negative
#         raise ValueError("Hop size (H) smaller or equal to 0")

    blockSize = xb.shape[1]
    hopSize = int(blockSize/2)
    
    h1 = (blockSize+1)//2
    h2 = blockSize//2
    
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
    
def extract_rms(xb):    
    # Returns an array (NumOfBlocks X k) of spectral flux for all the audio blocks: k = frequency bins
    # xb is a matrix of blocked audio data (dimension NumOfBlocks X blockSize)

    rms_dB = []
    for block in xb:
        rms = np.sqrt(np.dot(block, block)/len(block))
        # Replace rms<-100dB with -100dB. -100dB = 10^(-5)
        if rms <= 10**(-5):
            rms = 10**(-5)
        rms = 20*np.log10(rms)
        rms_dB.append(rms)
    rms_dB = np.array(rms_dB)
    return rms_dB

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