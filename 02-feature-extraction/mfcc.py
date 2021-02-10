from pydub import AudioSegment
import numpy as np
from scipy.fftpack import dct
import python_speech_features
# python_speech_features.fbank()

# If you want to see the spectrogram picture
import matplotlib
# matplotlib.use('Agg') 不显示
import matplotlib.pyplot as plt

def plot_spectrogram(spec, note,file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()
    plt.savefig(file_name)


def plot_show(arr):
    plt.plot([n for n in range(arr.size)], arr)
    plt.show()


#preemphasis config 
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=15kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

fs = 16000

# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] 
        frames[i,:] = frames[i,:] * win

    return frames

def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    return spectrum

def fbank(spectrum, num_filter = num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """
    mel = np.zeros([num_filter, spectrum.shape[1]])
    # 确定mel频率
    """
    获得filterbanks需要选择一个lower频率和upper频率，用300作为lower，8000
    作为upper是不错的选择。如果采样率是8000Hz那么upper频率应该限制为4000
    mel(f) = 2595*log10(1+f/700) 或者 mel(f) = 1125*ln(1+f/700)
    """
    lower = 2595*np.log10(1+(300/700))
    upper_hz = 8000 if fs > 16000 else fs/2
    upper = 2595*np.log10(1+(upper_hz/700))
    mel_fs = np.linspace(lower, upper, num_filter+2)
    # print(mel_fs)
    # 转换成频率
    mel_hz = (10**(mel_fs/2595)-1)*700
    # print(mel_hz)
    # 转换成滤波组
    # 对fft的大小进行缩放
    fft_bin = np.floor(mel_hz/upper_hz*spectrum.shape[1])
    # print(fft_bin)
    for i in range(0, num_filter):
        low = fft_bin[i-1]
        mid = fft_bin[i]
        upper = fft_bin[i+1]
        for j in range(int(low), int(mid)):
            mel[i][j] = (j - low) / (mid - low)
        for j in range(int(mid), int(upper)):
            mel[i][j] = (upper - j) / (upper - mid)

    #     plt.plot([n for n in range(mel[i].size)], mel[i])
    # plt.show()

    # 点积，得到23组滤波器的能量
    # feats.shape = [spectrum.shape[0], num_filter]
    feats = np.dot(spectrum, mel.T)
    feats = np.log10(feats)
    return feats

def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    r"""
    .. math::
        y_k =  f\sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)
        f = \begin{cases}
           \sqrt{\frac{1}{}} & \text{if }k=0, \\
           \sqrt{\frac{2}{N}} & \text{otherwise} \end{cases}
       
    """

    feats = np.zeros(fbank.shape)
    n = fbank.shape[1]
    for i in range(feats.shape[0]):
        for j in range(feats.shape[1]):
            for k in range(fbank.shape[1]):
                feats[i][j] += fbank[i][k]*np.cos(np.pi*j*(2*k+1)/(2*n))

        feats[i][0] *= np.sqrt(1/n)
        feats[i][1:] = feats[i][1:] * np.sqrt(2/n)

    return feats[:, 1:num_mfcc+1]


def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()





def main():
    wav = AudioSegment.from_wav('./test.wav')
    sound = wav.get_array_of_samples()
    arr = np.array(sound).astype(np.float32)
    m = np.max(arr)
    arr /= m
    # win = np.hamming(frame_len)
    # plot_show(win)
    # 预加重
    signal = preemphasis(arr)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    # 语谱图要取对数
    # d = 10*np.log10((np.abs(spectrum.T)*np.abs(spectrum.T)))
    # plot_spectrogram(d,"spectrum","111")
    # 用函数直接画语谱图
    # plt.specgram(arr, NFFT=256, Fs=fs, window=np.hanning(256))
    # plt.show()

    fbank_feats = fbank(spectrum)
    plot_spectrogram(fbank_feats.T, 'Filter Bank', 'fbank.png')
    write_file(fbank_feats,'./test.fbank')
    mfcc_feats = mfcc(fbank_feats)
    plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    write_file(mfcc_feats,'./test.mfcc')

if __name__ == '__main__':
    main()
