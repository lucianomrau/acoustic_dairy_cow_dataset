import soundfile
import numpy as np
from scipy import signal
from .constants import *


class AudioSignal:

    @staticmethod
    def check_audio_file(file):
        info = AudioSignal.get_file_info(file)
        if info.channels > 1:
            raise Exception('Input file has more than one channel. You should provide a single channel recording.')
        if info.samplerate != 44100:
            raise Exception('Input file must have a sampling frequency of 44100 Hz.')

    @staticmethod
    def get_file_info(file):
        file.seek(0)
        return soundfile.info(file)

    @staticmethod
    def blocks(file, blocksize=None, start=0, stop=None):
        file.seek(0)
        return soundfile.blocks(file, blocksize=blocksize, start=start, stop=stop)

    @staticmethod
    def adapt_base_filter(x, mu):
        a = np.array([1, 2*mu-1])
        b = np.array([0, 2*mu])
        w, _ = signal.lfilter(b, a, x, zi=x[0:1])
        return x-w

    @staticmethod
    def value_quantizier(block):
        if N_BITS < 2:
            raise Exception('N is out of range. Must be > 2')
        min_val = min(block)
        max_val = max(block)

        # Quantize
        quanta_size = (max_val-min_val) / ((1 << N_BITS) - 1)
        quant_values = (block - min_val) / quanta_size

        encoded_values = np.around(quant_values)

        return encoded_values * quanta_size + min_val

