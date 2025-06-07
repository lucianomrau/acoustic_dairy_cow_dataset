import numpy as np
from scipy import signal
import math
from collections import Counter
from .audio_signal import AudioSignal
from .constants import *
from .mlp import EventClassifier, ActivityClassifier


def run(file, **kwargs):
    return Bufar.run(file, **kwargs)


class Bufar:

    intake_type = ['numCHEW', 'numBITE', 'numCB', 'T', 'EnergiaChew', 'EnergiaBite', 'DurChew', 'DurBite', 'DurCB',
                   'AmpChew', 'AmpBite', 'AmpCB', 'IntChew', 'IntBite', 'IntCB']
    intake_dtype = np.dtype([(x, 'f4') for x in intake_type])

    event_stats = np.array([[1.81891, 0.97500, 25.41667, 0.25749], [1.36014, 0.23194, 7.27732, 0.17966]])
    event_tags = np.array(['c', 'b', 'x'])
    activity_tags = np.array(['Noise', 'Grazing', 'Rumination'])

    @classmethod
    def run(cls, file, debug=False):
        AudioSignal.check_audio_file(file)
        file_info = AudioSignal.get_file_info(file)
        freq = file_info.samplerate
        blocksize = BLOCK_DURATION * freq

        intake_total = {'chewEnergy': 0, 'biteEnergy': 0,
                        'numC': 0, 'numB': 0, 'numCB': 0,
                        'chew_duration': 0, 'bite_duration': 0, 'cb_duration': 0,
                        'chew_intensity': 0, 'bite_intensity': 0, 'cb_intensity': 0,
                        'chew_amplitude': 0, 'bite_amplitude': 0, 'cb_amplitude': 0
                        }

        n_block = 0
        activities = np.empty((0, 3))
        events = np.empty((0, 3))
        for block in AudioSignal.blocks(file, blocksize=blocksize):
            start = n_block * blocksize
            end = start + block.size

            features, intake_info, block_events = cls.extract_features(block, n_block, file_info)
            intake_total = Counter(intake_total) + Counter(intake_info)

            scaled_block = cls.scale_block(features)
            scaled_block = scaled_block.reshape(scaled_block.size, 1)

            block_activities = ActivityClassifier().classify(scaled_block)
            max_i = np.argmax(block_activities)
            activity = cls.activity_tags[max_i]

            # If activity is equal to last activity, just updates end
            if activities.size and activities[-1, 2] == activity:
                activities[-1, 1] = end/freq
            else:
                activities = np.append(activities, [[start/freq, end/freq, activity]], axis=0)

            events = np.append(events, block_events, axis=0)
            n_block += 1
        return activities, events, intake_total

    @classmethod
    def scale_block(cls, raw_block_features):
        std_features = np.array([[0.98053, 0.55620, 0.26060, 0.18320], [0.39316, 0.35622, 0.30806, 0.24945]])
        return (raw_block_features - std_features[0]) / std_features[1]

    @classmethod
    def extract_features(cls, block, n_block, file_info):
        freq = file_info.samplerate
        seconds = len(block) / freq
        rect_signal, a, processed_signal = cls.process_signal(block, freq)

        # Deteccion de picos
        peak_positions, silence_peak_positions = cls.detect_events(processed_signal)

        max_signal = np.max(processed_signal)

        signal_duration = np.zeros_like(processed_signal)
        signal_duration[:-1] = (processed_signal[:-1] >= (0.15 * max_signal)) + 0

        diff_sign = np.sign(np.diff(processed_signal))
        diff_sign = np.append(diff_sign, [0])

        signal_diff = signal_duration * diff_sign

        # Features
        peak_count = len(peak_positions)
        cross_zero = np.zeros(peak_count)
        duration = np.zeros(peak_count)
        amplitude = np.zeros(peak_count)
        relation = np.zeros(peak_count)
        verification_window = 0.2 * DECIMATE_FREQ
        for i, position in enumerate(peak_positions):
            start = max(0, int(position-verification_window) + 1)
            end = min(len(signal_diff), int(position+verification_window+1)) + 1
            window = signal_diff[start:end]
            cross_zero[i] = np.count_nonzero(np.abs(np.diff(window)) > 1)
            duration[i] = np.sum(signal_duration[start:end])
            if end < seconds*DECIMATE_FREQ:
                orig_start = math.floor((start * freq)/DECIMATE_FREQ)
                orig_end = math.floor((end * freq)/DECIMATE_FREQ)
                amplitude[i] = np.max(rect_signal[orig_start:orig_end])
                relation[i] = np.trapz(processed_signal[start:int(position)]) / np.trapz(processed_signal[start:end])

        # Classification
        raw_features = np.empty([4, peak_count])
        raw_features[0] = cls.standardize(np.transpose(cross_zero), cls.event_stats[0][0], cls.event_stats[1][0])
        raw_features[1] = cls.standardize(np.transpose(amplitude), cls.event_stats[0][1], cls.event_stats[1][1])
        raw_features[2] = cls.standardize(np.transpose(duration), cls.event_stats[0][2], cls.event_stats[1][2])
        raw_features[3] = cls.standardize(np.transpose(relation), cls.event_stats[0][3], cls.event_stats[1][3])

        recognized = EventClassifier().classify(raw_features)

        max_i = np.argmax(recognized, axis=0)
        event_tags = cls.event_tags[max_i]

        event_rate = peak_count / seconds
        peak_positions = peak_positions / DECIMATE_FREQ
        intensity = np.zeros(peak_count)
        for i, position in enumerate(peak_positions):
            start = int(round(position * DECIMATE_FREQ * DECIMATE_FACTOR) - 0.2 * freq)
            start = max(start, 1)
            end = int(round(position * DECIMATE_FREQ * DECIMATE_FACTOR) + 0.2 * freq)
            end = min(end, rect_signal.size)
            intensity[i] = np.mean(rect_signal[start:end])

        chew_positions = np.where(event_tags == 'c')
        bite_positions = np.where(event_tags == 'b')
        cb_positions = np.where(event_tags == 'x')

        chew_energy, bite_energy = cls.estimate_intake(rect_signal, peak_positions, event_tags, freq)

        intake = {
            'chewEnergy': chew_energy,
            'biteEnergy': bite_energy,
            'numC': len(chew_positions[0]),
            'numB': len(bite_positions[0]),
            'numCB': len(cb_positions[0]),
            'chew_duration': np.sum(duration[chew_positions]),
            'bite_duration': np.sum(duration[bite_positions]),
            'cb_duration': np.sum(duration[cb_positions]),
            'chew_intensity': np.sum(intensity[chew_positions]),
            'bite_intensity': np.sum(intensity[bite_positions]),
            'cb_intensity': np.sum(intensity[cb_positions]),
            'chew_amplitude': np.sum(amplitude[chew_positions]),
            'bite_amplitude': np.sum(amplitude[bite_positions]),
            'cb_amplitude': np.sum(amplitude[cb_positions])
        }

        features = np.empty(4)
        features[0] = event_rate
        if peak_count:
            features[1] = intake['numC']/peak_count
            features[2] = intake['numB']/peak_count
            features[3] = intake['numCB']/peak_count
        else:
            features[1] = features[2] = features[3] = 0

        events = np.empty([peak_count, 3], dtype=np.object)
        #events[:, 0] = peak_positions - 0.2 + n_block*300
        #events[:, 1] = peak_positions + 0.2 + n_block*300
        events[:, 0] = peak_positions - duration/DECIMATE_FREQ + n_block*300
        events[:, 1] = peak_positions + duration/DECIMATE_FREQ + n_block*300
        
        events[:, 2] = event_tags

        return features, intake, events

    @staticmethod
    def process_signal(block, freq):
        block = AudioSignal.adapt_base_filter(block, 0.01)
        block = AudioSignal.value_quantizier(block)

        # Rectificacion de la señal
        rect_signal = np.abs(block)
        # Aplicacion de filtro pasabajo
        b, a = signal.butter(2, LOW_PASS_FREQ / (freq / 2))
        filtered_block = signal.lfilter(b, a, rect_signal)
        # Submuestreo
        processed_signal = signal.decimate(filtered_block, DECIMATE_FACTOR, ftype='fir')

        return rect_signal, a, processed_signal

    @staticmethod
    def standardize(data, mean, sd):
        return (data-mean)/sd

    @classmethod
    def estimate_intake(cls, rect_signal, peak_positions, event_tags, freq):
        chew_energy = 0
        bite_energy = 0
        for i in range(0, len(peak_positions)):
            start = int(round(peak_positions[i]) - 0.2*freq)
            start = max(start, 1)
            end = int(round(peak_positions[i]) + 0.2*freq)
            end = min(end, len(rect_signal))

            if event_tags[i] == 'c':
                chew_energy += cls.get_energy(rect_signal[start: end])
            elif event_tags[i] == 'b':
                bite_energy += cls.get_energy(rect_signal[start: end])
            else:
                chew_energy += 2 * cls.get_energy(rect_signal[start: int(round(peak_positions[i]))])

        return chew_energy, bite_energy

    @staticmethod
    def get_energy(sig):
        return np.linalg.norm(sig) * np.linalg.norm(sig)

    @classmethod
    def detect_events(cls, processed_signal):
        n = len(processed_signal)
        sig_mean = np.mean(processed_signal)
        decay_factor = (DECAY_PERCENTAGE / 100) / (T_MAX - T_MIN)

        peak_positions = np.empty(0)
        silence_peak_positions = np.empty(0)

        Mm = np.max(processed_signal[:5 * DECIMATE_FREQ])
        MM = [Mm] * 5
        M = THRESHOLD_PERCENTAGE * (Mm - sig_mean) + sig_mean
        dM = M * decay_factor

        var_threshold = np.zeros(n)
        below_threshold = True
        ref = 0
        for i in range(n):
            ref += 1
            var_threshold[i] = M
            peak_threshold = (Mm - sig_mean) / 10 + sig_mean

            if not below_threshold and processed_signal[i] < M:
                below_threshold = True

            if ref > T_MIN:
                if ref < T_MAX:
                    M -= dM
                else:
                    silence_peak_positions = np.append(silence_peak_positions, i)

                if below_threshold and processed_signal[i] > var_threshold[i] and processed_signal[i] > peak_threshold and i > 1 and processed_signal[i] < processed_signal[i-2]:
                    peak_positions = np.append(peak_positions, i)
                    silence_peak_positions = np.append(silence_peak_positions, i)
                    MM[1:5] = MM[0:4]
                    MM[0] = processed_signal[i]

                    if MM[0] > 3 * Mm:
                        MM[0] = 2.2 * Mm

                    Mm = np.mean(MM)
                    M = THRESHOLD_PERCENTAGE * (Mm - sig_mean) + sig_mean
                    dM = M * decay_factor
                    ref = 0
                    below_threshold = False

        return peak_positions, silence_peak_positions
