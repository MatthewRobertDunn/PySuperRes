import numpy as np
import librosa
import sounddevice as sd


def normalize(sample):
    x1 = abs(sample.max())
    x2 = abs(sample.min())
    k = max(x1,x2)
    if k <= 1.0:
        return sample
    v = 1.0 / k
    sample = sample * v
    return sample

def scale10(source):
    return np.array(list(map(lambda audio_frame: (audio_frame + 1.0) / 2.0,source)), dtype=np.float32)

def unscale10(source):
    return np.array(list(map(lambda audio_frame: (audio_frame * 2.0) - 1.0,source)), dtype=np.float32)


def to_8_bit(source):
    return np.array(list(map(lambda audio_frame: int(audio_frame * 255.0) / 255.0, source)), dtype=np.float32)


def get_output_data(filename, source_len):
    dest, outrate = librosa.load(filename,mono=True, sr=48000)
    dest = normalize(dest)
    target_length = source_len * 3
    dest = np.array( dest[:target_length], dtype=np.float32)
    while len(dest) < target_length:
        dest = np.append(dest,dest[-1])

    #dest = scale10(dest)
    return dest.reshape(1, -1, 1, 3)

def get_input_data(filename):
    source, inrate = librosa.load(filename,mono=True, sr=16000)
    source = normalize(source)
    source = to_8_bit(source)
    #source = scale10(source)
    return source.reshape(1,-1,1)

def undo_output(output):
    res = output.reshape(-1)
    #res = unscale10(res)
    return res
