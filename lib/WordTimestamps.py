import json
import os
import sys
import wave

import fastdtw
import librosa
import numpy as np
from pydub import AudioSegment
from scipy.spatial.distance import euclidean
from vosk import KaldiRecognizer, Model


class Word:
    """A class representing a word from the JSON format for vosk speech recognition API"""

    def __init__(self, dict):
        """
        Parameters:
          dict (dict) dictionary from JSON, containing:
            conf (float): degree of confidence, from 0 to 1
            end (float): end time of the pronouncing the word, in seconds
            start (float): start time of the pronouncing the word, in seconds
            word (str): recognized word
        """

        self.conf = dict["conf"]
        self.end = dict["end"]
        self.start = dict["start"]
        self.word = dict["word"]
        self.audio = None

    def __sub__(a, b):
        if a.word != b.word:

            raise Exception(
                'Words not match, comparing "{}" and "{}"'.format(a.word, b.word)
            )

        return {
            "start": a.start - b.start,
            "end": a.end - b.end,
            "length": abs((b.end - b.start) - (a.end - a.start)),
        }

    def __str__(self) -> str:
        return "{:.02f} {:.02f} : {}\n".format(self.start, self.end, self.word)


class RecognizerResult:
    def __init__(self, dict) -> None:
        self.result = [Word(r) for r in dict["result"]]
        self.length = len(self.result)
        self.transcript = dict["text"]

    # def __sub__(a, b):
    #     """strict mode Subtract"""
    #     if a.trnascript != b.trnascript or a.length != b.length:
    #         # TODO:  當辨識結果不相符，先 match 詞句後再取差值
    #         words_a, words_b = a.trnascript.split(""), b.trnascript.split("")
    #         match_index = []
    #         raise Exception(
    #             'Words not match, comparing "{}" and "{}"'.format(
    #                 a.trnascript, b.trnascript
    #             )
    #         )

    #     else:
    #         result = []
    #         for i, j in zip(a.result, b.result):
    #             d = j - i
    #             d["word"] = i.word
    #             result.append(d)

    #         return result

    def __sub__(a, b):
        """fuzzy mode Subtract"""
        if a.trnascript != b.trnascript or a.length != b.length:
            raise Exception(
                'Words not match, comparing "{}" and "{}"'.format(
                    a.trnascript, b.trnascript
                )
            )

        else:
            result = []
            for i, j in zip(a.result, b.result):
                d = j - i
                d["word"] = i.word
                result.append(d)

            return result

    def __str__(self):

        return "".join(str(result) for result in self.result)

    def __getitem__(self, index):
        return self.result[index]


class Recognizer:
    def __init__(self, model_path="") -> None:

        model_path = "/workspace/persistent/Projects/Japanese-pronounce-project/lib/vosk-model-ja-0.22"

        if not os.path.exists(model_path):
            print(
                f"Please download the model from https://alphacephei.com/vosk/models and unpack as {model_path}"
            )
            sys.exit()

        print(f"Reading your vosk model '{model_path}'...")
        self.model = Model(model_path)
        print(f"'{model_path}' model was successfully read")

    def make_temp_file(self, audio_file) -> str:
        sound = AudioSegment.from_file(audio_file)
        temp_file = os.path.split(audio_file)[0] + "/temp.wav"
        sound.export(
            temp_file, format="wav", parameters=["-ac", "1"]
        )  # mono form required

        return temp_file

    def read_file(self, audio_file):
        # read file
        if not os.path.exists(audio_file):
            print("File '{audio_file}' doesn't exist")
            sys.exit()

        if os.path.splitext(audio_file)[1] != "wav":
            audio_file = self.make_temp_file(audio_file)

        wf = wave.open(audio_file, "rb")

        # check if audio is mono wav
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getcomptype() != "NONE"
        ):
            print("Audio file must be WAV format mono PCM.")
            sys.exit()

        return wf

    def recognize(self, audio_file: str) -> list:

        wf = self.read_file(audio_file)

        results = []

        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(True)

        # recognize speech using vosk model
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                results.append(part_result)

        part_result = json.loads(rec.FinalResult())
        results.append(part_result)

        return results[0]


def main():
    recognizer = Recognizer()

    audio_flienames = [
        "/workspace/persistent/japanese project/jvs_ver1/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav",
        "/workspace/persistent/japanese project/jvs_ver1/jvs002/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav",
        # "/workspace/persistent/japanese project/ohayo.wav",
        # "/workspace/persistent/japanese project/01-01-rec.m4a",
    ]

    # time and waveform analysis
    r1 = RecognizerResult(recognizer.recognize(audio_flienames[0]))
    # r2 = RecognizerResult(recognizer.recognize(audio_flienames[1]))

    # print(r1 - r2)

    # MFCC and DTW
    jvs_dataset = JvsDataset(root="/workspace/persistent/japanese project/jvs_ver1")

    out_file = (
        "/workspace/persistent/japanese project/compare_result/result_mfcc_far.txt"
    )

    # same parallel, different speaker
    for p in range(1, 11):

        for s1 in range(1, 11):

            for s2 in range(1, 11):

                get_distance_of_parallel(jvs_dataset, s1, p, s2, p)


def get_distance_of_parallel(
    jvs_dataset, speaker_1, parallel_1, speaker_2, parallel_2, out=None
):

    # print("parallel{:02d} MFCC compare".format(parallel))
    # out.write("parallel{:02d} MFCC compare".format(parallel) + "\n")

    # load teacher audio
    mfcc_1 = mfcc(jvs_dataset, speaker_1, parallel_1)

    mfcc_2 = mfcc(jvs_dataset, speaker_2, parallel_2)

    distance, path = fastdtw.fastdtw(mfcc_1.T, mfcc_2.T, dist=euclidean)

    # TODO: better print
    # TODO: outout to file
    print(
        "parallel:{:02d}.speaker:{:02d} <-> parallel:{:02d}.speaker:{:02d}, distance = {}".format(
            parallel_1, speaker_1, parallel_2, speaker_2, distance
        )
    )


def mfcc_of_parallel(jvs_dataset, speaker_1, parallel_1):
    a0 = jvs_dataset.get_audio_file_path(speaker_1, parallel_id=parallel_1)
    y, sr = librosa.load(a0)
    result = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
    return result


# -------------------------------------------------


def get_distance_of_mfcc(mfcc_1, mfcc_2):

    distance, path = fastdtw.fastdtw(mfcc_1.T, mfcc_2.T, dist=euclidean)
    # print("len of mfcc_1 = {}".format(len(mfcc_1)))
    # print("len of mfcc_2 = {}".format(len(mfcc_2)))

    return distance


def get_mfcc_of_clip(file, start, end):

    audio = AudioSegment.from_file(file)
    fr = audio.frame_rate
    clip = audio[start * 1000 : end * 1000].get_array_of_samples()
    clip = np.array(clip).astype(np.float32)
    mfcc = librosa.feature.mfcc(y=clip, sr=fr, n_mfcc=12)

    return mfcc


def get_mfcc_of_clips(file, starts, ends):
    mfccs = []
    if (
        not isinstance(starts, list)
        or not isinstance(ends, list)
        or len(starts) != len(ends)
    ):
        raise Exception("Parameter error, exccept tow list with save length")

    audio = AudioSegment.from_file(file)
    fr = audio.frame_rate

    for s, e in zip(starts, ends):
        clip = audio[s * 1000 : e * 1000].get_array_of_samples()
        clip = np.array(clip).astype(np.float32)
        mfcc = librosa.feature.mfcc(y=clip, sr=fr, n_mfcc=24)

        mfccs.append(mfcc)

    return mfccs


def get_mfcc_of_result(file, recognize_result):
    mfccs = []

    audio = AudioSegment.from_file(file)
    fr = audio.frame_rate

    for s, e in [(r.start, r.end) for r in recognize_result]:
        clip = audio[s * 1000 : e * 1000].get_array_of_samples()
        clip = np.array(clip).astype(np.float32)
        mfcc = librosa.feature.mfcc(y=clip, sr=fr, n_mfcc=24)

        mfccs.append(mfcc)

    return mfccs


def get_audios_of_result(file, recognize_result):
    audios = []

    audio = AudioSegment.from_file(file)
    fr = audio.frame_rate

    for s, e in [(r.start, r.end) for r in recognize_result]:
        clip = audio[s * 1000 : e * 1000].get_array_of_samples()
        clip = np.array(clip).astype(np.float32)

        audios.append(clip)

    return audios


if __name__ == "__main__":
    main()
