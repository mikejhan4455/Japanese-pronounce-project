import glob
import json
import os
import random
import shutil
import sys
import tempfile
import wave
from io import TextIOWrapper
from os import path
from typing import List, Union
from unittest import result

import fastdtw
import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.spatial.distance import euclidean


class Segment:
    def __init__(self, start, end, label):
        self.start = float(start)
        self.end = float(end)
        self.label = label

    def __repr__(self):
        return f"{self.label}\t: [{self.start}, {self.end}] - {self.length():0.4f}"

    def __getitem__(self, index):
        return (self.start, self.end, self.label)

    def length(self):
        return self.end - self.start


class Clips:
    class Clip:
        def __init__(
            self, audio: AudioSegment, segments: [Segment], transcription=None
        ):
            self.audio = audio
            self.segments = segments
            self.transcription = transcription

    def __init__(self, audio_segments, segments, transcriptions=None):

        self.__audio_clips = split_on_silence(
            audio_segments, silence_thresh=-65, min_silence_len=300, keep_silence=0
        )
        self.__segments_clips = self.split_segments(segments)
        self.__transcription_clips = transcriptions

        # transcription is not given and is True
        if transcriptions and isinstance(transcriptions, bool):

            self.__transcription_clips = self.get_transcription(self.__segments_clips)

        if self.__transcription_clips:

            self.clips = [
                self.Clip(audio, segments, transcription)
                for audio, segments, transcription in zip(
                    self.__audio_clips,
                    self.__segments_clips,
                    self.__transcription_clips,
                )
            ]

        else:
            self.clips = [
                self.Clip(audio, segments)
                for audio, segments in zip(self.__audio_clips, self.__segments_clips)
            ]

    def __getitem__(self, item):
        return self.clips[item]

    def split_segments(self, segments, space="pau"):

        segment_clips = []

        start = 0

        # remove first sil
        if segments[0].label == "sil":
            segments.pop(0)

        for idx, seg in enumerate(segments):
            if seg.label in [space, "sil"]:

                segment_clips.append(segments[start:idx])

                start = idx + 1

        # haddle start bias for every clip
        for clip_idx, segment_clip in enumerate(segment_clips):

            bias = segment_clip[0].start

            clip_start = segment_clip[0].start * 1000
            clip_end = segment_clip[-1].end * 1000

            for i, seg in enumerate(segment_clip):

                # handle bias
                seg.start -= bias
                seg.end -= bias

        return segment_clips

    def get_transcription(self, segments_clips):

        trasncriptions = []

        for clip_idx, segment_clip in enumerate(segments_clips):

            trasncription = ""
            for i, seg in enumerate(segment_clip):

                trasncription += seg.label

                if seg.label in ["a", "e", "i", "o", "u", "a:", "e:", "i:", "o:", "u:"]:

                    trasncription += " "
            trasncriptions.append(trasncription.strip())

        return trasncriptions


class JvsDataset:
    """
    Under dev, only available for parallel100 dataset now
    """

    def __init__(self, root="/workspace/jvs_ver1/"):

        # metadata
        self.MAX_SPEAKER_ID = 100
        self.MAX_PARALLEL_ID = 100
        self.__FRAME_RATE = 16000  # for suit to Julius

        # paths
        self.__ROOT_PATH = root
        self.__parallel100_root = path.join(
            self.__ROOT_PATH, "jvs" + "{speaker_id}", "parallel100"
        )
        self.__transcripts_path = path.join(
            self.__parallel100_root, "transcripts_utf8.txt"
        )

        # datas
        ## read transcription
        self.__read_transcription()

    def get_segment_file_path(self, speaker_id, parallel_id) -> str:

        speaker_id = f"{int(speaker_id):0>3d}"
        parallel_id = f"{int(parallel_id):0>3d}"

        segment_flie_path = path.join(
            self.__parallel100_root, "lab", "mon", "VOICEACTRESS100_{parallel_id}.lab"
        )

        return segment_flie_path.format_map(
            {"speaker_id": speaker_id, "parallel_id": parallel_id}
        )

    def get_audio_file_path(self, speaker_id, parallel_id) -> str:

        speaker_id = f"{int(speaker_id):0>3d}"
        parallel_id = f"{int(parallel_id):0>3d}"

        audio_file_path = path.join(
            self.__parallel100_root,
            "wav24kHz16bit",
            "VOICEACTRESS100_{parallel_id}.wav",
        )

        return audio_file_path.format_map(
            {"speaker_id": speaker_id, "parallel_id": parallel_id}
        )

    def get_parallel_folder_path(self, speaker_id) -> str:
        speaker_id = f"{int(speaker_id):0>3d}"

        audio_file_path = path.join(
            self.__parallel100_root, "wav24kHz16bit"
        ).format_map({"speaker_id": speaker_id})

        return audio_file_path

    def get_segments(self, speaker_id: int, parallel_id: int) -> [Segment]:
        """
        TODO: handle FileNotFoundError
        """

        return self.read_segment_from_file(
            self.get_segment_file_path(speaker_id, parallel_id)
        )

    def get_audio_from_file_path(self, file_path) -> AudioSegment:

        return (
            AudioSegment.from_file(file_path)
            .set_channels(1)
            .set_frame_rate(self.__FRAME_RATE)
        )

    def get_audio(self, speaker_id: int, parallel_id: int) -> AudioSegment:

        return (
            AudioSegment.from_file(self.get_audio_file_path(speaker_id, parallel_id))
            .set_channels(1)
            .set_frame_rate(self.__FRAME_RATE)
        )

    def read_segment_from_file(self, segment_file) -> [Segment]:

        with open(segment_file, "r") as f:
            return [Segment(*tuple(line.split())) for line in f]

    def __read_transcription(self) -> None:

        delimiter = ":"
        with open(self.__transcripts_path.format_map({"speaker_id": "001"})) as f:

            # 方便使用， index 0 設置爲 None
            self.__transcript = [line.split(delimiter)[1].strip() for line in f]

    def get_transcription_from_dataset(self, parallel_id) -> str:

        return self.__transcript[int(parallel_id) - 1]

    def get_transcription(self, segments_clips):

        if isinstance(segments_clips[0], list):

            trasncriptions = []
            for clip_idx, segment_clip in enumerate(segments_clips):

                trasncription = ""
                for i, seg in enumerate(segment_clip):

                    trasncription += seg.label

                    if seg.label in [
                        "a",
                        "e",
                        "i",
                        "o",
                        "u",
                        "a:",
                        "e:",
                        "i:",
                        "o:",
                        "u:",
                    ]:

                        trasncription += " "
                trasncriptions.append(trasncription.strip())

            return trasncriptions

        else:
            segment_clip = segments_clips
            trasncription = ""
            # test
            for i, seg in enumerate(segment_clip):

                trasncription += seg.label

                if seg.label in ["a", "e", "i", "o", "u", "a:", "e:", "i:", "o:", "u:"]:
                    trasncription += " "

            return trasncription

    def print_audio_segment(self, audio_segment: AudioSegment) -> None:
        """
        NOTE: Seems it's not really related to the dataset, just as a util
        """

        if isinstance(audio_segment, AudioSegment):

            IPython.display.display(
                IPython.display.Audio(
                    audio_segment.get_array_of_samples(), rate=audio_segment.frame_rate
                )
            )

    def print_audio_from_dataset(self, speaker_id, parallel_id) -> None:

        self.print_audio_segment(self.get_audio(speaker_id, parallel_id))

    def plot_segment(
        self,
        audio: Union[str, AudioSegment],
        segment: Union[str, List[Segment], None] = None,
        title: Union[str, None] = None,
    ) -> None:
        """Summary

        Args:
            audio (Union[str, AudioSegment]): if str: read from file, if AudioSegment: direact shown

            segment (Union[List[Segment], None], optional): if provide, display given Segments, if str: read from file

            title (Union[str, None], optional): if not provide, use audio or None


        Raises:
            TypeError: Description
        """

        # handle plt
        fig = plt.figure(figsize=(18, 5))
        ax1 = fig.add_subplot(1, 1, 1)

        # handle audio
        if isinstance(audio, str):
            audioSegment = AudioSegment.from_file(audio)

        elif isinstance(audio, AudioSegment):
            audioSegment = audio

        fr = audioSegment.frame_rate

        sample = np.array(audioSegment.get_array_of_samples())
        ax1.plot(sample)

        xticks = ax1.get_xticks()
        plt.xticks(xticks, xticks / fr)
        ax1.set_xlabel("time [second]")

        # handle segment

        segments = None

        if isinstance(segment, str):
            segments = self.read_segment_from_file(segment)

        elif isinstance(segment, list):
            for i in segment:
                if not isinstance(i, Segment):
                    raise TypeError("Some item is not Segment object")

            segments = segment

        if segments:
            for i, word in enumerate(segment):
                if not i % 2:
                    x0 = word.start * fr
                    x1 = word.end * fr
                    ax1.axvspan(x0, x1, alpha=0.1, color="red")

                if word.label != "sp":
                    ax1.annotate(word.label, (word.start * fr, 0), weight="bold")

        # handle title
        if not title:
            if isinstance(audio, str):
                title = audio
            else:
                title = "No title"

        ax1.set_title(title)

        plt.show()

    def plot_segment_from_dataset(self, speaker_id, parallel_id) -> None:

        audio = self.get_audio(speaker_id, parallel_id)
        segments = self.get_segments(speaker_id, parallel_id)
        title = "speaker_id: {}, parallel_id: {}".format(speaker_id, parallel_id)

        self.plot_segment(audio, segments, title)

    def get_clips_of_audio_from_database(self, speaker_id, parallel_id) -> Clips:

        return Clips(
            self.get_audio(speaker_id, parallel_id),
            self.get_segments(speaker_id, parallel_id),
            transcriptions=True,
        )
