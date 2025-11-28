# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audio processor class for Qwen2-VL."""

from typing import Optional, TypedDict, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import Unpack
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class Qwen2VLAudioProcessorKwargs(TypedDict, total=False):
    r"""
    audio_tokens_per_second (`float`, *optional*, defaults to `50.0`):
        Number of audio tokens per second of audio. Used to calculate the number of tokens for a given audio length.
    sampling_rate (`int`, *optional*, defaults to `16000`):
        Sampling rate of the audio signal in Hz.
    """

    audio_tokens_per_second: float
    sampling_rate: int


class Qwen2VLAudioProcessor:
    r"""
    Constructs a Qwen2-VL audio processor that processes audio signals and computes their token lengths.

    Args:
        audio_tokens_per_second (`float`, *optional*, defaults to `50.0`):
            Number of audio tokens per second of audio. Used to calculate the number of tokens for a given audio length.
        sampling_rate (`int`, *optional*, defaults to `16000`):
            Sampling rate of the audio signal in Hz.
    """

    model_input_names = ["audio_values", "audio_lengths"]

    def __init__(
        self,
        audio_tokens_per_second: float = 50.0,
        sampling_rate: int = 16000,
        **kwargs,
    ) -> None:
        self.audio_tokens_per_second = audio_tokens_per_second
        self.sampling_rate = sampling_rate

    def __call__(
        self,
        audios: Union[list[np.ndarray], np.ndarray],
        audio_tokens_per_second: Optional[float] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs: Unpack[Qwen2VLAudioProcessorKwargs],
    ) -> BatchFeature:
        """
        Process audio signals and compute their token lengths.

        Args:
            audios (`list[np.ndarray]` or `np.ndarray`):
                Audio signals to process. Each audio signal should be a 1D numpy array.
            audio_tokens_per_second (`float`, *optional*, defaults to `self.audio_tokens_per_second`):
                Number of audio tokens per second of audio.
            sampling_rate (`int`, *optional*, defaults to `self.sampling_rate`):
                Sampling rate of the audio signal in Hz.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **audio_values** -- Concatenated audio signals as a numpy array or tensor.
            - **audio_lengths** -- List of audio lengths in tokens for each audio signal.
        """
        audio_tokens_per_second = (
            audio_tokens_per_second if audio_tokens_per_second is not None else self.audio_tokens_per_second
        )
        sampling_rate = sampling_rate if sampling_rate is not None else self.sampling_rate

        # Normalize input to list
        if not isinstance(audios, list):
            audios = [audios]

        # Process each audio signal
        audio_lengths = []
        audio_values = []

        for audio in audios:
            # Ensure audio is numpy array
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)

            # Calculate duration in seconds
            duration_seconds = len(audio) / sampling_rate

            # Calculate number of tokens
            num_tokens = int(duration_seconds * audio_tokens_per_second)
            # Ensure at least 1 token
            num_tokens = max(1, num_tokens)

            audio_lengths.append(num_tokens)
            audio_values.append(audio)

        # Concatenate all audio signals into a single array
        if audio_values:
            audio_values = np.concatenate(audio_values, axis=0)
        else:
            audio_values = np.array([])

        data = {
            "audio_values": audio_values,
            "audio_lengths": audio_lengths,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_audio_tokens(self, audio_length_samples: int, sampling_rate: Optional[int] = None) -> int:
        """
        A utility that returns number of audio tokens for a given audio length.

        Args:
            audio_length_samples (`int`):
                Length of the audio signal in samples.
            sampling_rate (`int`, *optional*, defaults to `self.sampling_rate`):
                Sampling rate of the audio signal in Hz.

        Returns:
            `int`: Number of audio tokens for the given audio length.
        """
        sampling_rate = sampling_rate if sampling_rate is not None else self.sampling_rate
        duration_seconds = audio_length_samples / sampling_rate
        num_tokens = int(duration_seconds * self.audio_tokens_per_second)
        return max(1, num_tokens)


__all__ = ["Qwen2VLAudioProcessor"]

