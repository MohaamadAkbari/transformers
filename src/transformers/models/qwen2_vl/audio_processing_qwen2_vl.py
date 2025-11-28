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

from typing import Any, Optional, TypedDict, Union

import numpy as np

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import Unpack
from ...utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch

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


class Qwen2VLAudioProcessor(SequenceFeatureExtractor):
    r"""
    Constructs a Qwen2-VL audio processor that processes audio signals using Whisper-style feature extraction.

    This processor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
    Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to `16000`):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chunks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        dither (`float`, *optional*, defaults to 0.0):
            Adds dithering. In other words, adds a small Gaussian noise to each frame.
        audio_tokens_per_second (`float`, *optional*, defaults to `50.0`):
            Number of audio tokens per second of audio. Used to calculate the number of tokens for a given audio length.
    """

    model_input_names = ["padded_inputs", "audio_lengths"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        dither: float = 0.0,
        audio_tokens_per_second: float = 50.0,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.dither = dither
        self.audio_tokens_per_second = audio_tokens_per_second
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _np_extract_fbank_features(self, waveform_batch: np.ndarray, device: str) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        if device != "cpu":
            raise ValueError(
                f"Got device `{device}` for feature extraction, but feature extraction on CUDA accelerator "
                "devices requires torch, which is not installed. Either set `device='cpu'`, or "
                "install torch according to the official instructions: https://pytorch.org/get-started/locally/"
            )
        log_spec_batch = []
        for waveform in waveform_batch:
            log_spec = spectrogram(
                waveform,
                window_function(self.n_fft, "hann"),
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                dither=self.dither,
                mel_filters=self.mel_filters,
                log_mel="log10",
            )
            log_spec = log_spec[:, :-1]
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            log_spec_batch.append(log_spec)
        log_spec_batch = np.array(log_spec_batch)
        return log_spec_batch

    def _torch_extract_fbank_features(self, waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Compute the log-mel spectrogram of the audio using PyTorch's GPU-accelerated STFT implementation with batching,
        yielding results similar to cpu computing with 1e-5 tolerance.
        """
        waveform = torch.from_numpy(waveform).to(device, torch.float32)
        window = torch.hann_window(self.n_fft, device=device)

        # Note: it would be better to dither the chunked waveform,
        # so overlapping signal does not get the same dithering.
        # But, chunking is happening inside pytorch, so it is here.
        if self.dither != 0.0:
            waveform += self.dither * torch.randn(waveform.shape, dtype=waveform.dtype, device=waveform.device)

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if device != "cpu":
            log_spec = log_spec.detach().cpu()
        return log_spec.numpy()

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: list[np.ndarray], attention_mask: list[np.ndarray], padding_value: float = 0.0
    ) -> list[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values

    def __call__(
        self,
        audios: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        device: Optional[str] = "cpu",
        audio_tokens_per_second: Optional[float] = None,
        **kwargs: Unpack[Qwen2VLAudioProcessorKwargs],
    ) -> BatchFeature:
        """
        Process audio signals and compute their token lengths using Whisper-style feature extraction.

        Args:
            audios (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.
            return_tensors (`str` or `TensorType`, *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask.
            padding (`str`, *optional*, defaults to `"max_length"`):
                Padding strategy to use.
            max_length (`int`, *optional*):
                Maximum length for padding.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audios` input was sampled.
            do_normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to zero-mean unit-variance normalize the input.
            device (`str`, *optional*, defaults to `'cpu'`):
                Specifies the device for computation of the log-mel spectrogram.
            audio_tokens_per_second (`float`, *optional*, defaults to `self.audio_tokens_per_second`):
                Number of audio tokens per second of audio.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **padded_inputs** -- Padded and processed audio features (BatchFeature from pad method).
            - **audio_lengths** -- List of audio lengths in tokens for each audio signal.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this audio processor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `audios` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        audio_tokens_per_second = (
            audio_tokens_per_second if audio_tokens_per_second is not None else self.audio_tokens_per_second
        )

        # Normalize input format (similar to WhisperFeatureExtractor)
        is_batched_numpy = isinstance(audios, np.ndarray) and len(audios.shape) > 1
        if is_batched_numpy and len(audios.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(audios, (list, tuple)) and (isinstance(audios[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in audios]
        elif not is_batched and not isinstance(audios, np.ndarray):
            raw_speech = np.asarray(audios, dtype=np.float32)
        elif isinstance(audios, np.ndarray) and audios.dtype is np.dtype(np.float64):
            raw_speech = audios.astype(np.float32)
        else:
            raw_speech = audios

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        # Convert into correct format for padding (using inherited pad method)
        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask or do_normalize,
        )

        # Calculate audio_lengths from original raw_speech before padding
        audio_lengths = []
        for raw_speech_i in raw_speech:
            # Calculate duration in seconds
            duration_seconds = len(raw_speech_i) / self.sampling_rate
            # Calculate number of tokens
            num_tokens = int(duration_seconds * audio_tokens_per_second)
            # Ensure at least 1 token
            num_tokens = max(1, num_tokens)
            audio_lengths.append(num_tokens)

        # Zero-mean and unit-variance normalization (if requested)
        if do_normalize:
            padded_inputs["input_features"] = self.zero_mean_unit_var_norm(
                padded_inputs["input_features"],
                attention_mask=padded_inputs.get("attention_mask"),
                padding_value=self.padding_value,
            )
            padded_inputs["input_features"] = np.stack(padded_inputs["input_features"], axis=0)

        # Make sure list is in array format
        input_features = padded_inputs.get("input_features").transpose(2, 0, 1)

        # Extract mel-filter bank features
        extract_fbank_features = (
            self._torch_extract_fbank_features if is_torch_available() else self._np_extract_fbank_features
        )
        input_features = extract_fbank_features(input_features[0], device)

        if isinstance(input_features[0], list):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]
        else:
            padded_inputs["input_features"] = input_features

        if return_attention_mask:
            # Rescale from sample to feature dimension
            rescaled_attention_mask = padded_inputs["attention_mask"][:, :: self.hop_length]

            # The STFT computation produces L//hop_length + 1 frames, but we skip the last frame.
            # This means we need to trim the rescaled attention mask to match the actual number of frames.
            if padded_inputs["attention_mask"].shape[1] % self.hop_length != 0:
                rescaled_attention_mask = rescaled_attention_mask[:, :-1]
            padded_inputs["attention_mask"] = rescaled_attention_mask

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        # Return both padded_inputs and audio_lengths
        return BatchFeature(
            data={
                "padded_inputs": padded_inputs,
                "audio_lengths": audio_lengths,
            },
            tensor_type=return_tensors,
        )

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

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this audio processor instance.
        """
        import copy

        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        # Remove non-serializable attributes if any
        if "mel_filters" in output:
            del output["mel_filters"]
        return output


__all__ = ["Qwen2VLAudioProcessor"]
