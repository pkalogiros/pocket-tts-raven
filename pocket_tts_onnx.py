"""
PocketTTS ONNX - bundle-aware ONNX inference for Pocket TTS.
"""

import json
import os
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from safetensors import safe_open

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import scipy.signal

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class PocketTTSOnnx:
    HF_REPO_ID = "kyutai/pocket-tts"
    DEFAULT_LANGUAGE = "english_2026-04"
    VALID_PRECISIONS = ("int8", "fp32")
    TOKENS_PER_SECOND_ESTIMATE = 3.0
    GEN_SECONDS_PADDING = 2.0

    def __init__(
        self,
        models_dir: str = "onnx",
        language: str = DEFAULT_LANGUAGE,
        tokenizer_path: Optional[str] = None,
        precision: str = "int8",
        device: str = "auto",
        temperature: float = 0.7,
        lsd_steps: int = 1,
    ):
        if precision not in self.VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {self.VALID_PRECISIONS}, got '{precision}'")

        self.models_root = Path(models_dir)
        self.language = self._normalize_language(language)
        self.bundle_dir = self._resolve_bundle_dir(self.models_root, self.language)
        self.metadata = self._load_metadata(self.bundle_dir)

        self.precision = precision
        self.temperature = temperature
        self.lsd_steps = lsd_steps
        self.providers = self._get_providers(device)

        self.sample_rate = int(self.metadata["sample_rate"])
        self.frame_rate = float(self.metadata["frame_rate"])
        self.samples_per_frame = int(self.metadata["samples_per_frame"])
        self.frame_duration = self.samples_per_frame / self.sample_rate
        self.latent_dim = int(self.metadata["latent_dim"])
        self.conditioning_dim = int(self.metadata["conditioning_dim"])
        self.pad_with_spaces_for_short_inputs = bool(
            self.metadata.get("pad_with_spaces_for_short_inputs", False)
        )
        self.remove_semicolons = bool(self.metadata.get("remove_semicolons", False))
        self.model_recommended_frames_after_eos = self.metadata.get(
            "model_recommended_frames_after_eos"
        )
        self.max_token_per_chunk = int(self.metadata.get("max_token_per_chunk", 50))
        self.insert_bos_before_voice = bool(self.metadata.get("insert_bos_before_voice", False))
        self.predefined_voices = tuple(self.metadata.get("predefined_voices", []))

        tokenizer_file = tokenizer_path or str(self.bundle_dir / self.metadata["tokenizer_file"])
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_file)

        self.bos_before_voice = None
        bos_file = self.metadata.get("bos_before_voice_file")
        if bos_file:
            self.bos_before_voice = np.load(self.bundle_dir / bos_file).astype(np.float32)

        self.flow_state_manifest = self.metadata["flow_lm_state_manifest"]
        self.mimi_state_manifest = self.metadata["mimi_state_manifest"]

        self._load_models()
        self._precompute_flow_buffers()
        self._voice_cache: dict[str, np.ndarray] = {}
        self._voice_state_cache: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _normalize_language(language: str) -> str:
        if language == "english":
            return "english_2026-04"
        return language.replace("_2026_", "_2026-")

    @staticmethod
    def _resolve_bundle_dir(models_root: Path, language: str) -> Path:
        candidate = models_root / language
        if candidate.is_dir():
            return candidate
        if (models_root / "bundle.json").exists():
            return models_root
        raise FileNotFoundError(
            f"Could not find ONNX bundle for '{language}' under {models_root}."
        )

    @staticmethod
    def _load_metadata(bundle_dir: Path) -> dict:
        metadata_path = bundle_dir / "bundle.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing bundle metadata: {metadata_path}")
        return json.loads(metadata_path.read_text())

    def _get_providers(self, device: str) -> list[str]:
        if device == "cpu":
            return ["CPUExecutionProvider"]
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _make_session_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(os.cpu_count() or 4, 4)
        opts.inter_op_num_threads = 1
        return opts

    def _model_file(self, stem: str) -> str:
        if self.precision == "int8":
            quantized = self.bundle_dir / f"{stem}_int8.onnx"
            if quantized.exists():
                return quantized.name
        fp32 = self.bundle_dir / f"{stem}.onnx"
        if fp32.exists():
            return fp32.name
        raise FileNotFoundError(f"Missing ONNX file for {stem} in {self.bundle_dir}")

    def _load_models(self):
        opts = self._make_session_options()

        self.mimi_encoder = ort.InferenceSession(
            str(self.bundle_dir / "mimi_encoder.onnx"), sess_options=opts, providers=self.providers
        )
        self.text_conditioner = ort.InferenceSession(
            str(self.bundle_dir / "text_conditioner.onnx"),
            sess_options=opts,
            providers=self.providers,
        )
        self.flow_lm_main = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("flow_lm_main")),
            sess_options=opts,
            providers=self.providers,
        )
        self.flow_lm_flow = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("flow_lm_flow")),
            sess_options=opts,
            providers=self.providers,
        )
        self.mimi_decoder = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("mimi_decoder")),
            sess_options=opts,
            providers=self.providers,
        )

    def _precompute_flow_buffers(self):
        dt = 1.0 / self.lsd_steps
        self._st_buffers = []
        for j in range(self.lsd_steps):
            s = j / self.lsd_steps
            t = s + dt
            self._st_buffers.append(
                (
                    np.array([[s]], dtype=np.float32),
                    np.array([[t]], dtype=np.float32),
                )
            )

    @staticmethod
    def _numpy_dtype(dtype: str):
        return {
            "float32": np.float32,
            "float16": np.float16,
            "int64": np.int64,
            "bool": np.bool_,
        }[dtype]

    def _make_filled_array(self, shape: list[int], dtype, fill: str) -> np.ndarray:
        if fill == "nan":
            return np.full(shape, np.nan, dtype=dtype)
        if fill == "ones":
            return np.ones(shape, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    def _init_state(self, manifest: list[dict]) -> dict[str, np.ndarray]:
        state = {}
        for entry in manifest:
            dtype = self._numpy_dtype(entry["dtype"])
            state[entry["input_name"]] = self._make_filled_array(
                entry["shape"], dtype=dtype, fill=entry["fill"]
            )
        return state

    @staticmethod
    def _clone_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in state.items()}

    def _update_state_from_outputs(
        self,
        state: dict[str, np.ndarray],
        result: list[np.ndarray],
        manifest: list[dict],
        output_offset: int,
    ):
        for entry in manifest:
            state[entry["input_name"]] = result[output_offset + entry["index"]]

    def _load_audio(self, path: Union[str, Path]) -> np.ndarray:
        path = Path(path)

        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as wav_file:
                sr = wav_file.getframerate()
                raw_data = wav_file.readframes(-1)
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            if not HAS_SOUNDFILE:
                raise ImportError("soundfile required for non-wav voice cloning inputs.")
            audio, sr = sf.read(str(path))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

        if sr != self.sample_rate:
            if not HAS_SCIPY:
                raise ImportError("scipy required for resampling.")
            gcd = np.gcd(int(sr), int(self.sample_rate))
            up = int(self.sample_rate // gcd)
            down = int(sr // gcd)
            audio = scipy.signal.resample_poly(audio, up, down, axis=-1).astype(np.float32)

        return audio.reshape(1, 1, -1)

    def encode_voice(self, audio_path: Union[str, Path]) -> np.ndarray:
        audio = self._load_audio(audio_path)
        embeddings = self.mimi_encoder.run(None, {"audio": audio})[0]
        while embeddings.ndim > 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim < 3:
            embeddings = embeddings[None]
        return embeddings.astype(np.float32, copy=False)

    def _prepare_voice_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        while embeddings.ndim > 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim < 3:
            embeddings = embeddings.reshape(1, -1, embeddings.shape[-1])
        if self.insert_bos_before_voice and self.bos_before_voice is not None:
            embeddings = np.concatenate([self.bos_before_voice, embeddings], axis=1)
        return embeddings

    def _hf_model_state(self, filename: str) -> dict[str, dict[str, np.ndarray]]:
        cached = hf_hub_download(repo_id=self.HF_REPO_ID, filename=filename)
        result: dict[str, dict[str, np.ndarray]] = {}
        with safe_open(cached, framework="np") as handle:
            for key in handle.keys():
                module_name, tensor_key = key.split("/", 1)
                result.setdefault(module_name, {})
                result[module_name][tensor_key] = handle.get_tensor(key)
        return result

    @staticmethod
    def _import_model_state_file(source: Union[str, Path]) -> dict[str, dict[str, np.ndarray]]:
        result: dict[str, dict[str, np.ndarray]] = {}
        with safe_open(str(source), framework="np") as handle:
            for key in handle.keys():
                module_name, tensor_key = key.split("/", 1)
                result.setdefault(module_name, {})
                result[module_name][tensor_key] = handle.get_tensor(key)
        return result

    @staticmethod
    def _derive_step(module_state: dict[str, np.ndarray]) -> np.ndarray:
        if "step" in module_state:
            return np.asarray(module_state["step"], dtype=np.int64).reshape(1)
        if "offset" in module_state and "end_offset" not in module_state:
            return np.asarray(module_state["offset"], dtype=np.int64).reshape(1)
        if "current_end" in module_state:
            return np.array([module_state["current_end"].shape[0]], dtype=np.int64)
        return np.array([0], dtype=np.int64)

    def _adapt_state_tensor(self, source: np.ndarray, entry: dict) -> np.ndarray:
        target_shape = tuple(entry["shape"])
        target_dtype = self._numpy_dtype(entry["dtype"])
        source = np.asarray(source, dtype=target_dtype)

        if source.shape == target_shape:
            return source.copy()

        if source.size == np.prod(target_shape, dtype=np.int64):
            return source.reshape(target_shape).copy()

        target = self._make_filled_array(list(target_shape), target_dtype, entry["fill"])
        if source.ndim != len(target_shape):
            return target

        slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source.shape, target_shape))
        if all(s.start == s.stop for s in slices):
            return target
        target[slices] = source[slices]
        return target

    def _state_from_model_state(
        self, model_state: dict[str, dict[str, np.ndarray]], manifest: list[dict]
    ) -> dict[str, np.ndarray]:
        state = self._init_state(manifest)
        for entry in manifest:
            module_state = model_state.get(entry["module"], {})
            tensor = module_state.get(entry["key"])
            if tensor is None and entry["key"] == "step":
                tensor = self._derive_step(module_state)
            if tensor is None:
                continue
            state[entry["input_name"]] = self._adapt_state_tensor(tensor, entry)
        return state

    def _condition_with_voice_embeddings(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        voice_embeddings = self._prepare_voice_embeddings(embeddings)
        state = self._init_state(self.flow_state_manifest)
        empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        result = self.flow_lm_main.run(
            None,
            {"sequence": empty_seq, "text_embeddings": voice_embeddings, **state},
        )
        self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)
        return state

    def prepare_voice_state(self, voice: Union[str, Path, np.ndarray]) -> dict[str, np.ndarray]:
        if isinstance(voice, np.ndarray):
            return self._condition_with_voice_embeddings(voice)

        voice_str = str(voice)
        if voice_str in self._voice_state_cache:
            return self._clone_state(self._voice_state_cache[voice_str])

        if voice_str in self.predefined_voices:
            filename = f"languages/{self.language}/embeddings/{voice_str}.safetensors"
            model_state = self._hf_model_state(filename)
            state = self._state_from_model_state(model_state, self.flow_state_manifest)
            self._voice_state_cache[voice_str] = self._clone_state(state)
            return state

        voice_path = Path(voice_str)
        if voice_path.exists() and voice_path.suffix == ".safetensors":
            model_state = self._import_model_state_file(voice_path)
            return self._state_from_model_state(model_state, self.flow_state_manifest)

        if voice_str in self._voice_cache:
            embeddings = self._voice_cache[voice_str]
        elif voice_path.exists():
            embeddings = self.encode_voice(voice_path)
            self._voice_cache[voice_str] = embeddings
        else:
            raise ValueError(f"Voice '{voice}' not found.")
        return self._condition_with_voice_embeddings(embeddings)

    def _prepare_text_prompt(self, text: str) -> tuple[str, int]:
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
        if self.remove_semicolons:
            text = text.replace(";", ",")

        number_of_words = len(text.split())
        frames_after_eos_guess = 3 if number_of_words <= 4 else 1

        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text[-1].isalnum():
            text = text + "."
        if self.pad_with_spaces_for_short_inputs and len(text.split()) < 5:
            text = " " * 8 + text
        return text, frames_after_eos_guess

    def _tokenize(self, text: str) -> np.ndarray:
        prepared, _ = self._prepare_text_prompt(text)
        token_ids = self.tokenizer.Encode(prepared)
        return np.array(token_ids, dtype=np.int64).reshape(1, -1)

    @staticmethod
    def _find_boundary_indices(tokens: list[int], boundary_tokens: set[int]) -> list[int]:
        indices = [0]
        previous_was_boundary = False
        for index, token in enumerate(tokens):
            if token in boundary_tokens:
                previous_was_boundary = True
            else:
                if previous_was_boundary:
                    indices.append(index)
                previous_was_boundary = False
        indices.append(len(tokens))
        return indices

    def _segments_from_boundaries(
        self, tokens: list[int], boundary_indices: list[int]
    ) -> list[tuple[int, str]]:
        segments = []
        for i in range(len(boundary_indices) - 1):
            start = boundary_indices[i]
            end = boundary_indices[i + 1]
            text = self.tokenizer.Decode(tokens[start:end])
            segments.append((end - start, text))
        return segments

    def _split_into_best_sentences(self, text: str) -> list[str]:
        prepared, _ = self._prepare_text_prompt(text)
        prepared = prepared.strip()
        tokens = self.tokenizer.Encode(prepared)

        eos_tokens = set(self.tokenizer.Encode(".!...?")[1:])
        boundaries = self._find_boundary_indices(tokens, eos_tokens)
        segments = self._segments_from_boundaries(tokens, boundaries)

        fallback_tokens = set(self.tokenizer.Encode(",;:")[1:])
        refined_segments = []
        for count, segment_text in segments:
            if count <= self.max_token_per_chunk:
                refined_segments.append((count, segment_text))
                continue
            sub_tokens = self.tokenizer.Encode(segment_text.strip())
            sub_boundaries = self._find_boundary_indices(sub_tokens, fallback_tokens)
            sub_segments = self._segments_from_boundaries(sub_tokens, sub_boundaries)
            if len(sub_segments) > 1:
                refined_segments.extend(sub_segments)
            else:
                refined_segments.append((count, segment_text))

        chunks = []
        current_chunk = ""
        current_count = 0
        for count, segment_text in refined_segments:
            if not current_chunk:
                current_chunk = segment_text
                current_count = count
                continue
            if current_count + count > self.max_token_per_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = segment_text
                current_count = count
            else:
                current_chunk += " " + segment_text
                current_count += count
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self.TOKENS_PER_SECOND_ESTIMATE + self.GEN_SECONDS_PADDING
        return int(np.ceil(gen_len_sec * self.frame_rate))

    def _run_flow_lm_chunk(
        self,
        initial_state: dict[str, np.ndarray],
        text_ids: np.ndarray,
        max_frames: Optional[int],
        frames_after_eos: int,
    ) -> Generator[np.ndarray, None, None]:
        state = self._clone_state(initial_state)
        text_embeddings = self.text_conditioner.run(None, {"token_ids": text_ids})[0]
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings[None]

        empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        empty_text = np.zeros((1, 0, self.conditioning_dim), dtype=np.float32)

        result = self.flow_lm_main.run(
            None,
            {"sequence": empty_seq, "text_embeddings": text_embeddings, **state},
        )
        self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)

        curr = np.full((1, 1, self.latent_dim), np.nan, dtype=np.float32)
        eos_step = None
        frame_limit = max_frames or self._estimate_max_gen_len(text_ids.shape[1])
        dt = 1.0 / self.lsd_steps

        for step in range(frame_limit):
            result = self.flow_lm_main.run(
                None,
                {"sequence": curr, "text_embeddings": empty_text, **state},
            )
            conditioning = result[0]
            eos_logit = result[1]
            self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)

            if eos_logit[0][0] > -4.0 and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            if self.temperature > 0:
                std = np.sqrt(self.temperature)
                x = np.random.normal(0.0, std, (1, self.latent_dim)).astype(np.float32)
            else:
                x = np.zeros((1, self.latent_dim), dtype=np.float32)

            for s_arr, t_arr in self._st_buffers:
                flow = self.flow_lm_flow.run(
                    None,
                    {"c": conditioning, "s": s_arr, "t": t_arr, "x": x},
                )[0]
                x = x + flow * dt

            latent = x.reshape(1, 1, self.latent_dim)
            yield latent
            curr = latent

    def generate_latents(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
    ) -> np.ndarray:
        base_state = self.prepare_voice_state(voice)
        latent_chunks = []

        for chunk in self._split_into_best_sentences(text):
            _, guess = self._prepare_text_prompt(chunk)
            effective_frames = (
                frames_after_eos
                if frames_after_eos is not None
                else (self.model_recommended_frames_after_eos or (guess + 2))
            )
            text_ids = self._tokenize(chunk)
            latent_chunks.extend(
                self._run_flow_lm_chunk(base_state, text_ids, max_frames, effective_frames)
            )

        if not latent_chunks:
            return np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        return np.concatenate(latent_chunks, axis=1)

    def decode_latents(self, latents: np.ndarray, chunk_size: int = 15) -> np.ndarray:
        state = self._init_state(self.mimi_state_manifest)
        audio_chunks = []

        for index in range(0, latents.shape[1], chunk_size):
            chunk = latents[:, index : index + chunk_size, :]
            result = self.mimi_decoder.run(None, {"latent": chunk, **state})
            audio_chunks.append(result[0].reshape(-1))
            self._update_state_from_outputs(state, result, self.mimi_state_manifest, output_offset=1)

        if not audio_chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(audio_chunks)

    def _decode_worker(self, latent_queue: queue.Queue, audio_chunks: list, decode_chunk_size: int = 12):
        mimi_state = self._init_state(self.mimi_state_manifest)
        buffered = []
        decoded = 0

        while True:
            item = latent_queue.get()
            if item is None:
                break
            buffered.append(item)

            if len(buffered) - decoded >= decode_chunk_size:
                chunk = np.concatenate(buffered[decoded : decoded + decode_chunk_size], axis=1)
                result = self.mimi_decoder.run(None, {"latent": chunk, **mimi_state})
                audio_chunks.append(result[0].reshape(-1))
                self._update_state_from_outputs(
                    mimi_state, result, self.mimi_state_manifest, output_offset=1
                )
                decoded += decode_chunk_size

        if decoded < len(buffered):
            chunk = np.concatenate(buffered[decoded:], axis=1)
            result = self.mimi_decoder.run(None, {"latent": chunk, **mimi_state})
            audio_chunks.append(result[0].reshape(-1))

    def generate(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
    ) -> np.ndarray:
        base_state = self.prepare_voice_state(voice)
        full_audio = []

        for chunk in self._split_into_best_sentences(text):
            _, guess = self._prepare_text_prompt(chunk)
            effective_frames = (
                frames_after_eos
                if frames_after_eos is not None
                else (self.model_recommended_frames_after_eos or (guess + 2))
            )
            text_ids = self._tokenize(chunk)

            latent_queue: queue.Queue = queue.Queue()
            audio_chunks: list[np.ndarray] = []
            decoder = threading.Thread(
                target=self._decode_worker,
                args=(latent_queue, audio_chunks),
                daemon=True,
            )
            decoder.start()

            for latent in self._run_flow_lm_chunk(base_state, text_ids, max_frames, effective_frames):
                latent_queue.put(latent)
            latent_queue.put(None)
            decoder.join()

            if audio_chunks:
                full_audio.append(np.concatenate(audio_chunks))

        if not full_audio:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(full_audio)

    def stream(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
        first_chunk_frames: int = 2,
        target_buffer_sec: float = 0.2,
        max_chunk_frames: int = 15,
    ) -> Generator[np.ndarray, None, None]:
        base_state = self.prepare_voice_state(voice)

        for chunk_text in self._split_into_best_sentences(text):
            _, guess = self._prepare_text_prompt(chunk_text)
            effective_frames = (
                frames_after_eos
                if frames_after_eos is not None
                else (self.model_recommended_frames_after_eos or (guess + 2))
            )
            text_ids = self._tokenize(chunk_text)

            mimi_state = self._init_state(self.mimi_state_manifest)
            generated_latents = []
            decoded_frames = 0
            playback_start_time = None
            start_time = time.time()

            for latent in self._run_flow_lm_chunk(base_state, text_ids, max_frames, effective_frames):
                generated_latents.append(latent)
                pending = len(generated_latents) - decoded_frames
                chunk_size = 0

                if playback_start_time is None:
                    if pending >= first_chunk_frames:
                        chunk_size = first_chunk_frames
                else:
                    elapsed = time.time() - start_time
                    audio_decoded_sec = decoded_frames * self.frame_duration
                    playback_elapsed = elapsed - playback_start_time
                    buffer_sec = audio_decoded_sec - playback_elapsed

                    if buffer_sec < target_buffer_sec and pending >= 1:
                        chunk_size = min(pending, 3)
                    elif pending >= max_chunk_frames:
                        chunk_size = max_chunk_frames

                if chunk_size > 0:
                    latents_chunk = np.concatenate(
                        generated_latents[decoded_frames : decoded_frames + chunk_size], axis=1
                    )
                    result = self.mimi_decoder.run(None, {"latent": latents_chunk, **mimi_state})
                    self._update_state_from_outputs(
                        mimi_state, result, self.mimi_state_manifest, output_offset=1
                    )
                    decoded_frames += chunk_size
                    if playback_start_time is None:
                        playback_start_time = time.time() - start_time
                    yield result[0].reshape(-1)

            if decoded_frames < len(generated_latents):
                latents_chunk = np.concatenate(generated_latents[decoded_frames:], axis=1)
                result = self.mimi_decoder.run(None, {"latent": latents_chunk, **mimi_state})
                yield result[0].reshape(-1)

    def save_audio(self, audio: np.ndarray, path: Union[str, Path]):
        if not HAS_SOUNDFILE:
            raise ImportError("soundfile required.")
        sf.write(str(path), audio, self.sample_rate)

    @property
    def device(self) -> str:
        if "CUDAExecutionProvider" in self.providers:
            return "cuda"
        return "cpu"

    def __repr__(self) -> str:
        return (
            f"PocketTTSOnnx("
            f"language={self.language!r}, "
            f"device={self.device!r}, "
            f"precision={self.precision!r}, "
            f"temperature={self.temperature}, "
            f"lsd_steps={self.lsd_steps}, "
            f"sample_rate={self.sample_rate})"
        )
