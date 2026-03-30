"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

import asyncio
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TypeAlias

import tinker
from ttt_discover.tinker_utils.misc_utils import Tokenizer

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None
    maybe_mask: list[float] | None = None  # Optional mask: 1.0 = train, 0.0 = don't train

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs

    @property
    def mask(self) -> list[float]:
        """Return mask, defaulting to all 1.0 if not provided."""
        if self.maybe_mask is None:
            return [1.0] * len(self.tokens)
        return self.maybe_mask


class TokenCompleter:
    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        raise NotImplementedError


def _flatten_text_tokens(model_input: tinker.ModelInput) -> list[int]:
    tokens: list[int] = []
    for chunk in model_input.chunks:
        if not isinstance(chunk, tinker.types.EncodedTextChunk):
            raise ValueError("Local inference backend only supports text-only prompts")
        tokens.extend(chunk.tokens)
    return tokens


class _StopOnTokenSequence:
    def __init__(self, stop_sequences: list[list[int]]):
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self.stop_sequences:
            return False
        sequence = input_ids[0].tolist()
        return any(
            len(stop) <= len(sequence) and sequence[-len(stop):] == stop
            for stop in self.stop_sequences
        )


@dataclass
class LocalHFTokenCompleter(TokenCompleter):
    model_name_or_path: str
    tokenizer: Tokenizer
    max_new_tokens: int = 2048
    temperature: float = 1.0
    device_map: str = "auto"

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self._torch = torch
        load_kwargs = {
            "torch_dtype": "auto",
            "local_files_only": False,
        }

        # For a single Colab GPU, loading the whole model onto one device is
        # more reliable than `device_map="auto"`, which can leave meta tensors
        # around for some model sizes/configurations.
        self._manual_device = None
        if self.device_map == "auto" and torch.cuda.is_available() and torch.cuda.device_count() == 1:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **load_kwargs,
            )
            self._manual_device = torch.device("cuda:0")
            self._model.to(self._manual_device)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=self.device_map,
                **load_kwargs,
            )

        if self._model.generation_config.pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            self._model.generation_config.pad_token_id = pad_token_id

    def _input_device(self):
        if self._manual_device is not None:
            return self._manual_device
        try:
            emb_device = self._model.get_input_embeddings().weight.device
            if emb_device.type != "meta":
                return emb_device
        except Exception:
            pass
        for param in self._model.parameters():
            if param.device.type != "meta":
                return param.device
        raise RuntimeError("Could not determine a real device for local inference model")

    def _encode_stop_sequences(self, stop: StopCondition) -> list[list[int]]:
        encoded: list[list[int]] = []
        for item in stop:
            if isinstance(item, int):
                encoded.append([item])
            else:
                tokens = self.tokenizer.encode(item, add_special_tokens=False)
                if tokens:
                    encoded.append(tokens)
        return encoded

    async def __call__(self, model_input: tinker.ModelInput, stop: StopCondition) -> TokensWithLogprobs:
        from transformers import StoppingCriteriaList

        prompt_tokens = _flatten_text_tokens(model_input)
        prompt = self._torch.tensor([prompt_tokens], device=self._input_device())
        attention_mask = self._torch.ones_like(prompt)
        stop_sequences = self._encode_stop_sequences(stop)
        stopping_criteria = StoppingCriteriaList([_StopOnTokenSequence(stop_sequences)])

        generation_kwargs = {
            "input_ids": prompt,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._model.generation_config.pad_token_id,
            "stopping_criteria": stopping_criteria,
        }
        if self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
        else:
            generation_kwargs["do_sample"] = False

        generated = await asyncio.to_thread(self._model.generate, **generation_kwargs)
        tokens = generated[0][len(prompt_tokens):].tolist()
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=[0.0] * len(tokens))


@dataclass
class GeminiTokenCompleter(TokenCompleter):
    model_name: str
    tokenizer: Tokenizer
    max_new_tokens: int = 2048
    temperature: float = 1.0
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    max_retries: int = 4
    retry_base_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        self._api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini backend requires GEMINI_API_KEY or GOOGLE_API_KEY in the environment"
            )
        self._model_path = self.model_name
        if not self._model_path.startswith("models/"):
            self._model_path = f"models/{self._model_path}"

    def _decode_prompt(self, model_input: tinker.ModelInput) -> str:
        return self.tokenizer.decode(_flatten_text_tokens(model_input))

    def _stop_sequences(self, stop: StopCondition) -> list[str]:
        stop_sequences: list[str] = []
        for item in stop:
            if isinstance(item, int):
                stop_sequences.append(self.tokenizer.decode([item]))
            else:
                stop_sequences.append(item)
        return [seq for seq in stop_sequences if seq]

    def _request_payload(self, prompt_text: str, stop: StopCondition) -> dict:
        generation_config: dict[str, object] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_new_tokens,
        }
        stop_sequences = self._stop_sequences(stop)
        if stop_sequences:
            generation_config["stopSequences"] = stop_sequences
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                }
            ],
            "generationConfig": generation_config,
        }

    def _extract_text(self, response_data: dict) -> str:
        candidates = response_data.get("candidates", [])
        if not candidates:
            prompt_feedback = response_data.get("promptFeedback")
            raise RuntimeError(f"Gemini returned no candidates: {prompt_feedback!r}")
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if part.get("text")]
        if not texts:
            finish_reason = candidates[0].get("finishReason")
            raise RuntimeError(f"Gemini returned no text parts (finish_reason={finish_reason!r})")
        return "".join(texts)

    def _generate_sync(self, prompt_text: str, stop: StopCondition) -> str:
        payload = json.dumps(self._request_payload(prompt_text, stop)).encode("utf-8")
        url = (
            f"{self.api_base}/{self._model_path}:generateContent?"
            f"{urllib.parse.urlencode({'key': self._api_key})}"
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                return self._extract_text(response_data)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                is_retryable = exc.code in {429, 500, 503}
                last_error = RuntimeError(f"Gemini API request failed ({exc.code}): {body}")
                if not is_retryable or attempt == self.max_retries:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"Gemini API request failed: {exc.reason}")
                if attempt == self.max_retries:
                    raise last_error from exc
            time.sleep(self.retry_base_delay_seconds * (2 ** attempt))
        assert last_error is not None
        raise last_error

    async def __call__(self, model_input: tinker.ModelInput, stop: StopCondition) -> TokensWithLogprobs:
        prompt_text = self._decode_prompt(model_input)
        response_text = await asyncio.to_thread(self._generate_sync, prompt_text, stop)
        tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=[0.0] * len(tokens))


@dataclass
class TwoPhaseTokenCompleter(TokenCompleter):
    """
    Two-phase completer for gpt-oss: if Phase 1 exhausts tokens without stop, Phase 2 forces final answer.
    Uses full context window dynamically.
    """
    sampling_client: tinker.SamplingClient
    tokenizer: Tokenizer
    phase1_max_tokens: int  # Phase 1 limit (e.g., 27000)
    temperature: float = 1.0
    context_window: int = 32768
    context_buffer: int = 50

    PHASE2_PREFILL = "\n\n... okay, I am out of thinking tokens. I need to send my final message now."
    # Full marker to transition from analysis to final channel
    GPTOSS_FINAL_MARKER = "<|end|><|start|>assistant<|channel|>final<|message|>"
    # Marker that indicates we're already in the final channel
    GPTOSS_FINAL_CHANNEL_INDICATOR = "<|channel|>final<|message|>"

    def _hit_stop_sequence(self, tokens: list[int], stop: StopCondition) -> bool:
        """Check if the last token(s) match any stop sequence."""
        if not tokens:
            return False
        for s in stop:
            if isinstance(s, int):
                if tokens[-1] == s:
                    return True
            else:
                stop_tokens = self.tokenizer.encode(s, add_special_tokens=False)
                if len(stop_tokens) <= len(tokens) and tokens[-len(stop_tokens):] == stop_tokens:
                    return True
        return False

    def _contains_subsequence(self, tokens: list[int], pattern: str) -> bool:
        """Check if tokens contain the given pattern as a subsequence."""
        pattern_tokens = self.tokenizer.encode(pattern, add_special_tokens=False)
        if len(pattern_tokens) > len(tokens):
            return False
        for i in range(len(tokens) - len(pattern_tokens) + 1):
            if tokens[i:i + len(pattern_tokens)] == pattern_tokens:
                return True
        return False

    async def __call__(self, model_input: tinker.ModelInput, stop: StopCondition) -> TokensWithLogprobs:
        prompt_length = model_input.length
        
        # phase1_max_tokens is the total context budget for phase 1 (prompt + output)
        # This guarantees (context_window - phase1_max_tokens - buffer) tokens for phase 2
        # e.g., context_window = 32768, buffer = 50, prompt_length = 2000, phase1_max_tokens = 25000
        # then, in phase 1, we can generate at most 25000 - 2000 = 23000 tokens
        # in phase 2, we can generate at most 32768 - 2000 - 23000 - 50 = 7718 tokens
        # If prompt_length = 8000, then we can generate at most 25000 - 8000 = 17000 thinking tokens
        phase1_max = self.phase1_max_tokens - prompt_length
        if phase1_max <= 0:
            raise ValueError(f"Prompt length {prompt_length} exceeds phase1_max_tokens {self.phase1_max_tokens}.")
        
        phase1_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(stop=stop, max_tokens=phase1_max, temperature=self.temperature),
        )
        phase1_tokens = phase1_result.sequences[0].tokens
        phase1_logprobs = phase1_result.sequences[0].logprobs
        assert phase1_logprobs is not None

        # Check if we hit stop sequence
        if self._hit_stop_sequence(phase1_tokens, stop) or len(phase1_tokens) < phase1_max:
            return TokensWithLogprobs(tokens=phase1_tokens, maybe_logprobs=phase1_logprobs)

        # Phase 2: Didn't hit stop, force completion
        # Phase 2 budget = context_window - prompt - phase1 - buffer
        
        # Already in final channel? Just continue without prefill
        if self._contains_subsequence(phase1_tokens, self.GPTOSS_FINAL_CHANNEL_INDICATOR):
            new_chunks = list(model_input.chunks) + [tinker.types.EncodedTextChunk(tokens=phase1_tokens)]
            phase2_max = self.context_window - prompt_length - len(phase1_tokens) - self.context_buffer
            if phase2_max <= 0:
                return TokensWithLogprobs(tokens=phase1_tokens, maybe_logprobs=phase1_logprobs)
            phase2_result = await self.sampling_client.sample_async(
                prompt=tinker.ModelInput(chunks=new_chunks), num_samples=1,
                sampling_params=tinker.SamplingParams(stop=stop, max_tokens=phase2_max, temperature=self.temperature),
            )
            phase2_tokens = phase2_result.sequences[0].tokens
            phase2_logprobs = phase2_result.sequences[0].logprobs
            assert phase2_logprobs is not None
            return TokensWithLogprobs(tokens=phase1_tokens + phase2_tokens, maybe_logprobs=phase1_logprobs + phase2_logprobs)

        # Need prefill to transition to final channel
        end_token_seq = self.tokenizer.encode("<|end|>", add_special_tokens=False)
        ends_with_end = len(end_token_seq) <= len(phase1_tokens) and phase1_tokens[-len(end_token_seq):] == end_token_seq
        if ends_with_end:
            prefill_text = self.PHASE2_PREFILL + "<|start|>assistant<|channel|>final<|message|>"
        else:
            prefill_text = self.PHASE2_PREFILL + self.GPTOSS_FINAL_MARKER
        prefill_tokens = self.tokenizer.encode(prefill_text, add_special_tokens=False)

        new_chunks = list(model_input.chunks) + [
            tinker.types.EncodedTextChunk(tokens=phase1_tokens),
            tinker.types.EncodedTextChunk(tokens=prefill_tokens),
        ]
        phase2_max = self.context_window - prompt_length - len(phase1_tokens) - len(prefill_tokens) - self.context_buffer
        if phase2_max <= 0:
            return TokensWithLogprobs(
                tokens=phase1_tokens + prefill_tokens,
                maybe_logprobs=phase1_logprobs + [0.0] * len(prefill_tokens),
                maybe_mask=[1.0] * len(phase1_tokens) + [0.0] * len(prefill_tokens),
            )

        phase2_result = await self.sampling_client.sample_async(
            prompt=tinker.ModelInput(chunks=new_chunks), num_samples=1,
            sampling_params=tinker.SamplingParams(stop=stop, max_tokens=phase2_max, temperature=self.temperature),
        )
        phase2_tokens = phase2_result.sequences[0].tokens
        phase2_logprobs = phase2_result.sequences[0].logprobs
        assert phase2_logprobs is not None

        return TokensWithLogprobs(
            tokens=phase1_tokens + prefill_tokens + phase2_tokens,
            maybe_logprobs=phase1_logprobs + [0.0] * len(prefill_tokens) + phase2_logprobs,
            maybe_mask=[1.0] * len(phase1_tokens) + [0.0] * len(prefill_tokens) + [1.0] * len(phase2_tokens),
        )
