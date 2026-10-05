from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Sequence

import numpy as np

from overprompting_exp.models.base import ModelAdapter, ScoredDistribution


@dataclass(frozen=True)
class HFModelConfig:
    model_name: str
    dtype: str = "bfloat16"
    device: str = "auto"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    add_generation_prompt: bool = True
    batch_size: int = 4
    attn_implementation: str | None = None


class HFModelAdapter(ModelAdapter):
    def __init__(self, cfg: HFModelConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError("Hugging Face backend requires extras: pip install -e '.[hf]'") from e

        self._torch = torch
        self._cfg = cfg
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=cfg.trust_remote_code
        )
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else None,
            device_map=cfg.device,
            attn_implementation=cfg.attn_implementation,
        )
        self._model.eval()
        emb = self._model.get_input_embeddings()
        self._input_device = emb.weight.device if emb is not None else torch.device("cpu")
        self._supports_logits_to_keep = "logits_to_keep" in inspect.signature(self._model.forward).parameters
        self._supports_position_ids = "position_ids" in inspect.signature(self._model.forward).parameters

    def format_prompt(self, user_prompt: str, system_prompt: str = "") -> str:
        if not self._cfg.use_chat_template:
            return super().format_prompt(user_prompt=user_prompt, system_prompt=system_prompt)

        messages = []
        system_prompt = system_prompt.strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=self._cfg.add_generation_prompt
        )

    def score_candidates(self, prompt: str, candidates: Sequence[str]) -> ScoredDistribution:
        return self.score_candidates_many([prompt], candidates)[0]

    def score_candidates_many(self, prompts: Sequence[str], candidates: Sequence[str]) -> list[ScoredDistribution]:
        torch = self._torch
        prompt_tok = self._tokenizer(list(prompts), add_special_tokens=False)["input_ids"]
        prompt_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in prompt_tok]
        cand_tok = self._tokenizer(list(candidates), add_special_tokens=False)["input_ids"]
        cand_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in cand_tok]
        logp = self._score_prompt_batch(prompt_ids_list, cand_ids_list, pad_left_to_len=None)
        return [
            ScoredDistribution(candidates=list(candidates), logp=np.asarray(row, dtype=np.float64))
            for row in logp
        ]

    def score_candidates_padded(
        self,
        prompt: str,
        candidates: Sequence[str],
        *,
        pad_left_to_len: int | None = None,
    ) -> ScoredDistribution:
        return self.score_candidates_many_padded([prompt], candidates, pad_left_to_len=pad_left_to_len)[0]

    def score_candidates_many_padded(
        self,
        prompts: Sequence[str],
        candidates: Sequence[str],
        *,
        pad_left_to_len: int | None = None,
    ) -> list[ScoredDistribution]:
        torch = self._torch
        prompt_tok = self._tokenizer(list(prompts), add_special_tokens=False)["input_ids"]
        prompt_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in prompt_tok]
        cand_tok = self._tokenizer(list(candidates), add_special_tokens=False)["input_ids"]
        cand_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in cand_tok]
        logp = self._score_prompt_batch(prompt_ids_list, cand_ids_list, pad_left_to_len=pad_left_to_len)
        return [
            ScoredDistribution(candidates=list(candidates), logp=np.asarray(row, dtype=np.float64))
            for row in logp
        ]

    def _score_prompt_batch(
        self,
        prompt_ids_list,
        cand_ids_list,
        *,
        pad_left_to_len: int | None,
    ) -> np.ndarray:
        """Score the same candidate set for many prompts in flattened prompt-candidate batches."""
        torch = self._torch

        if not prompt_ids_list:
            return np.zeros((0, len(cand_ids_list)), dtype=np.float64)

        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id; set pad_token or eos_token_id.")

        prompt_infos: list[tuple[int, int, torch.Tensor | None, torch.Tensor]] = []
        for prompt_ids in prompt_ids_list:
            prompt_len = int(prompt_ids.shape[0])
            if prompt_len <= 0:
                raise RuntimeError("Prompt tokenized to empty input_ids; cannot score candidates.")
            target = int(pad_left_to_len) if pad_left_to_len is not None else prompt_len
            pad_left = max(0, target - prompt_len)
            left_pad_ids = (
                torch.full((pad_left,), fill_value=pad_id, dtype=prompt_ids.dtype) if pad_left > 0 else None
            )
            prompt_infos.append((prompt_len, prompt_len + pad_left, left_pad_ids, prompt_ids))

        # Exact fast path for single-token candidates:
        # P(candidate | prompt) is just the next-token probability at the prompt boundary.
        if cand_ids_list and all(int(cids.shape[0]) == 1 for cids in cand_ids_list):
            candidate_token_ids = [int(cids[0]) for cids in cand_ids_list]
            result = np.zeros((len(prompt_ids_list), len(cand_ids_list)), dtype=np.float64)
            order = sorted(range(len(prompt_infos)), key=lambda i: prompt_infos[i][1], reverse=True)
            # For single-token candidates, the exact next-token distribution already gives the
            # desired scores. We intentionally process one prompt at a time here: it is still
            # dramatically faster than re-scoring the full prompt 4x, and it avoids the small
            # batch-dependent BF16 logprob drift we observed when mixing multiple prompts.
            batch_size = 1

            for start in range(0, len(order), batch_size):
                idxs = order[start : start + batch_size]
                seqs = []
                last_positions = []
                for idx in idxs:
                    prompt_len, prompt_len_padded, left_pad_ids, prompt_ids = prompt_infos[idx]
                    if left_pad_ids is not None:
                        seq = torch.cat([left_pad_ids, prompt_ids], dim=0)
                        last_positions.append(prompt_len_padded - 1)
                    else:
                        seq = prompt_ids
                        last_positions.append(prompt_len - 1)
                    seqs.append(seq)

                lengths = [int(seq.shape[0]) for seq in seqs]
                max_len = max(lengths)
                input_ids = torch.full((len(seqs), max_len), fill_value=pad_id, dtype=seqs[0].dtype)
                attn = torch.zeros((len(seqs), max_len), dtype=seqs[0].dtype)
                for row_idx, seq in enumerate(seqs):
                    L = int(seq.shape[0])
                    input_ids[row_idx, :L] = seq
                    if pad_left_to_len is None:
                        attn[row_idx, :L] = 1
                    else:
                        prompt_len, prompt_len_padded, _left_pad_ids, _prompt_ids = prompt_infos[idxs[row_idx]]
                        pad_left = max(0, prompt_len_padded - prompt_len)
                        attn[row_idx, pad_left:L] = 1

                position_ids = None
                if pad_left_to_len is not None and self._supports_position_ids:
                    position_ids = torch.arange(max_len, dtype=seqs[0].dtype).unsqueeze(0).repeat(len(seqs), 1)

                input_ids = input_ids.to(self._input_device)
                attn = attn.to(self._input_device)
                if position_ids is not None:
                    position_ids = position_ids.to(self._input_device)

                with torch.no_grad():
                    forward_kwargs = {"input_ids": input_ids, "attention_mask": attn, "use_cache": False}
                    if position_ids is not None:
                        forward_kwargs["position_ids"] = position_ids
                    if len(set(last_positions)) == 1 and self._supports_logits_to_keep:
                        forward_kwargs["logits_to_keep"] = 1
                    out = self._model(**forward_kwargs)
                    logits = out.logits
                    log_denom = torch.logsumexp(logits, dim=-1)

                    if int(logits.shape[1]) == 1:
                        next_logits = logits[:, 0, :]
                        next_denom = log_denom[:, 0]
                        for row_idx, prompt_idx in enumerate(idxs):
                            for cand_idx, token_id in enumerate(candidate_token_ids):
                                result[prompt_idx, cand_idx] = float(next_logits[row_idx, token_id] - next_denom[row_idx])
                    else:
                        for row_idx, prompt_idx in enumerate(idxs):
                            pos = int(last_positions[row_idx])
                            for cand_idx, token_id in enumerate(candidate_token_ids):
                                result[prompt_idx, cand_idx] = float(logits[row_idx, pos, token_id] - log_denom[row_idx, pos])

            return result

        flat_items: list[tuple[int, int, int, torch.Tensor]] = []
        for prompt_idx, (prompt_len, prompt_len_padded, left_pad_ids, prompt_ids) in enumerate(prompt_infos):
            for cand_idx, cand_ids in enumerate(cand_ids_list):
                if left_pad_ids is not None:
                    seq = torch.cat([left_pad_ids, prompt_ids, cand_ids], dim=0)
                else:
                    seq = torch.cat([prompt_ids, cand_ids], dim=0)
                flat_items.append((prompt_idx, cand_idx, prompt_len_padded, seq))

        # Batch together sequences with similar lengths to reduce padding waste.
        flat_items.sort(key=lambda item: int(item[3].shape[0]), reverse=True)

        result = np.zeros((len(prompt_ids_list), len(cand_ids_list)), dtype=np.float64)
        batch_size = max(int(self._cfg.batch_size), 1)

        for start in range(0, len(flat_items), batch_size):
            chunk = flat_items[start : start + batch_size]
            seqs = [item[3] for item in chunk]
            lengths = [int(seq.shape[0]) for seq in seqs]
            max_len = max(lengths)

            cand_lens = [int(cand_ids_list[cand_idx].shape[0]) for _, cand_idx, _, _ in chunk]
            max_cand_len = max(cand_lens) if cand_lens else 0

            input_ids = torch.full((len(chunk), max_len), fill_value=pad_id, dtype=seqs[0].dtype)
            attn = torch.zeros((len(chunk), max_len), dtype=seqs[0].dtype)
            for row_idx, (_prompt_idx, _cand_idx, prompt_len_padded, seq) in enumerate(chunk):
                L = int(seq.shape[0])
                input_ids[row_idx, :L] = seq
                if pad_left_to_len is None:
                    attn[row_idx, :L] = 1
                else:
                    pad_left = max(0, prompt_len_padded - int(prompt_infos[_prompt_idx][0]))
                    attn[row_idx, pad_left:L] = 1

            position_ids = None
            if pad_left_to_len is not None and self._supports_position_ids:
                position_ids = torch.arange(max_len, dtype=seqs[0].dtype).unsqueeze(0).repeat(len(chunk), 1)

            input_ids = input_ids.to(self._input_device)
            attn = attn.to(self._input_device)
            if position_ids is not None:
                position_ids = position_ids.to(self._input_device)

            with torch.no_grad():
                forward_kwargs = {"input_ids": input_ids, "attention_mask": attn, "use_cache": False}
                if position_ids is not None:
                    forward_kwargs["position_ids"] = position_ids
                if self._supports_logits_to_keep and max_cand_len > 0:
                    forward_kwargs["logits_to_keep"] = max_cand_len + 1
                out = self._model(**forward_kwargs)
                logits = out.logits
                log_denom = torch.logsumexp(logits, dim=-1)
                time_offset = max_len - int(logits.shape[1])

                for row_idx, (prompt_idx, cand_idx, prompt_len_padded, _seq) in enumerate(chunk):
                    cand_ids = cand_ids_list[cand_idx]
                    lp = 0.0
                    for token_offset in range(int(cand_ids.shape[0])):
                        token_id = int(cand_ids[token_offset])
                        pos = prompt_len_padded + token_offset - 1 - time_offset
                        if pos < 0 or pos >= int(logits.shape[1]):
                            continue
                        lp += float(logits[row_idx, pos, token_id] - log_denom[row_idx, pos])
                    result[prompt_idx, cand_idx] = lp

        return result

    def get_output_weight(self):  # type: ignore[no-untyped-def]
        emb = self._model.get_output_embeddings()
        if emb is None or not hasattr(emb, "weight"):
            raise RuntimeError("HF model does not expose output embeddings weight.")
        return emb.weight

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        """Generate text completion for the given prompt."""
        torch = self._torch
        
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self._input_device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()

    def get_query_hidden(self, prompt: str) -> np.ndarray:
        """Return a last-layer representation at the generation boundary.

        This is an intentionally simple and stable choice: we take the final token in the
        *formatted* prompt (i.e., right before the model would generate the answer).
        """
        torch = self._torch
        ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        attn = torch.ones_like(ids)
        ids = ids.to(self._input_device)
        attn = attn.to(self._input_device)
        with torch.no_grad():
            out = self._model(
                input_ids=ids,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        h = out.hidden_states[-1][0, -1, :]
        return h.detach().float().cpu().numpy()

    def prompt_token_count(self, prompt: str) -> int:
        ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return int(len(ids))

    def query_start_token(self, prompt: str, query_prefix: str) -> int | None:
        idx = str(prompt).rfind(str(query_prefix))
        if idx < 0:
            return None
        prefix = str(prompt)[:idx]
        ids = self._tokenizer(prefix, add_special_tokens=False)["input_ids"]
        return int(len(ids))
