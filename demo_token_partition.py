# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
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

from __future__ import annotations
import argparse
from functools import partial
from normalizers import normalization_strategy_lookup
import gradio as gr
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
from datasets import load_dataset
import collections
from math import sqrt
import scipy.stats
import torch
from torch import Tensor
import nltk
import ssl
from nltk.util import ngrams
from transformers import LogitsProcessor
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LogitsProcessorList,
)
from watermark_processor import WatermarkLogitsProcessor_kgw, WatermarkDetector_kgw
from demo_watermark import generate_kgw, detect_kgw
from nltk.corpus import wordnet as wn
from functools import lru_cache
from tqdm import tqdm
import itertools
import networkx as nx

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')

#########################
# Helper Functions
#########################

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(
        description="A demo for watermarking using a revised green/red partition based on WordNet synonyms."
    )
    parser.add_argument("--demo_public", type=str2bool, default=False,
                        help="Expose the gradio demo publicly.")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b",
                        help="Identifier for the pretrained model from Hugging Face.")
    parser.add_argument("--prompt_max_length", type=int, default=None,
                        help="Truncation length for the prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--generation_seed", type=int, default=123,
                        help="Seed for generation reproducibility.")
    parser.add_argument("--use_sampling", type=str2bool, default=True,
                        help="Use multinomial sampling for generation.")
    parser.add_argument("--sampling_temp", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--n_beams", type=int, default=1,
                        help="Number of beams for beam search (if not sampling).")
    parser.add_argument("--use_gpu", type=str2bool, default=True,
                        help="Run inference on GPU if available.")
    parser.add_argument("--seeding_scheme", type=str, default="simple_1",
                        help="Seeding scheme for watermarking.")
    parser.add_argument("--gamma", type=float, default=0.25,
                        help="Target fraction of tokens for the green list.")
    parser.add_argument("--delta", type=float, default=2.0,
                        help="Bias to add to green list token logits.")
    parser.add_argument("--normalizers", type=str, default="",
                        help="Comma separated normalizer names for detection.")
    parser.add_argument("--ignore_repeated_bigrams", type=str2bool, default=False,
                        help="Use repeated bigram variant in detection.")
    parser.add_argument("--detection_z_threshold", type=float, default=4.0,
                        help="Z-score threshold for detection.")
    parser.add_argument("--select_green_tokens", type=str2bool, default=True,
                        help="Legacy option for selecting green tokens.")
    parser.add_argument("--skip_model_load", type=str2bool, default=False,
                        help="Skip model loading (for debugging).")
    parser.add_argument("--seed_separately", type=str2bool, default=True,
                        help="Seed separately for each generation call.")
    parser.add_argument("--load_fp16", type=str2bool, default=False,
                        help="Load model in FP16 mode.")
    args = parser.parse_args()
    # Convert normalizers into a list (if provided)
    args.normalizers = args.normalizers.split(",") if args.normalizers else []
    return args


#############################
# Preprocessing: Vocabulary & Matching
#############################

def get_vocabulary(tokenizer) -> list[str]:
    """
    Returns a list of tokens (where index corresponds to token ID)
    extracted from the tokenizer's vocabulary.
    """
    vocab_dict = tokenizer.get_vocab()
    vocab_list = [None] * len(vocab_dict)
    for token, idx in vocab_dict.items():
        vocab_list[idx] = token
    return vocab_list


@lru_cache(maxsize=None)
def get_lemma_set(token: str) -> set:
    """
    Given a token string, first strip any leading BPE marker (e.g. "Ġ")
    and return the set of lowercased lemma names from all its WordNet synsets.
    """
    # Strip off common prefix markers
    word = token.lstrip("Ġ")
    synsets = wn.synsets(word)
    return {lemma.lower() for s in synsets for lemma in s.lemma_names()}


def are_synonyms(token1: str, token2: str) -> bool:
    """
    Determines whether two tokens are synonyms by checking if token1 (after stripping)
    appears in token2's lemma set and vice versa.
    """
    word1 = token1.lstrip("Ġ")
    word2 = token2.lstrip("Ġ")
    lemmas1 = get_lemma_set(word1)
    lemmas2 = get_lemma_set(word2)
    if not lemmas1 or not lemmas2:
        return False
    return (word1.lower() in lemmas2) and (word2.lower() in lemmas1)


def filter_tokens_with_synonyms(vocab_list: list[str]) -> (list[int], list[int]):
    """
    Splits the vocabulary indices into:
      - unique_indices: indices of tokens that have no synonym in the vocabulary.
      - paired_indices: indices of tokens that have at least one synonym.
    Uses a progress bar via tqdm.
    """
    unique_indices = []
    paired_indices = []
    n = len(vocab_list)
    # Precompute lemma sets for each token (stripped of any leading marker)
    lemma_sets = [get_lemma_set(token) for token in vocab_list]
    for i in tqdm(range(n), desc="Filtering vocabulary"):
        token_i = vocab_list[i]
        lemmas_i = lemma_sets[i]
        has_synonym = False
        for j in range(n):
            if i == j:
                continue
            if (token_i.lstrip("Ġ")).lower() in lemma_sets[j] and (vocab_list[j].lstrip("Ġ")).lower() in lemmas_i:
                has_synonym = True
                break
        if has_synonym:
            paired_indices.append(i)
        else:
            unique_indices.append(i)
    return unique_indices, paired_indices


def construct_similarity_matrix(vocab_list: list[str], indices: list[int]) -> list[list[float]]:
    """
    Constructs an m x m similarity matrix for tokens specified by indices.
    Entry [i][j] is 1.0 if the tokens are synonyms, 0 otherwise.
    Uses nested loops with tqdm progress bar.
    """
    m = len(indices)
    C = [[0.0 for _ in range(m)] for _ in range(m)]
    # Precompute lemma sets for tokens in indices
    lemma_dict = {i: get_lemma_set(vocab_list[i].lstrip("Ġ")) for i in indices}
    for a in tqdm(range(m), desc="Constructing similarity matrix (outer loop)"):
        for b in range(a + 1, m):
            token_a = vocab_list[indices[a]].lstrip("Ġ")
            token_b = vocab_list[indices[b]].lstrip("Ġ")
            lemmas_a = lemma_dict[indices[a]]
            lemmas_b = lemma_dict[indices[b]]
            weight = 1.0 if (token_a.lower() in lemmas_b and token_b.lower() in lemmas_a) else 0.0
            C[a][b] = weight
            C[b][a] = weight
    return C


def find_perfect_matching(similarity_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """
    Constructs an undirected graph from the similarity matrix (only edges with weight>0)
    and returns a maximum–weight matching (as a list of index pairs relative to the input list).
    Uses tqdm over the pairs.
    """
    m = len(similarity_matrix)
    G = nx.Graph()
    G.add_nodes_from(range(m))
    pairs = list(itertools.combinations(range(m), 2))
    for i, j in tqdm(pairs, total=len(pairs), desc="Building graph for matching"):
        weight = similarity_matrix[i][j]
        if weight > 0:
            G.add_edge(i, j, weight=weight)
    matching = nx.max_weight_matching(G, maxcardinality=True)
    pairing = [tuple(sorted(pair)) for pair in matching]
    return pairing


#############################
# Revised Watermark Processor Classes
#############################

class WatermarkBase:
    def __init__(
            self,
            vocab: list[int] = None,
            gamma: float = 0.5,
            delta: float = 2.0,
            seeding_scheme: str = "simple_1",
            hash_key: int = 15485863,
            select_green_tokens: bool = True,
            precomputed_pairing: list[tuple[int, int]] = None,
            unique_tokens: list[int] = None,
    ):
        self.vocab = vocab  # list of token IDs (usually 0, ..., n-1)
        self.vocab_size = len(vocab)
        self.gamma = gamma  # target fraction of tokens for the green list
        self.delta = delta  # bias added to green token logits
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        self.pairing = precomputed_pairing  # perfect matching (pairs) for tokens with synonyms
        self.unique_tokens = unique_tokens  # token IDs that have no synonyms

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        """
        Seeds the RNG deterministically using the last token in input_ids.
        For the "simple_1" scheme, the seed is hash_key * (last token id).
        """
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)
        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, "Input must have at least one token."
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Seeding scheme {seeding_scheme} not implemented.")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """
        Returns the green list token IDs.
        If precomputed pairing and unique_tokens are provided, then:
          - All unique tokens (set A) are in the green list.
          - For each pair in the perfect matching, one token is chosen by a coin flip.
        Otherwise, falls back to a random permutation method.
        The final list is optionally truncated to a target size (gamma * vocab_size).
        """
        self._seed_rng(input_ids)
        if self.pairing is None or self.unique_tokens is None:
            # Fallback: use random permutation.
            greenlist_size = int(self.vocab_size * self.gamma)
            vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
            if self.select_green_tokens:
                return vocab_permutation[:greenlist_size].tolist()
            else:
                return vocab_permutation[-greenlist_size:].tolist()
        else:
            greenlist_ids = self.unique_tokens.copy()
            for pair in self.pairing:
                coin_flip = (torch.rand(1, generator=self.rng).item() < 0.5)
                chosen = pair[0] if coin_flip else pair[1]
                greenlist_ids.append(chosen)
            # desired_size = int(self.vocab_size * self.gamma)
            # if len(greenlist_ids) > desired_size:
            #    perm = torch.randperm(len(greenlist_ids), generator=self.rng).tolist()
            #    indices = perm[:desired_size]
            #    greenlist_ids = [greenlist_ids[i] for i in indices]
            return greenlist_ids


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        return green_tokens_mask.bool()

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)
        batched_greenlist_ids = []
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids.append(greenlist_ids)
        green_tokens_mask = self._calc_greenlist_mask(scores, batched_greenlist_ids)
        scores = self._bias_greenlist_logits(scores, green_tokens_mask, self.delta)
        return scores


class WatermarkDetector(WatermarkBase):
    def __init__(
            self,
            *args,
            device: torch.device = None,
            tokenizer=None,
            z_threshold: float = 4.0,
            normalizers: list[str] = ["unicode"],
            ignore_repeated_bigrams: bool = True,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert device, "Device must be provided."
        assert tokenizer, "A tokenizer is required for detection."
        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)
        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Seeding scheme {self.seeding_scheme} not implemented.")
        self.normalizers = [normalization_strategy_lookup(norm) for norm in normalizers]
        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "Repeated bigram variant requires simple_1 seeding."

    def _compute_z_score(self, observed_count, T):
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        return numer / denom

    def _compute_p_value(self, z):
        return scipy.stats.norm.sf(z)

    def _score_sequence(
            self,
            input_ids: Tensor,
            return_num_tokens_scored: bool = True,
            return_num_green_tokens: bool = True,
            return_green_fraction: bool = True,
            return_green_token_mask: bool = False,
            return_z_score: bool = True,
            return_p_value: bool = True,
    ):
        if self.ignore_repeated_bigrams:
            # Repeated bigram variant: T = number of unique bigrams.
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for bigram in freq.keys():
                prefix = torch.tensor([bigram[0]], device=self.device)
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = bigram[1] in greenlist_ids
            green_token_count = sum(bigram_table.values())
        else:
            # Standard variant: T = total tokens (after min_prefix_len)
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError("Not enough tokens to score.")
            green_token_count = 0
            green_token_mask = []
            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)
        # Debug prints:
        print(f"Total tokens scored (T): {num_tokens_scored}")
        print(f"Green token count: {green_token_count}")
        print(f"Green fraction: {green_token_count / num_tokens_scored:.2%}")

        score_dict = {}
        if return_num_tokens_scored:
            score_dict["num_tokens_scored"] = num_tokens_scored
        if return_num_green_tokens:
            score_dict["num_green_tokens"] = green_token_count
        if return_green_fraction:
            score_dict["green_fraction"] = green_token_count / num_tokens_scored
        if return_z_score:
            score_dict["z_score"] = self._compute_z_score(green_token_count, num_tokens_scored)
        if return_p_value:
            z = score_dict.get("z_score", self._compute_z_score(green_token_count, num_tokens_scored))
            score_dict["p_value"] = self._compute_p_value(z)
        if return_green_token_mask:
            score_dict["green_token_mask"] = green_token_mask
        return score_dict

    def detect(
            self,
            text: str = None,
            tokenized_text: list[int] = None,
            return_prediction: bool = True,
            return_scores: bool = True,
            z_threshold: float = None,
            **kwargs,
    ) -> dict:
        assert (text is not None) ^ (tokenized_text is not None), "Provide either raw or tokenized text."
        if return_prediction:
            kwargs["return_p_value"] = True
        for normalizer in self.normalizers:
            text = normalizer(text)
        if tokenized_text is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
                self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            if self.tokenizer is not None and tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        if return_prediction:
            z_threshold = z_threshold if z_threshold is not None else self.z_threshold
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]
        return output_dict


#############################
# Demo Code (Generation & Detection)
#############################

def load_model(args):
    # Set model type attributes on args.
    args.is_seq2seq_model = any(model_type in args.model_name_or_path for model_type in ["t5", "T0"])
    args.is_decoder_only_model = any(model_type in args.model_name_or_path for model_type in ["gpt", "opt", "bloom"])

    if args.is_seq2seq_model:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16,
                                                         device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not args.load_fp16:
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None):
    # print(f"Generating with {args}")
    # Set model type attributes on args.
    args.is_seq2seq_model = any(model_type in args.model_name_or_path for model_type in ["t5", "T0"])
    args.is_decoder_only_model = any(model_type in args.model_name_or_path for model_type in ["gpt", "opt", "bloom"])

    # Instantiate the watermark processor with precomputed pairing/unique tokens.
    # Assume that in main() we precomputed these values (see below).
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
        precomputed_pairing=args.precomputed_pairing,
        unique_tokens=args.unique_tokens
    )

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True,
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        **gen_kwargs
    )

    if not args.prompt_max_length:
        if hasattr(model.config, "max_position_embedding"):
            args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
        else:
            args.prompt_max_length = 2048 - args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=args.prompt_max_length).to(device)
    truncation_warning = tokd_input["input_ids"].shape[-1] == args.prompt_max_length
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)
    if args.seed_separately:
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        output_without_watermark = output_without_watermark[:, tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:, tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (
        redecoded_input, int(truncation_warning), decoded_output_without_watermark, decoded_output_with_watermark, args)


def format_names(s):
    s = s.replace("num_tokens_scored", "Tokens Counted (T)")
    s = s.replace("num_green_tokens", "# Tokens in Greenlist")
    s = s.replace("green_fraction", "Fraction of T in Greenlist")
    s = s.replace("z_score", "z-score")
    s = s.replace("p_value", "p value")
    s = s.replace("prediction", "Prediction")
    s = s.replace("confidence", "Confidence")
    return s


def list_format_scores(score_dict, detection_threshold):
    lst_2d = []
    for k, v in score_dict.items():
        if k == 'green_fraction':
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k == 'confidence':
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float):
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else:
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2, ["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1, ["z-score Threshold", f"{detection_threshold}"])
    return lst_2d


def detect(input_text, args, device=None, tokenizer=None):
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        normalizers=args.normalizers,
        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
        select_green_tokens=args.select_green_tokens
    )
    if len(input_text) - 1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        output = [["Error", "string too short to compute metrics"]]
        output += [["", ""] for _ in range(6)]
    return output, args


def run_gradio(args, model=None, device=None, tokenizer=None):
    generate_partial = partial(generate, model=model, device=device, tokenizer=tokenizer)
    detect_partial = partial(detect, device=device, tokenizer=tokenizer)
    with gr.Blocks() as demo:
        gr.Markdown("Gradio demo not shown in command-line mode.")
        demo.launch()


# === Helper functions for detection metrics ===

def count_green_tokens_paired(tokenizer, watermark_processor, text: str) -> (int, int, float):
    """
    Tokenizes the input text and for each token (after a minimum prefix),
    computes the green list based on the current prefix (using watermark_processor).
    Only paired tokens (i.e. tokens that have a synonym pair) are counted (unique tokens are excluded).

    Returns:
      green_count: number of paired tokens that appear in the green list.
      tokens_scored: number of tokens scored (only tokens that are paired).
      proportion: green_count / tokens_scored.
    """
    # Tokenize without special tokens
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    total_tokens = len(input_ids)
    # Use a minimum prefix length; if not defined in the processor, default to 1.
    start_idx = getattr(watermark_processor, "min_prefix_len", 1)
    green_count = 0
    tokens_scored = 0

    # For each token position starting at start_idx, consider only if the token is a paired token.
    for idx in range(start_idx, total_tokens):
        token = input_ids[idx].item()
        # Only count if token is NOT in the unique set (i.e. token is paired)
        if watermark_processor.unique_tokens is not None and token not in watermark_processor.unique_tokens:
            tokens_scored += 1
            prefix = input_ids[:idx]
            greenlist_ids = watermark_processor._get_greenlist_ids(prefix)
            if token in greenlist_ids:
                green_count += 1

    proportion = green_count / tokens_scored if tokens_scored > 0 else 0.0
    return green_count, tokens_scored, proportion


def compute_p_value(green_count: int, tokens_scored: int) -> (float, float):
    """
    Given the number of paired tokens that appear in the green list (green_count)
    out of tokens_scored (only paired tokens are scored), compute a z–score and p–value.
    Under the null hypothesis, each paired token is green with probability 0.5.
    """
    import math
    expected = 0.5 * tokens_scored
    std = math.sqrt(0.25 * tokens_scored)
    z = (green_count - expected) / std if std > 0 else 0.0
    p = scipy.stats.norm.sf(z)
    return z, p


def compute_perplexity(model, tokenizer, text: str, device):
    """
    Computes perplexity of the text using the provided model and tokenizer.
    Here we use the language-modeling loss computed by the model.
    """
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity

# === Cos similarity and greedy matching ===

import torch
import torch.nn.functional as F

def construct_similarity_matrix_cos(vocab_list: list[str], indices: list[int], embedding_matrix: torch.Tensor) -> torch.Tensor:
    """
    Constructs an m x m similarity matrix for tokens specified by indices,
    using cosine similarity between token embeddings.

    Args:
      vocab_list: List of all tokens (strings).
      indices: List of token indices corresponding to the non-unique tokens (set B).
      embedding_matrix: A torch.Tensor of shape (vocab_size, hidden_dim) representing the token embeddings.

    Returns:
      A torch.Tensor of shape (m, m) where each entry [i][j] is the cosine similarity between
      the embeddings of vocab_list[indices[i]] and vocab_list[indices[j]], with the diagonal entries set to 0.
    """
    # Extract embeddings for the selected tokens (non-unique tokens).
    selected_embeddings = embedding_matrix[indices]  # shape: (m, hidden_dim)

    # Normalize the embeddings along the feature dimension.
    norm_embeddings = F.normalize(selected_embeddings, p=2, dim=1)

    # Compute the cosine similarity matrix as the dot product between normalized embeddings.
    sim_matrix = torch.mm(norm_embeddings, norm_embeddings.t())

    # Set the diagonal entries to 0 (to ignore self-similarity).
    sim_matrix.fill_diagonal_(0)

    return sim_matrix

import math
import random

def find_perfect_matching_greedy_random(similarity_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """
    Constructs a greedy matching from the similarity matrix using random sampling.
    In each iteration, a random token i is selected from the unmatched set.
    Then, approximately ceil(log2(n)) tokens (where n is the current number of unmatched tokens)
    are randomly sampled from the remaining tokens, and the token j with the highest similarity
    (i.e. highest value in similarity_matrix[i][j]) is selected as a match.

    The function returns a list of tuples (i, j) (with i < j) representing the matched token indices.
    Note: The similarity matrix should have 0 on its diagonal.
    """
    m = len(similarity_matrix)
    unmatched = list(range(m))
    matching = []

    pbar = tqdm(total=len(unmatched)//2, desc="Greedy random matching")
    while len(unmatched) > 1:
        n = len(unmatched)
        # Set sample size to ceil(log2(n)); ensure at least one candidate.
        sample_size = math.ceil(math.log(n, 2)) if n > 1 else 1
        # Randomly choose one token i from the unmatched set.
        i = random.choice(unmatched)
        # Build a list of candidates (all unmatched tokens except i).
        remaining = [x for x in unmatched if x != i]
        # Adjust sample_size if there are fewer candidates than sample_size.
        sample_size = min(sample_size, len(remaining))
        # Randomly sample sample_size candidates.
        candidates = random.sample(remaining, sample_size)
        # Find the candidate j with maximum similarity with i.
        best_j = candidates[0]
        best_weight = similarity_matrix[i][best_j]
        for j in candidates:
            w = similarity_matrix[i][j]
            if w > best_weight:
                best_weight = w
                best_j = j
        # Add the pair (min(i, best_j), max(i, best_j)) for consistency.
        matching.append((min(i, best_j), max(i, best_j)))
        # Remove both tokens from the unmatched set.
        unmatched.remove(i)
        if best_j in unmatched:
            unmatched.remove(best_j)
        pbar.update(1)
    pbar.close()
    return matching


import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# --- Helper: Check if both outputs meet length condition (≥195 tokens) ---
def valid_length(wm_text, nw_text, tokenizer, min_tokens=195):
    len_wm = len(tokenizer(wm_text)["input_ids"])
    len_nw = len(tokenizer(nw_text)["input_ids"])
    return (len_wm >= min_tokens) and (len_nw >= min_tokens)

# ---- Main Evaluation Function ----

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc  # Instead of roc_auc_score

def evaluate_watermarking(truncated_texts, model, tokenizer, args, args_cos=None):
    """
    For each prompt in truncated_texts, generate completions using three methods:
      1. Default watermark method (our method)
      2. KGW method (via generate_kgw and detect_kgw)
      3. Cosine–similarity based method (via generate with args_cos and detect_cos)

    For each method, compute:
      - Number of tokens generated
      - Detection metrics (only on paired tokens): green token count, tokens scored, green fraction,
        z–score and p–value, judgement (watermarked vs. non-watermarked), perplexity.

    Only prompts where all outputs (for all three methods) have length ≥195 tokens are considered.
    We then compute aggregated metrics (average perplexity) and an ROC–AUC by computing
    the false-positive and true-positive rates (via roc_curve) and passing them to auc().

    Returns:
      results: a list of per–prompt result dictionaries.
      aggregated: a dictionary with aggregated metrics for each method.
    """

    results = []

    # Accumulators for default method:
    wm_z_default_acc, nw_z_default_acc, labels_default = [], [], []
    ppl_wm_default_acc, ppl_nw_default_acc = [], []
    green_counts_w_default_acc, tokens_scored_w_default_acc, props_w_default_acc = [], [], []

    # For KGW method:
    wm_z_kgw_acc, nw_z_kgw_acc, labels_kgw = [], [], []
    ppl_wm_kgw_acc, ppl_nw_kgw_acc = [], []
    green_counts_w_kgw_acc, tokens_scored_w_kgw_acc, props_w_kgw_acc = [], [], []

    # For Cosine-based method:
    wm_z_cos_acc, nw_z_cos_acc, labels_cos = [], [], []
    ppl_wm_cos_acc, ppl_nw_cos_acc = [], []
    green_counts_w_cos_acc, tokens_scored_w_cos_acc, props_w_cos_acc = [], [], []

    for prompt in tqdm(truncated_texts, desc="Evaluating prompts"):
        # --- Default Watermark Generation ---
        redecoded_input, truncation_warning, decoded_nw, decoded_wm, _ = generate(
            prompt, args, model=model, device=device, tokenizer=tokenizer
        )
        wm_processor_default = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            seeding_scheme=args.seeding_scheme,
            select_green_tokens=args.select_green_tokens,
            precomputed_pairing=args.precomputed_pairing,
            unique_tokens=args.unique_tokens
        )

        # Count green tokens (only among paired tokens) for watermarked text:
        green_count_w, tokens_scored_w, prop_w = count_green_tokens_paired(
            tokenizer, wm_processor_default, decoded_wm
        )
        z_default, p_default = compute_p_value(green_count_w, tokens_scored_w)

        # Non-watermarked text:
        green_count_nw, tokens_scored_nw, prop_nw = count_green_tokens_paired(
            tokenizer, wm_processor_default, decoded_nw
        )
        z_nw_default, p_nw_default = compute_p_value(green_count_nw, tokens_scored_nw)

        judgement_default = (
            "LLM-generated (watermarked)" if z_default > args.detection_z_threshold else "Human-generated (non-watermarked)"
        )
        perplexity_wm = compute_perplexity(model, tokenizer, decoded_wm, device)
        perplexity_nw = compute_perplexity(model, tokenizer, decoded_nw, device)
        tokens_generated_default = len(tokenizer(decoded_wm)["input_ids"])

        # --- KGW Method ---
        redecoded_input_kgw, truncation_warning_kgw, decoded_nw_kgw, decoded_wm_kgw, _ = generate_kgw(
            prompt, args, model=model, device=device, tokenizer=tokenizer
        )
        detect_result_w_kgw = detect_kgw(decoded_wm_kgw, args, device=device, tokenizer=tokenizer)[1]
        if len(decoded_wm_kgw) < 195:
            continue
        z_kgw = detect_result_w_kgw["z_score"]
        green_count_w_kgw = detect_result_w_kgw["num_green_tokens"]
        tokens_scored_w_kgw_local = detect_result_w_kgw["num_tokens_scored"]
        prop_w_kgw = detect_result_w_kgw["green_fraction"]

        detect_result_nw_kgw = detect_kgw(decoded_nw_kgw, args, device=device, tokenizer=tokenizer)[1]
        print(decoded_nw_kgw)
        if len(decoded_nw_kgw) < 195:
            continue
        z_nw_kgw = detect_result_nw_kgw["z_score"]
        green_count_nw_kgw = detect_result_nw_kgw["num_green_tokens"]

        judgement_kgw = (
            "LLM-generated (watermarked)" if z_kgw > args.detection_z_threshold else "Human-generated (non-watermarked)"
        )
        perplexity_wm_kgw = compute_perplexity(model, tokenizer, decoded_wm_kgw, device)
        perplexity_nw_kgw = compute_perplexity(model, tokenizer, decoded_nw_kgw, device)
        tokens_generated_kgw = len(tokenizer(decoded_wm_kgw)["input_ids"])

        # --- Cosine-based Method ---
        redecoded_input_cos, truncation_warning_cos, decoded_nw_cos, decoded_wm_cos, _ = generate(
            prompt, args_cos, model=model, device=device, tokenizer=tokenizer
        )
        wm_processor_cos = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args_cos.gamma,
            delta=args_cos.delta,
            seeding_scheme=args_cos.seeding_scheme,
            select_green_tokens=args_cos.select_green_tokens,
            precomputed_pairing=args_cos.precomputed_pairing,
            unique_tokens=args_cos.unique_tokens
        )
        green_count_w_cos, tokens_scored_w_cos, prop_w_cos = count_green_tokens_paired(
            tokenizer, wm_processor_cos, decoded_wm_cos
        )
        z_cos, p_cos = compute_p_value(green_count_w_cos, tokens_scored_w_cos)

        green_count_nw_cos, tokens_scored_nw_cos, prop_nw_cos = count_green_tokens_paired(
            tokenizer, wm_processor_cos, decoded_nw_cos
        )
        z_nw_cos, p_nw_cos = compute_p_value(green_count_nw_cos, tokens_scored_nw_cos)
        judgement_cos = (
            "LLM-generated (watermarked)" if z_cos > args_cos.detection_z_threshold else "Human-generated (non-watermarked)"
        )
        perplexity_wm_cos = compute_perplexity(model, tokenizer, decoded_wm_cos, device)
        perplexity_nw_cos = compute_perplexity(model, tokenizer, decoded_nw_cos, device)
        tokens_generated_cos = len(tokenizer(decoded_wm_cos)["input_ids"])

        # Store per-prompt results
        result = {
            "prompt": redecoded_input,
            "default": {
                "decoded_wm": decoded_wm,
                "decoded_nw": decoded_nw,
                "green_count_w": green_count_w,
                "green_count_nw": green_count_nw,
                "tokens_scored_w": tokens_scored_w,
                "tokens_scored_nw": tokens_scored_nw,
                "prop_w": prop_w,
                "z_w": z_default,
                "z_nw": z_nw_default,
                "p_w": p_default,
                "judgement": judgement_default,
                "ppl_wm": perplexity_wm,
                "ppl_nw": perplexity_nw,
                "tokens_generated": tokens_generated_default
            },
            "kgw": {
                "decoded_wm": decoded_wm_kgw,
                "decoded_nw": decoded_nw_kgw,
                "green_count_w": green_count_w_kgw,
                "tokens_scored_w": tokens_scored_w_kgw_local,
                "prop_w": prop_w_kgw,
                "z_w": z_kgw,
                "p_w": detect_result_w_kgw.get("p_value", None),
                "judgement": judgement_kgw,
                "ppl_wm": perplexity_wm_kgw,
                "ppl_nw": perplexity_nw_kgw,
                "tokens_generated": tokens_generated_kgw
            },
            "cos": {
                "decoded_wm": decoded_wm_cos,
                "decoded_nw": decoded_nw_cos,
                "green_count_w": green_count_w_cos,
                "tokens_scored_w": tokens_scored_w_cos,
                "prop_w": prop_w_cos,
                "z_w": z_cos,
                "p_w": p_cos,
                "judgement": judgement_cos,
                "ppl_wm": perplexity_wm_cos,
                "ppl_nw": perplexity_nw_cos,
                "tokens_generated": tokens_generated_cos
            }
        }
        results.append(result)
        # Optional prompt-level printing:
        print(result)

        # Only consider prompts where all three methods produce outputs of length ≥ 195 tokens.
        if (valid_length(decoded_wm, decoded_nw, tokenizer) and
            valid_length(decoded_wm_kgw, decoded_nw_kgw, tokenizer) and
            valid_length(decoded_wm_cos, decoded_nw_cos, tokenizer)):

            # Accumulate Default method metrics.
            wm_z_default_acc.append(z_default)
            nw_z_default_acc.append(z_nw_default)
            labels_default.extend([1, 0])
            ppl_wm_default_acc.append(perplexity_wm)
            ppl_nw_default_acc.append(perplexity_nw)
            green_counts_w_default_acc.append(green_count_w)
            tokens_scored_w_default_acc.append(tokens_scored_w)

            # KGW method accumulators.
            wm_z_kgw_acc.append(z_kgw)
            nw_z_kgw_acc.append(z_nw_kgw)
            labels_kgw.extend([1, 0])
            ppl_wm_kgw_acc.append(perplexity_wm_kgw)
            ppl_nw_kgw_acc.append(perplexity_nw_kgw)
            green_counts_w_kgw_acc.append(green_count_w_kgw)
            tokens_scored_w_kgw_acc.append(tokens_scored_w_kgw_local)

            # Cosine-based method accumulators.
            wm_z_cos_acc.append(z_cos)
            nw_z_cos_acc.append(z_nw_cos)
            labels_cos.extend([1, 0])
            ppl_wm_cos_acc.append(perplexity_wm_cos)
            ppl_nw_cos_acc.append(perplexity_nw_cos)
            green_counts_w_cos_acc.append(green_count_w_cos)
            tokens_scored_w_cos_acc.append(tokens_scored_w_cos)

            num_valid = len(ppl_wm_default_acc)
            curr_avg_ppl_wm_default = np.mean(ppl_wm_default_acc)
            curr_avg_ppl_nw_default = np.mean(ppl_nw_default_acc)
            curr_avg_ppl_wm_kgw = np.mean(ppl_wm_kgw_acc)
            curr_avg_ppl_nw_kgw = np.mean(ppl_nw_kgw_acc)
            curr_avg_ppl_wm_cos = np.mean(ppl_wm_cos_acc)
            curr_avg_ppl_nw_cos = np.mean(ppl_nw_cos_acc)

            # -- Use roc_curve() and auc() here instead of roc_auc_score() --
            try:
                # For default method:
                all_scores_default = wm_z_default_acc + nw_z_default_acc
                fpr_def, tpr_def, _ = roc_curve(labels_default, all_scores_default)
                curr_auc_default = auc(fpr_def, tpr_def)
            except Exception:
                curr_auc_default = float('nan')

            try:
                # For KGW method:
                all_scores_kgw = wm_z_kgw_acc + nw_z_kgw_acc
                fpr_kgw, tpr_kgw, _ = roc_curve(labels_kgw, all_scores_kgw)
                curr_auc_kgw = auc(fpr_kgw, tpr_kgw)
            except Exception:
                curr_auc_kgw = float('nan')

            try:
                # For Cosine-based method:
                all_scores_cos = wm_z_cos_acc + nw_z_cos_acc
                fpr_cos, tpr_cos, _ = roc_curve(labels_cos, all_scores_cos)
                curr_auc_cos = auc(fpr_cos, tpr_cos)
            except Exception:
                curr_auc_cos = float('nan')

            print(f"\nAfter {num_valid} valid prompts:")
            print(" Default Method:    avg ppl (wm) = {0:.2f}, avg ppl (nw) = {1:.2f}, AUC = {2:.3f}".format(
                curr_avg_ppl_wm_default, curr_avg_ppl_nw_default, curr_auc_default))
            print(" KGW Method:        avg ppl (wm) = {0:.2f}, avg ppl (nw) = {1:.2f}, AUC = {2:.3f}".format(
                curr_avg_ppl_wm_kgw, curr_avg_ppl_nw_kgw, curr_auc_kgw))
            print(" Cosine-based Method: avg ppl (wm) = {0:.2f}, avg ppl (nw) = {1:.2f}, AUC = {2:.3f}\n".format(
                curr_avg_ppl_wm_cos, curr_avg_ppl_nw_cos, curr_auc_cos))

    # Final aggregated metrics
    aggregated = {}

    # Summaries for each method:
    def finalize_metrics(labels, wm_z_list, nw_z_list, ppl_wm_list, ppl_nw_list, method_name):
        out_dict = {}
        if len(ppl_wm_list) > 0:
            out_dict["avg_ppl_wm"] = np.mean(ppl_wm_list)
            out_dict["avg_ppl_nw"] = np.mean(ppl_nw_list)
            out_dict["num_valid"] = len(ppl_wm_list)
        else:
            out_dict["avg_ppl_wm"] = None
            out_dict["avg_ppl_nw"] = None
            out_dict["num_valid"] = 0

        if wm_z_list and nw_z_list:
            all_scores = wm_z_list + nw_z_list
            try:
                fpr, tpr, _ = roc_curve(labels, all_scores)
                out_dict["auc"] = auc(fpr, tpr)
            except Exception:
                out_dict["auc"] = None
        else:
            out_dict["auc"] = None

        aggregated[method_name] = out_dict

    # Default
    finalize_metrics(labels_default,
                     wm_z_default_acc, nw_z_default_acc,
                     ppl_wm_default_acc, ppl_nw_default_acc,
                     "default")

    # KGW
    finalize_metrics(labels_kgw,
                     wm_z_kgw_acc, nw_z_kgw_acc,
                     ppl_wm_kgw_acc, ppl_nw_kgw_acc,
                     "kgw")

    # Cosine
    finalize_metrics(labels_cos,
                     wm_z_cos_acc, nw_z_cos_acc,
                     ppl_wm_cos_acc, ppl_nw_cos_acc,
                     "cos")

    return results, aggregated


# --- Example usage ---
if __name__ == "__main__":
    args = parse_args()
    model, tokenizer, device = load_model(args)

    # Assume truncated_texts is a list of 500 prompt strings (each truncated to less than 200 words).
    # For example, you may load them from a file or sample from a dataset.
    # Here we assume truncated_texts is already defined.

    # --- Precompute Vocabulary and Perfect Matching via Dictionary ---
    tokenizer_for_vocab = AutoTokenizer.from_pretrained(args.model_name_or_path)
    vocab_list = get_vocabulary(tokenizer_for_vocab)
    print(f"Vocabulary size: {len(vocab_list)}")
    unique_indices, paired_indices = filter_tokens_with_synonyms(vocab_list)
    print(f"Unique tokens (set A): {len(unique_indices)}")
    print(f"Tokens with synonyms (set B): {len(paired_indices)}")
    similarity_matrix = construct_similarity_matrix(vocab_list, paired_indices)
    matching = find_perfect_matching(similarity_matrix)
    mapped_pairing = [(paired_indices[i], paired_indices[j]) for (i, j) in matching]
    args.precomputed_pairing = mapped_pairing
    args.unique_tokens = unique_indices

    # --- Precompute Vocabulary and Perfect Matching via Cosine Similarity ---
    args_cos = parse_args()
    embedding_matrix = model.get_input_embeddings().weight  # shape: (vocab_size, hidden_dim)
    similarity_matrix_cos = construct_similarity_matrix_cos(vocab_list, paired_indices, embedding_matrix)
    matching_cos = find_perfect_matching_greedy_random(similarity_matrix_cos)
    mapped_pairing_cos = [(paired_indices[i], paired_indices[j]) for (i, j) in matching_cos]
    args_cos.precomputed_pairing = mapped_pairing_cos
    args_cos.unique_tokens = unique_indices

    # --- Load the "realnewslike" subset of C4 (English) and Shuffle the dataset with a fixed seed for reproducibility ---
    c4_realnewslike = load_dataset("c4", "realnewslike", split="train", streaming=False, trust_remote_code=True)
    shuffled_dataset = c4_realnewslike.shuffle(seed=45)
    sampled_examples = shuffled_dataset.select(range(300))
    sampled_texts = [example["text"] for example in sampled_examples]
    print(f"Sampled {len(sampled_texts)} news-like texts from C4.")
    max_words = 150
    truncated_texts = []
    for text in sampled_texts:
        words = text.split()  # split text into words
        truncated_text = " ".join(words[:max_words])
        truncated_texts.append(truncated_text)

    results, aggregated = evaluate_watermarking(truncated_texts, model, tokenizer, args, args_cos)

    # Print per-prompt results for the first 5 prompts.
    for r in results[:5]:
        print("=== Prompt ===")
        print(r["prompt"])
        print("--- Default Method ---")
        print("Watermarked Text:")
        print(r["default"]["decoded_wm"])
        print("Detection (Default):")
        print(f"  Green tokens (paired): {r['default']['green_count_w']} / {r['default']['tokens_scored_w']} ({r['default']['prop_w']:.2%})")
        print(f"  z–score: {r['default']['z_w']:.2f}, p–value: {r['default']['p_w']:.4f}")
        print(f"  Judgement: {r['default']['judgement']}")
        print(f"  Perplexity: {r['default']['ppl_wm']:.2f}")
        print("--- KGW Method ---")
        print("Watermarked Text:")
        print(r["kgw"]["decoded_wm"])
        print("Detection (KGW):")
        print(f"  Green tokens (paired): {r['kgw']['green_count_w']} / {r['kgw']['tokens_scored_w']} ({r['kgw']['prop_w']:.2%})")
        print(f"  z–score: {r['kgw']['z_w']:.2f}, p–value: {r['kgw']['p_w']:.4f}")
        print(f"  Judgement: {r['kgw']['judgement']}")
        print(f"  Perplexity: {r['kgw']['ppl_wm']:.2f}")
        print("--- Cosine-based Method ---")
        print("Watermarked Text:")
        print(r["cos"]["decoded_wm"])
        print("Detection (Cos):")
        print(f"  Green tokens (paired): {r['cos']['green_count_w']} / {r['cos']['tokens_scored_w']} ({r['cos']['prop_w']:.2%})")
        print(f"  z–score: {r['cos']['z_w']:.2f}, p–value: {r['cos']['p_w']:.4f}")
        print(f"  Judgement: {r['cos']['judgement']}")
        print(f"  Perplexity: {r['cos']['ppl_wm']:.2f}")
        print("\n")

    # Print aggregated metrics.
    print("=== Aggregated Metrics ===")
    print("Default Method:")
    print("  Average Perplexity (Watermarked):", aggregated["default"]["avg_ppl_wm"])
    print("  Average Perplexity (Non-watermarked):", aggregated["default"]["avg_ppl_nw"])
    print("  AUC:", aggregated["default"]["auc"])
    print("  Valid prompts:", aggregated["default"]["num_valid"])

    print("\nKGW Method:")
    print("  Average Perplexity (Watermarked):", aggregated["kgw"]["avg_ppl_wm"])
    print("  Average Perplexity (Non-watermarked):", aggregated["kgw"]["avg_ppl_nw"])
    print("  AUC:", aggregated["kgw"]["auc"])
    print("  Valid prompts:", aggregated["kgw"]["num_valid"])

    print("\nCosine-based Method:")
    print("  Average Perplexity (Watermarked):", aggregated["cos"]["avg_ppl_wm"])
    print("  Average Perplexity (Non-watermarked):", aggregated["cos"]["avg_ppl_nw"])
    print("  AUC:", aggregated["cos"]["auc"])
    print("  Valid prompts:", aggregated["cos"]["num_valid"])
