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
    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]])
    args.is_decoder_only_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom"]])
    if args.is_seq2seq_model:
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


# === Main evaluation loop over 500 prompts ===

def evaluate_watermarking(truncated_texts, model, tokenizer, args):
    """
    For each prompt in truncated_texts:
      - Generate watermarked and non-watermarked completions.
      - Compute detection metrics: green token count (paired only), z–score, p–value, detection judgement.
      - Compute perplexity.
    Finally, among prompts where both outputs have length >=195 tokens, compute average perplexity
    and AUC (using z–scores as continuous detection scores, with label 1 for watermarked and 0 for non-watermarked).
    Returns a dictionary of aggregated metrics and a list of per-prompt results.
    """
    results = []  # store per-prompt metrics
    watermarked_z = []  # list of z–scores for watermarked outputs
    watermarked_z_kgw = []
    nonwatermarked_z = []  # list of z–scores for non-watermarked outputs
    nonwatermarked_z_kgw = []
    true_labels = []  # ground-truth: 1 for watermarked, 0 for non-watermarked
    true_labels_kgw = []
    watermarked_perplexities = []
    watermarked_perplexities_kgw = []
    nonwatermarked_perplexities = []
    nonwatermarked_perplexities_kgw = []

    # Loop over each prompt in the list.
    for prompt in tqdm(truncated_texts, desc="Evaluating prompts"):
        # Generate outputs (this function returns: redecoded_input, truncation_warning, decoded_output_without_watermark, decoded_output_with_watermark, args)
        redecoded_input, truncation_warning, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(
            prompt, args, model=model, device=device, tokenizer=tokenizer
        )

        # Create a watermark processor instance (assume args already contains precomputed pairing and unique tokens)
        wm_processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            seeding_scheme=args.seeding_scheme,
            select_green_tokens=args.select_green_tokens,
            precomputed_pairing=args.precomputed_pairing,
            unique_tokens=args.unique_tokens
        )

        # --- kgw mark ---
        _, _, decoded_output_without_watermark_kgw, decoded_output_with_watermark_kgw, _ = generate_kgw(prompt,
                                                                                                args,
                                                                                                model=model,
                                                                                                device=device,
                                                                                                tokenizer=tokenizer)
        without_watermark_detection_result_kgw = detect_kgw(decoded_output_without_watermark_kgw,
                                                        args,
                                                        device=device,
                                                        tokenizer=tokenizer)
        with_watermark_detection_result_kgw = detect_kgw(decoded_output_with_watermark_kgw,
                                                     args,
                                                     device=device,
                                                     tokenizer=tokenizer)

        # Compute perplexity for both watermarked and non-watermarked outputs:
        ppl_nw_kgw = compute_perplexity(model, tokenizer, decoded_output_without_watermark_kgw, device)
        ppl_wm_kgw = compute_perplexity(model, tokenizer, decoded_output_with_watermark_kgw, device)
        green_count_kgw = with_watermark_detection_result_kgw[1]['num_green_tokens']
        tokens_scored_kgw = with_watermark_detection_result_kgw[1]['num_tokens_scored']
        prop_kgw = with_watermark_detection_result_kgw[1]['green_fraction']
        z_kgw = with_watermark_detection_result_kgw[1]['z_score']
        z_nw_kgw = without_watermark_detection_result_kgw[1]['z_score']
        p_kgw = with_watermark_detection_result_kgw[1]['p_value']
        judgement_kgw = "LLM-generated (watermarked)" if z_kgw > args.detection_z_threshold else "Human-generated (non-watermarked)"

        # --- our method ---
        # Compute detection metrics for watermarked text (only paired tokens are counted).
        green_count_w, tokens_scored_w, prop_w = count_green_tokens_paired(tokenizer, wm_processor,
                                                                           decoded_output_with_watermark)
        z_w, p_w = compute_p_value(green_count_w, tokens_scored_w)
        judgement_w = "LLM-generated (watermarked)" if z_w > args.detection_z_threshold else "Human-generated (non-watermarked)"

        # Compute detection metrics for non-watermarked text.
        green_count_nw, tokens_scored_nw, prop_nw = count_green_tokens_paired(tokenizer, wm_processor,
                                                                              decoded_output_without_watermark)
        z_nw, p_nw = compute_p_value(green_count_nw, tokens_scored_nw)
        judgement_nw = "LLM-generated (watermarked)" if z_nw > args.detection_z_threshold else "Human-generated (non-watermarked)"

        # Compute perplexity for both outputs.
        ppl_wm = compute_perplexity(model, tokenizer, decoded_output_with_watermark, device)
        ppl_nw = compute_perplexity(model, tokenizer, decoded_output_without_watermark, device)

        # Store per-prompt results.
        result = {
            "prompt": redecoded_input,
            "decoded_output_with_watermark": decoded_output_with_watermark,
            "decoded_output_without_watermark": decoded_output_without_watermark,
            "green_count_w": green_count_w,
            "tokens_scored_w": tokens_scored_w,
            "prop_w": prop_w,
            "z_w": z_w,
            "p_w": p_w,
            "judgement_w": judgement_w,
            "ppl_wm": ppl_wm,
            "green_count_nw": green_count_nw,
            "tokens_scored_nw": tokens_scored_nw,
            "prop_nw": prop_nw,
            "z_nw": z_nw,
            "p_nw": p_nw,
            "judgement_nw": judgement_nw,
            "ppl_nw": ppl_nw,
            "decoded_output_with_watermark_kgw": decoded_output_with_watermark_kgw,
            "green_count_kgw": green_count_kgw,
            "tokens_scored_kgw": tokens_scored_kgw,
            "prop_kgw": prop_kgw,
            "z_kgw": z_kgw,
            "p_kgw": p_kgw,
            "judgement_kgw": judgement_kgw,
            "ppl_wm_kgw": ppl_wm_kgw
        }
        results.append(result)

        # Only consider prompts where both generated outputs have length >= 195 tokens.
        len_wm = len(tokenizer(decoded_output_with_watermark)["input_ids"])
        len_nw = len(tokenizer(decoded_output_without_watermark)["input_ids"])
        len_wm_kgw = len(tokenizer(decoded_output_with_watermark_kgw)["input_ids"])
        if len_wm >= 195 and len_nw >= 195:
            watermarked_z.append(z_w)
            nonwatermarked_z.append(z_nw)
            true_labels.append(1)  # watermarked output label is 1
            true_labels.append(0)  # non-watermarked output label is 0
            watermarked_perplexities.append(ppl_wm)
            nonwatermarked_perplexities.append(ppl_nw)

        if len_wm_kgw >= 195 and len_nw >= 195:
            watermarked_z_kgw.append(z_kgw)
            nonwatermarked_z_kgw.append(z_nw_kgw)
            true_labels_kgw.append(1)  # watermarked output label is 1
            true_labels_kgw.append(0)  # non-watermarked output label is 0
            watermarked_perplexities_kgw.append(ppl_wm_kgw)
            nonwatermarked_perplexities_kgw.append(ppl_nw_kgw)

    # Compute average perplexity over valid prompts.
    avg_ppl_wm = np.mean(watermarked_perplexities) if watermarked_perplexities else None
    avg_ppl_wm_kgw = np.mean(watermarked_perplexities_kgw) if watermarked_perplexities_kgw else None
    avg_ppl_nw = np.mean(nonwatermarked_perplexities) if nonwatermarked_perplexities else None

    # Compute AUC using the combined watermarked and non-watermarked z-scores.
    all_scores = watermarked_z + nonwatermarked_z
    all_labels = [1] * len(watermarked_z) + [0] * len(nonwatermarked_z)
    auc = roc_auc_score(all_labels, all_scores) if len(all_scores) > 0 else None

    # Compute AUC using the combined watermarked and non-watermarked z-scores.
    all_scores_kgw = watermarked_z_kgw + nonwatermarked_z_kgw
    all_labels_kgw = [1] * len(watermarked_z_kgw) + [0] * len(nonwatermarked_z_kgw)
    auc_kgw = roc_auc_score(all_labels_kgw, all_scores_kgw) if len(all_scores_kgw) > 0 else None

    aggregated = {
        "avg_ppl_wm": avg_ppl_wm,
        "avg_ppl_wm_kgw": avg_ppl_wm_kgw,
        "avg_ppl_nw": avg_ppl_nw,
        "auc": auc,
        "auc_kgw": auc_kgw,
        "num_valid_prompts": len(watermarked_perplexities),  # number of prompts meeting the length condition
        "num_valid_prompts_kgw": len(watermarked_perplexities_kgw)  # number of prompts meeting the length condition
    }

    return results, aggregated


# === Example usage ===
if __name__ == "__main__":
    args = parse_args()
    model, tokenizer, device = load_model(args)

    # Assume truncated_texts is a list of 500 prompt strings (each truncated to less than 200 words).
    # For example, you may load them from a file or sample from a dataset.
    # Here we assume truncated_texts is already defined.

    # --- Precompute Vocabulary and Matching ---
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

    # --- Load the "realnewslike" subset of C4 (English) and Shuffle the dataset with a fixed seed for reproducibility ---
    c4_realnewslike = load_dataset("c4", "realnewslike", split="train", streaming=False, trust_remote_code=True)
    shuffled_dataset = c4_realnewslike.shuffle(seed=42)
    sampled_examples = shuffled_dataset.select(range(200))
    sampled_texts = [example["text"] for example in sampled_examples]
    print(f"Sampled {len(sampled_texts)} news-like texts from C4.")
    max_words = 150
    truncated_texts = []
    for text in sampled_texts:
        words = text.split()  # split text into words
        truncated_text = " ".join(words[:max_words])
        truncated_texts.append(truncated_text)

    # --- Evaluate Watermarking on all 500 prompts ---
    results, aggregated = evaluate_watermarking(truncated_texts, model, tokenizer, args)

    # --- Print per-prompt results for the first 5 examples ---
    for r in results[:5]:
        print("=== Prompt ===")
        print(r["prompt"])
        print("--- Watermarked Text ---")
        print(r["decoded_output_with_watermark"])
        print("Detection (Watermarked):")
        print(f"  Green tokens (paired only): {r['green_count_w']} / {r['tokens_scored_w']} ({r['prop_w']:.2%})")
        print(f"  z–score: {r['z_w']:.2f}, p–value: {r['p_w']:.4f}")
        print(f"  Judgement: {r['judgement_w']}")
        print(f"  Perplexity: {r['ppl_wm']:.2f}")
        print("--- Watermarked Text KGW ---")
        print(r["decoded_output_with_watermark_kgw"])
        print("Detection (Watermarked KGW):")
        print(f"  Green tokens (paired only): {r['green_count_kgw']} / {r['tokens_scored_kgw']} ({r['prop_kgw']:.2%})")
        print(f"  z–score: {r['z_kgw']:.2f}, p–value: {r['p_kgw']:.4f}")
        print(f"  Judgement: {r['judgement_kgw']}")
        print(f"  Perplexity: {r['ppl_wm_kgw']:.2f}")
        print("--- Non-watermarked Text ---")
        print(r["decoded_output_without_watermark"])
        print("Detection (Non-watermarked):")
        print(f"  Green tokens (paired only): {r['green_count_nw']} / {r['tokens_scored_nw']} ({r['prop_nw']:.2%})")
        print(f"  z–score: {r['z_nw']:.2f}, p–value: {r['p_nw']:.4f}")
        print(f"  Judgement: {r['judgement_nw']}")
        print(f"  Perplexity: {r['ppl_nw']:.2f}")
        print("\n")

    print("=== Aggregated Metrics ===")
    print(f"Average Perplexity (Watermarked): {aggregated['avg_ppl_wm']:.2f}")
    print(f"Average Perplexity (Watermarked KGW): {aggregated['avg_ppl_wm_kgw']:.2f}")
    print(f"Average Perplexity (Non-watermarked): {aggregated['avg_ppl_nw']:.2f}")
    print(f"AUC for Watermark Detection: {aggregated['auc']:.3f}")
    print(f"AUC for Watermark Detection (KGW): {aggregated['auc_kgw']:.3f}")
    print(f"Number of valid prompts (>=195 tokens in both outputs): {aggregated['num_valid_prompts']}")
