"""Reproducible translation pipeline for deriving Turkish CORDIS records.

Implements backend abstraction, sentence-level alignment, persistent cache,
automatic quality assurance, and human-in-the-loop review sampling.
Turkish records strictly preserve project ID, split assignment, direct labels,
and hierarchical closures from their English source counterparts.
"""

from __future__ import annotations

import abc
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.manifest import (
    compute_file_sha256,
    create_manifest,
    get_source_config,
    relative_path,
    validate_manifest,
)
from naltra.data.preprocessing import validate_record


def segment_sentences(text: str) -> list[str]:
    """Deterministically segment text into sentences while respecting abbreviations.

    Splits on sentence-final punctuation (. ! ?) followed by whitespace, avoiding
    splits on common technical/scientific abbreviations (e.g., e.g., i.e., vs., Fig., Dr.).
    Also handles unspaced sentence punctuation, bullet items, and list enumerations.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    # Protect common abbreviations by temporarily escaping them
    protected = clean_text
    abbreviations = [
        "e.g.",
        "i.e.",
        "al.",
        "Fig.",
        "et al.",
        "vs.",
        "Dr.",
        "Prof.",
        "approx.",
        "dept.",
        "no.",
        "ca.",
        "c.a.",
        "viz.",
    ]
    sub_map = {}
    for idx, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR_{idx}__"
        if abbr in protected:
            protected = protected.replace(abbr, placeholder)
            sub_map[placeholder] = abbr

    # 1. Normalize unspaced punctuation between lowercase/digit and uppercase (e.g. "5G.As")
    protected = re.sub(r"([a-z0-9])\.([A-Z])", r"\1. \2", protected)

    # 2. Normalize bullet points: ensure space around bullet characters
    bullet_chars = r"[•\u2022\u25cf\u25cb\u25aa\u25b6\u25ba\u2713\u2714]"
    protected = re.sub(rf"(?<=[^\s\.\:\;])\s*({bullet_chars})\s*", r". \1 ", protected)
    protected = re.sub(rf"({bullet_chars})\s*", r"\1 ", protected)

    # 3. Normalize list item markers (e.g., ";2.To" -> "; 2. To")
    protected = re.sub(
        r"([;:])\s*([0-9ivx]+\.|\([0-9ivxa-z]+\)|[0-9ivxa-z]+\))\s*([A-Za-z])",
        r"\1 \2 \3",
        protected,
    )

    # 4. Safe dehyphenation of line-break wraps (e.g., "tech- niques" -> "techniques")
    protected = re.sub(r"\b([a-zA-Z]{3,})-\s+([a-zA-Z]{3,})\b", r"\1\2", protected)

    # 5. Split on sentence boundaries, bullets, and enumerated list items
    raw_sentences = re.split(
        rf"(?<=[.!?])\s+(?=[A-Z0-9\"'\(•\u2022\u25cf\u25cb\u25aa\u25b6\u25ba\-])"
        rf"|(?<=[^\s])\s*(?={bullet_chars})"
        rf"|(?<=[;:])\s+(?=(?:[0-9ivx]+\.|\([0-9ivxa-z]+\)|[0-9ivxa-z]+\))\s+[A-Z])",
        protected,
    )

    sentences: list[str] = []
    for s in raw_sentences:
        restored = s.strip()
        for placeholder, original in sub_map.items():
            restored = restored.replace(placeholder, original)
        if restored and restored != ".":
            sentences.append(restored)

    return sentences if sentences else [clean_text]


def chunk_sentence_if_needed(sentence: str, tokenizer: Any, max_tokens: int = 300) -> list[str]:
    """Deterministically chunk a sentence if it exceeds positional token limit.

    Splits at clause boundaries (semicolons, colons, commas, list markers) while preserving order.
    """
    if not sentence.strip():
        return []
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive.")
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) <= max_tokens:
        return [sentence]

    clauses = re.split(r"(?<=[;,])\s+|(?<=[^\s])\s+(?=\([0-9ivx]+\)\s+[A-Za-z])", sentence)
    chunks: list[str] = []
    curr = ""
    for c in clauses:
        candidate = f"{curr} {c}".strip() if curr else c
        try:
            tok_len = len(tokenizer.tokenize(candidate))
        except Exception:
            tok_len = len(candidate.split())
        if tok_len <= max_tokens:
            curr = candidate
        else:
            if curr:
                chunks.append(curr)
            curr = ""
            for word in c.split():
                if len(tokenizer.tokenize(word)) > max_tokens:
                    raise ValueError("A single translation token span exceeds the model limit.")
                candidate_word = f"{curr} {word}".strip()
                if len(tokenizer.tokenize(candidate_word)) > max_tokens:
                    chunks.append(curr)
                    curr = word
                else:
                    curr = candidate_word
    if curr:
        chunks.append(curr)
    return chunks if chunks else [sentence]


@dataclass(frozen=True, slots=True)
class AlignedSentence:
    """One aligned sentence pair between English source and Turkish translation."""

    index: int
    en: str
    tr: str


@dataclass(frozen=True, slots=True)
class TranslationOutput:
    """Complete translation result for one project text with alignment."""

    text: str
    sentence_pairs: list[AlignedSentence]
    backend: str
    model: str
    source_hash: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sentence_pairs": [asdict(sp) for sp in self.sentence_pairs],
            "backend": self.backend,
            "model": self.model,
            "source_hash": self.source_hash,
            "success": self.success,
            "error": self.error,
        }


class BaseTranslator(abc.ABC):
    """Abstract base class for reproducible translation backends."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Translator backend name."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model or checkpoint identifier."""
        raise NotImplementedError

    @abc.abstractmethod
    def translate_sentence(self, sentence: str) -> str:
        """Translate a single sentence from English to Turkish."""
        raise NotImplementedError

    def translate_sentences(self, sentences: Sequence[str]) -> list[str]:
        """Translate multiple sentences with batching fallback."""
        return [self.translate_sentence(s) for s in sentences]

    def translate_document(self, text: str) -> TranslationOutput:
        """Translate a document preserving sentence alignment."""
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sentences = segment_sentences(text)
        if not sentences:
            return TranslationOutput(
                text="",
                sentence_pairs=[],
                backend=self.name,
                model=self.model_name,
                source_hash=source_hash,
                success=False,
                error="Empty input text",
            )

        try:
            translated_sentences = self.translate_sentences(sentences)
            aligned: list[AlignedSentence] = [
                AlignedSentence(index=idx, en=s, tr=tr_s)
                for idx, (s, tr_s) in enumerate(zip(sentences, translated_sentences, strict=False))
            ]
            full_tr_text = " ".join(translated_sentences).strip()
            return TranslationOutput(
                text=full_tr_text,
                sentence_pairs=aligned,
                backend=self.name,
                model=self.model_name,
                source_hash=source_hash,
                success=True,
            )
        except Exception as exc:
            return TranslationOutput(
                text="",
                sentence_pairs=[],
                backend=self.name,
                model=self.model_name,
                source_hash=source_hash,
                success=False,
                error=str(exc),
            )

    def translate_documents(self, texts: Sequence[str]) -> list[TranslationOutput]:
        """Translate multiple documents, with per-document fallback."""
        return [self.translate_document(t) for t in texts]


class MockTranslator(BaseTranslator):
    """Deterministic synthetic translator for tests, offline CI, and pipeline verification.

    Applies deterministic vocabulary substitutions and grammatical prefixes, ensuring
    valid Turkish Unicode characters (ç, ğ, ı, ö, ş, ü) are present while preserving sentence count.
    """

    def __init__(self, model_name: str = "mock-en-tr-v1") -> None:
        self._model_name = model_name
        self._term_map = {
            "project": "proje",
            "research": "araştırma",
            "science": "bilim",
            "system": "sistem",
            "technology": "teknoloji",
            "development": "geliştirme",
            "data": "veri",
            "analysis": "analiz",
            "european": "avrupa",
            "energy": "enerji",
            "health": "sağlık",
            "environment": "çevre",
            "network": "ağ",
            "model": "model",
            "objective": "amaç",
            "method": "yöntem",
            "results": "sonuçlar",
            "new": "yeni",
            "clinical": "klinik",
            "artificial": "yapay",
            "intelligence": "zeka",
        }

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def translate_sentence(self, sentence: str) -> str:
        words = sentence.strip().split()
        if not words:
            return ""
        translated_words = []
        for w in words:
            clean_w = re.sub(r"[^\w]", "", w.lower())
            tr_w = self._term_map.get(clean_w, w)
            translated_words.append(tr_w)
        body = " ".join(translated_words)
        return f"Bu araştırma kapsamında {body} geliştirilmektedir."


class HuggingFaceLocalTranslator(BaseTranslator):
    """Local neural machine translation using HuggingFace MarianMT (RTX 3060 CUDA / CPU).

    Translates English to Turkish using Helsinki-NLP/opus-mt-tc-big-en-tr (revision
    e539fc16a8a1a0ea5950eb339b595bfcce990e90), the official production model.
    A legacy compatibility fallback is maintained for callers requesting
    'Helsinki-NLP/opus-mt-en-tr' to redirect cleanly to the production checkpoint
    without asserting repository equivalence.
    """

    LEGACY_MODEL_FALLBACKS: dict[str, str] = {
        "Helsinki-NLP/opus-mt-en-tr": "Helsinki-NLP/opus-mt-tc-big-en-tr",
        "opus-mt-en-tr": "Helsinki-NLP/opus-mt-tc-big-en-tr",
    }
    # Backward compatibility reference
    MODEL_ALIASES = LEGACY_MODEL_FALLBACKS

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-tc-big-en-tr",
        device: str | None = None,
        batch_size: int = 24,
        max_length: int = 512,
    ) -> None:
        self._requested_model_name = model_name
        self._checkpoint_name = self.LEGACY_MODEL_FALLBACKS.get(model_name, model_name)
        if device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer = None
        self._model = None
        self._revision = get_source_config("translation")["revision"]

    @property
    def name(self) -> str:
        return "huggingface_local"

    @property
    def model_name(self) -> str:
        return self._checkpoint_name

    @property
    def checkpoint_name(self) -> str:
        return self._checkpoint_name

    @property
    def model_revision(self) -> str:
        return self._revision

    def _load_model(self) -> None:
        if self._model is None or self._tokenizer is None:
            import torch
            from transformers import MarianMTModel, MarianTokenizer

            self._tokenizer = MarianTokenizer.from_pretrained(
                self._checkpoint_name, revision=self._revision
            )
            self._model = MarianMTModel.from_pretrained(
                self._checkpoint_name, revision=self._revision
            )
            if self._device == "cuda":
                self._model.to(torch.device("cuda"))
            self._model.eval()

    def translate_sentence(self, sentence: str) -> str:
        res = self.translate_sentences([sentence])
        return res[0] if res else ""

    def translate_sentences(self, sentences: Sequence[str]) -> list[str]:
        if not sentences:
            return []
        self._load_model()
        import torch

        # Check if any sentence needs chunking
        expanded_chunks: list[tuple[int, str]] = []
        for idx, s in enumerate(sentences):
            chunks = chunk_sentence_if_needed(
                s, self._tokenizer, max_tokens=min(500, self._max_length - 4)
            )
            for c in chunks:
                expanded_chunks.append((idx, c))

        translated_chunks: list[str] = []
        batch_size = self._batch_size

        for i in range(0, len(expanded_chunks), batch_size):
            batch = [c[1].strip() for c in expanded_chunks[i : i + batch_size]]
            non_empty_indices = [idx for idx, s in enumerate(batch) if s]
            batch_non_empty = [batch[idx] for idx in non_empty_indices]

            if not batch_non_empty:
                translated_chunks.extend(["" for _ in batch])
                continue

            inputs = self._tokenizer(
                batch_non_empty,
                return_tensors="pt",
                padding=True,
                truncation=False,
                max_length=self._max_length,
            )
            if self._device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            try:
                with torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self._max_length,
                        no_repeat_ngram_size=4,
                    )
                decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_exc:
                if "out of memory" in str(oom_exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Fallback to sub-batches of 8 for this memory-intensive batch
                    decoded = []
                    sub_bs = 8
                    for sub_i in range(0, len(batch_non_empty), sub_bs):
                        sub_batch = batch_non_empty[sub_i : sub_i + sub_bs]
                        sub_inputs = self._tokenizer(
                            sub_batch,
                            return_tensors="pt",
                            padding=True,
                            truncation=False,
                            max_length=self._max_length,
                        )
                        if self._device == "cuda":
                            sub_inputs = {k: v.to("cuda") for k, v in sub_inputs.items()}
                        with torch.inference_mode():
                            sub_out = self._model.generate(
                                **sub_inputs,
                                max_new_tokens=self._max_length,
                                no_repeat_ngram_size=4,
                            )
                        decoded.extend(
                            self._tokenizer.batch_decode(sub_out, skip_special_tokens=True)
                        )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

            batch_res = ["" for _ in batch]
            for idx, text in zip(non_empty_indices, decoded, strict=False):
                batch_res[idx] = text.strip()
            translated_chunks.extend(batch_res)

        grouped: dict[int, list[str]] = defaultdict(list)
        for (orig_idx, _), tr_c in zip(expanded_chunks, translated_chunks, strict=False):
            if tr_c:
                grouped[orig_idx].append(tr_c)

        final_sentences: list[str] = []
        for idx in range(len(sentences)):
            final_sentences.append(" ".join(grouped[idx]).strip())

        return final_sentences

    def translate_documents(self, texts: Sequence[str]) -> list[TranslationOutput]:
        """Translate multiple documents with global sentence-level batching."""
        if not texts:
            return []

        doc_outputs: list[TranslationOutput | None] = [None] * len(texts)
        all_sentences: list[str] = []
        doc_sent_indices: list[list[int]] = []
        doc_hashes: list[str] = []

        for d_idx, text in enumerate(texts):
            s_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            doc_hashes.append(s_hash)
            sents = segment_sentences(text)
            if not sents:
                doc_outputs[d_idx] = TranslationOutput(
                    text="",
                    sentence_pairs=[],
                    backend=self.name,
                    model=self.model_name,
                    source_hash=s_hash,
                    success=False,
                    error="Empty input text",
                )
                doc_sent_indices.append([])
            else:
                sent_indices = []
                for s in sents:
                    sent_indices.append(len(all_sentences))
                    all_sentences.append(s)
                doc_sent_indices.append(sent_indices)

        if all_sentences:
            try:
                translated_all_sents = self.translate_sentences(all_sentences)
            except Exception:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Fallback to per-document translation if batch fails
                return [self.translate_document(t) for t in texts]

            for d_idx, text in enumerate(texts):
                if doc_outputs[d_idx] is not None:
                    continue
                sents = segment_sentences(text)
                indices = doc_sent_indices[d_idx]
                tr_sents = [translated_all_sents[i] for i in indices]
                aligned = [
                    AlignedSentence(index=idx, en=s, tr=tr_s)
                    for idx, (s, tr_s) in enumerate(zip(sents, tr_sents, strict=False))
                ]
                full_tr_text = " ".join(tr_sents).strip()
                doc_outputs[d_idx] = TranslationOutput(
                    text=full_tr_text,
                    sentence_pairs=aligned,
                    backend=self.name,
                    model=self.model_name,
                    source_hash=doc_hashes[d_idx],
                    success=True,
                )

        return [o for o in doc_outputs if o is not None]


class TranslationCache:
    """Resumable persistent file cache for translations."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/translations/cordis_h2020",
        *,
        namespace: str = "development",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_key(self, project_id: str, source_hash: str, backend: str, model: str) -> str:
        raw_key = f"{self.namespace}:{project_id}:{source_hash}:{backend}:{model}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get_cache_key(self, project_id: str, source_hash: str, backend: str, model: str) -> str:
        return self.compute_key(project_id, source_hash, backend, model)

    def _compute_key(self, project_id: str, source_hash: str, backend: str, model: str) -> str:
        return self.compute_key(project_id, source_hash, backend, model)

    def get(
        self, project_id: str, source_hash: str, backend: str, model: str
    ) -> TranslationOutput | None:
        key = self._compute_key(project_id, source_hash, backend, model)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            sentence_pairs = [
                AlignedSentence(index=sp["index"], en=sp["en"], tr=sp["tr"])
                for sp in data.get("sentence_pairs", [])
            ]
            return TranslationOutput(
                text=data["text"],
                sentence_pairs=sentence_pairs,
                backend=data["backend"],
                model=data["model"],
                source_hash=data["source_hash"],
                success=data["success"],
                error=data.get("error"),
            )
        except Exception:
            return None

    def set(
        self,
        project_id: str,
        output: TranslationOutput,
    ) -> None:
        key = self._compute_key(project_id, output.source_hash, output.backend, output.model)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(output.to_dict(), f, indent=2, ensure_ascii=False)


def check_translation_qa(en_text: str, tr_text: str) -> tuple[bool, list[str]]:
    """Perform automated Quality Assurance checks on translated scientific text."""
    issues: list[str] = []
    if not tr_text or not tr_text.strip():
        issues.append("Empty Turkish translation.")
        return False, issues

    if en_text.strip().lower() == tr_text.strip().lower():
        issues.append("Translation identical to English source.")

    len_ratio = len(tr_text) / max(1, len(en_text))
    if len_ratio < 0.35 or len_ratio > 3.0:
        issues.append(f"Unusual character length ratio: {len_ratio:.2f}")

    # Check for unprintable control characters
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", tr_text):
        issues.append("Translation contains unprintable control characters.")

    return len(issues) == 0, issues


def create_stratified_human_review_sample(
    translated_records: list[dict[str, Any]],
    output_path: str | Path = "data/review/translation_review_sample.csv",
    sample_size: int = 100,
    taxonomy_path: str | Path = "taxonomy/taxonomy.json",
) -> Path:
    """Sample stratified records for human translation review without fabricating results."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load label depths and roots from taxonomy
    depth_map: dict[str, int] = {}
    root_map: dict[str, str] = {}
    tax_p = Path(taxonomy_path)
    if tax_p.exists():
        with open(tax_p, encoding="utf-8") as f:
            tax_doc = json.load(f)
        for lbl in tax_doc.get("labels", []):
            depth_map[lbl["id"]] = lbl.get("depth", 1)
        root_ids = tax_doc.get("root_ids")
        if not root_ids:
            root_ids = [lbl["id"] for lbl in tax_doc.get("labels", []) if lbl.get("parent") is None]
        parent_map = {lbl["id"]: lbl.get("parent") for lbl in tax_doc.get("labels", [])}
        for lbl_id in depth_map:
            curr = lbl_id
            visited = set()
            while curr and curr not in visited:
                visited.add(curr)
                if curr in root_ids:
                    root_map[lbl_id] = curr
                    break
                curr = parent_map.get(curr)

    # Stratify by top-level domain
    domain_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in translated_records:
        domain = "other"
        for lbl in r.get("labels_direct", r.get("labels", [])):
            if lbl in root_map:
                domain = root_map[lbl]
                break
            for root_id in [
                "natural_sciences",
                "engineering_and_technology",
                "medical_and_health_sciences",
                "agricultural_sciences",
                "social_sciences",
                "humanities_and_the_arts",
            ]:
                if root_id in lbl:
                    domain = root_id
                    break
        domain_buckets[domain].append(r)

    sampled: list[dict[str, Any]] = []
    per_bucket = max(1, sample_size // max(1, len(domain_buckets)))
    for _d, recs in sorted(domain_buckets.items()):
        sampled.extend(recs[:per_bucket])

    if len(sampled) < sample_size:
        remaining = [r for r in translated_records if r not in sampled]
        sampled.extend(remaining[: sample_size - len(sampled)])
    sampled = sampled[:sample_size]

    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "project_id",
                "split",
                "top_level_domain",
                "hierarchy_depth",
                "labels_direct",
                "english_text",
                "turkish_text",
                "translation_model",
                "review_status",
                "meaning_preserved",
                "terminology_quality",
                "fluency",
                "notes",
            ]
        )
        for r in sampled:
            d_labels = r.get("labels_direct", [])
            max_depth = max([depth_map.get(lbl, 1) for lbl in d_labels], default=1)
            primary_domain = "other"
            for lbl in d_labels:
                if lbl in root_map:
                    primary_domain = root_map[lbl]
                    break

            writer.writerow(
                [
                    r.get("project_id", r.get("id")),
                    r.get("split", ""),
                    primary_domain,
                    max_depth,
                    ";".join(d_labels),
                    r.get("english_source_text", ""),
                    r.get("text", ""),
                    r.get("translation_model", "Helsinki-NLP/opus-mt-tc-big-en-tr"),
                    "pending",
                    "pending",
                    "pending",
                    "pending",
                    "Pending human expert translation review.",
                ]
            )

    print(f"Generated pending human review sample ({len(sampled)} records) -> {out_file}")
    return out_file


def translate_cordis_dataset(
    en_processed_dir: str | Path = "data/processed/cordis_h2020/en",
    output_dir: str | Path = "data/processed/cordis_h2020/tr",
    cache_dir: str | Path = "data/cache/translations/cordis_h2020",
    taxonomy_path: str | Path = "taxonomy/taxonomy.json",
    translator: BaseTranslator | None = None,
    splits: Sequence[str] = ("train", "validation", "test"),
    max_records: int | None = None,
    review_sample_path: str | Path = "data/review/translation_review_sample.csv",
    review_sample_size: int = 100,
    batch_documents_size: int = 16,
    allow_mock: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Translate English CORDIS records to Turkish by split with caching and sentence alignment."""
    in_path = Path(en_processed_dir)
    out_path = Path(output_dir)
    # The manifest records these relative to itself; refuse a cross-drive layout before writing.
    for path in (in_path, cache_dir, *(in_path / f"{s}.jsonl" for s in splits)):
        relative_path(path, out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    trans = translator or HuggingFaceLocalTranslator()
    if trans.name == "mock" and not allow_mock:
        raise ValueError("Mock translation is for explicit development fixtures only.")
    for split_name in splits:
        if not (in_path / f"{split_name}.jsonl").exists():
            raise FileNotFoundError(f"Missing English translation input: {split_name}.")
    validate_manifest(
        in_path,
        "cordis_h2020_en",
        [f"{s}.jsonl" for s in ("train", "validation", "test")],
        Path(taxonomy_path).parent,
    )
    model_revision = getattr(trans, "model_revision", None)
    namespace = json.dumps(
        {
            "revision": model_revision,
            "code": compute_file_sha256(Path(__file__)),
            "max_length": getattr(trans, "_max_length", None),
        },
        sort_keys=True,
    )
    cache = TranslationCache(cache_dir=cache_dir, namespace=namespace)

    with open(taxonomy_path, encoding="utf-8") as f:
        tax_doc = json.load(f)
    allowed_labels = {lbl["id"] for lbl in tax_doc.get("labels", [])}

    results: dict[str, list[dict[str, Any]]] = {}
    total_translated = 0
    total_cached = 0
    total_new = 0
    total_failures = 0
    total_qa_failures = 0
    all_translated_for_sample: list[dict[str, Any]] = []

    failure_log_file = Path(cache_dir) / "failures.jsonl"

    for split_name in splits:
        split_file = in_path / f"{split_name}.jsonl"
        if not split_file.exists():
            continue

        en_recs = load_jsonl(split_file)
        split_limit = max_records if max_records is not None else len(en_recs)
        en_recs_to_process = en_recs[:split_limit]

        print(
            f"\n--- Translating split '{split_name}' ({len(en_recs_to_process)} records) "
            f"using {trans.name} ({trans.model_name}) ---"
        )

        tr_recs: list[dict[str, Any]] = []
        split_t0 = time.time()
        split_new = 0
        split_cached = 0
        split_failures = 0

        # Step 1: Scan cache first
        uncached_indices: list[int] = []
        cached_outputs: dict[int, TranslationOutput] = {}

        for idx, en_rec in enumerate(en_recs_to_process):
            pid = en_rec.get(
                "project_id", en_rec["id"].split(":")[1] if ":" in en_rec["id"] else en_rec["id"]
            )
            en_text = en_rec["text"]
            src_hash = hashlib.sha256(en_text.encode("utf-8")).hexdigest()

            cached = cache.get(pid, src_hash, trans.name, trans.model_name)
            if (
                cached is not None
                and cached.success
                and cached.source_hash == src_hash
                and cached.backend == trans.name
                and cached.model == trans.model_name
            ):
                cached_outputs[idx] = cached
                total_cached += 1
                split_cached += 1
            else:
                uncached_indices.append(idx)

        print(
            f"Split '{split_name}': {len(cached_outputs)} cached, "
            f"{len(uncached_indices)} to translate"
        )

        # Step 2: Translate uncached in chunks
        chunk_size = batch_documents_size
        for c_start in range(0, len(uncached_indices), chunk_size):
            chunk_rec_indices = uncached_indices[c_start : c_start + chunk_size]
            chunk_recs = [en_recs_to_process[i] for i in chunk_rec_indices]
            chunk_texts = [r["text"] for r in chunk_recs]

            try:
                chunk_outputs = trans.translate_documents(chunk_texts)
            except Exception as exc:
                print(
                    f"Warning: Batch translation exception: {exc}. "
                    "Falling back to single-document translation."
                )
                chunk_outputs = [trans.translate_document(t) for t in chunk_texts]

            for rec_idx, rec, tr_out in zip(
                chunk_rec_indices, chunk_recs, chunk_outputs, strict=True
            ):
                pid = rec.get(
                    "project_id", rec["id"].split(":")[1] if ":" in rec["id"] else rec["id"]
                )
                if tr_out.success:
                    qa_ok, qa_issues = check_translation_qa(rec["text"], tr_out.text)
                    if not qa_ok and not allow_mock:
                        raise ValueError(f"Translation QA failed for {pid}: {qa_issues}")
                    cache.set(pid, tr_out)
                    cached_outputs[rec_idx] = tr_out
                    total_new += 1
                    split_new += 1
                else:
                    total_failures += 1
                    split_failures += 1
                    with open(failure_log_file, "a", encoding="utf-8") as f_err:
                        f_err.write(
                            json.dumps(
                                {
                                    "project_id": pid,
                                    "source_hash": hashlib.sha256(
                                        rec["text"].encode("utf-8")
                                    ).hexdigest(),
                                    "error": tr_out.error,
                                    "split": split_name,
                                }
                            )
                            + "\n"
                        )
                    print(f"Warning: Failed to translate project {pid}: {tr_out.error}")

            processed_so_far = split_cached + split_new + split_failures
            if processed_so_far % 100 < chunk_size or processed_so_far == len(en_recs_to_process):
                elapsed = time.time() - split_t0
                speed = (split_new / elapsed) * 60 if elapsed > 0 else 0
                rem_new = len(uncached_indices) - split_new
                rem_mins = (
                    (rem_new / (split_new / elapsed)) / 60 if split_new > 0 and elapsed > 0 else 0
                )
                print(
                    f"[{split_name} {processed_so_far}/{len(en_recs_to_process)}] "
                    f"{processed_so_far/len(en_recs_to_process)*100:5.1f}% | "
                    f"New: {split_new}, Cached: {split_cached}, Fail: {split_failures} | "
                    f"Speed: {speed:5.1f} new proj/min | ETA: {rem_mins:4.1f}m"
                )

        # Step 3: Build records in canonical order
        for idx, en_rec in enumerate(en_recs_to_process):
            if idx not in cached_outputs:
                continue
            tr_out = cached_outputs[idx]
            pid = en_rec.get(
                "project_id", en_rec["id"].split(":")[1] if ":" in en_rec["id"] else en_rec["id"]
            )
            en_text = en_rec["text"]
            src_hash = hashlib.sha256(en_text.encode("utf-8")).hexdigest()

            # Run automated QA
            qa_ok, qa_issues = check_translation_qa(en_text, tr_out.text)
            if not qa_ok:
                total_qa_failures += 1
                if not allow_mock:
                    raise ValueError(f"Cached translation QA failed for {pid}: {qa_issues}")

            tr_rec = {
                "id": f"cordis:{pid}:tr",
                "project_id": pid,
                "pair_id": en_rec.get("pair_id", f"cordis:{pid}"),
                "source_id": pid,
                "variant_of": en_rec["id"],
                "title": en_rec.get("title", ""),
                "text": tr_out.text,
                "labels_direct": en_rec.get("labels_direct", en_rec["labels"]),
                "labels": en_rec["labels"],
                "language": "tr",
                "source": "cordis_h2020",
                "license": en_rec.get("license", "CC BY 4.0"),
                "split": split_name,
                "taxonomy_version": en_rec.get("taxonomy_version", "0.4.0"),
                "synthetic_language_variant": True,
                "translation_backend": trans.name,
                "translation_model": trans.model_name,
                "translation_model_revision": model_revision,
                "translation_source_hash": src_hash,
                "sentence_alignment": [asdict(sp) for sp in tr_out.sentence_pairs],
                "english_source_text": en_text,
            }

            validate_record(tr_rec, allowed_labels=allowed_labels)
            tr_recs.append(tr_rec)
            all_translated_for_sample.append(tr_rec)
            total_translated += 1

        out_file = out_path / f"{split_name}.jsonl"
        save_jsonl(tr_recs, out_file)
        results[split_name] = tr_recs
        print(f"Saved {len(tr_recs)} Turkish records -> {out_file}")

    # Generate review sample in owned data/ path
    if all_translated_for_sample:
        create_stratified_human_review_sample(
            all_translated_for_sample,
            output_path=review_sample_path,
            sample_size=review_sample_size,
            taxonomy_path=taxonomy_path,
        )

    if total_failures:
        raise ValueError(
            f"CORDIS translation incomplete: {total_failures} failures; no manifest published."
        )
    # Manifest
    create_manifest(
        benchmark_name="cordis_h2020_tr",
        output_dir=out_path,
        generation_parameters={
            "backend": trans.name,
            "model_name": trans.model_name,
            "model_revision": model_revision,
            "total_translated": total_translated,
            "cached_count": total_cached,
            "new_count": total_new,
            "failures": total_failures,
            "qa_failures": total_qa_failures,
            "max_records": max_records,
        },
        # Relative to the manifest, so the record does not depend on the checkout location.
        source_metadata={
            name: relative_path(path, out_path)
            for name, path in (("source_en_dir", in_path), ("cache_dir", cache_dir))
        },
        taxonomy_dir=Path(taxonomy_path).parent,
        input_files=[in_path / f"{s}.jsonl" for s in splits],
    )

    return results
