from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from typing import Union, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import faiss

DONT_KNOW_VOCAB = [
    "I don't know",
    "I have no idea",
    "I am not sure",
    "I don't remember",
    "I can't remember",
    "I forgot",
    "I can't recall it",
    "I have forgotten",
    "I don't understand",
    "I didn't understand the question",
    "I am confused",
    "this is confusing",
    "I am not certain",
    "maybe, I'm not sure",
    "I really don't know",
    "I cannot think of it right now",
    "it slipped my mind",
    "my memory is blank",
    "I can't think of anything",
    "I don't know what you mean"
]
STOP_VOCAB = [
    "I wanna rest.",
    'I want a rest',
    "stop",
    "stop it",
    "please stop",
    "I want to stop",
    "let's stop",
    "stop asking me",
    "don't ask me that",
    "I don't want to continue",
    "I don't want to answer",
    "I'd like to stop now",
    "that's enough",
    "end this",
    "terminate the session",
    "no more questions",
    "leave me alone",
    "I don't want to talk anymore",
    "can we stop here",
    "I want to quit",
    "please end this conversation",
]

EMERGENCY_VOCAB = [
    "help me",
    "this is an emergency",
    "I need urgent help",
    "call an ambulance",
    "call the police",
    "I am in danger",
    "someone is hurt",
    "I am hurt",
    "I can't breathe",
    "I am having a heart attack",
    "there is a fire",
    "I am bleeding",
    "I need immediate help",
    "this is serious",
    "please help me now",
    "I am not safe",
    "someone is attacking me",
    "I need medical help",
    "it's an emergency situation",
    "I might die",
    "I am not feeling well",
    "I don't feel well",
    "I feel sick",
    "I feel dizzy",
    "I feel faint",
    "I need assistance",
    "I need help",
    "I need a nurse",
    "I need a doctor",
    "please get a nurse",
    "please get help",
    "something is wrong",
    "I want to die",
    "I want to hurt myself",
    "I can't move",
    "I feel very unwell",
    "I am in pain",
    "my chest hurts"
]

REPEAT_VOCAB = [
    "can you repeat that",
    "could you repeat that",
    "please repeat that",
    "repeat the question",
    "say that again",
    "can you say that again",
    "could you say that again",
    "please say that again",
    "I did not hear you",
    "I didn't hear you",
    "I couldn't hear you",
    "I can't hear you",
    "I could not hear that",
    "pardon",
    "pardon me",
    "excuse me",
    "what did you say",
    "what was that",
    "come again",
    "sorry I missed that",
    "I missed that",
    "can you speak louder",
    "speak up please",
    "I didn't catch that",
    "I did not catch that",
]

REQUIRE_VOCAB = [
    "I want some water",
    "I want something to eat",
    "I want some food",
    "I want a drink",
    "I want some tea",
    "I want some coffee",
    "I am hungry",
    "I am thirsty",
    "I need some water",
    "I need something to eat",
    "I need food",
    "I need a drink",
    "please bring me some water",
    "please bring me some food",
    "please get me something to eat",
    "please get me a drink",
    "could you bring me some water",
    "could you bring me something to eat",
    "could you get me some food",
    "could you get me a drink",
    "can I have some water",
    "can I have something to eat",
    "can I have some food",
    "can I get a drink",
    "I would like some water",
    "I would like something to eat",
    "I would like some food",
    "I would like a drink",
]

NEGATIONS = ["not", "no", "never", "n't", "cannot", "can't", "dont", "don't"]

TURN_WORDS = [
    "but", "however", "though", "although",
    "yet", "still", "instead", "no", 'No'
]
GLOBAL_VOCABS = {
    "dont_know": DONT_KNOW_VOCAB,
    "stop": STOP_VOCAB,
    "emergency": EMERGENCY_VOCAB,
    "repeat": REPEAT_VOCAB,
    "require": REQUIRE_VOCAB,
}
SPLIT_PUNCT = r"[,.!?;:]"


def has_negation(text: str, window: int = 5) -> bool:
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok in NEGATIONS:
            return True
    return False


class Classifier:
    def __init__(self, embedding_model: str = "BAAI/bge-large-en-v1.5",
                 max_length: int = 1024,
                 device: str = None):

        if embedding_model == "BAAI/bge-large-en-v1.5":
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
            self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")

        if embedding_model == "Qwen/Qwen3-Embedding-0.6B":
            # Load the model
            self.model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                model_kwargs={"attn_implementation": "eager", "device_map": "auto"},
                tokenizer_kwargs={"padding_side": "left"},
            )

        # set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.max_length = max_length
        self.build_faiss_index()

    def embedding(self, input_text: Union[str, List[str]]) -> np.ndarray:
        single = isinstance(input_text, str)
        texts = [input_text] if single else input_text
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings = self.normalize_vectors(embeddings)

        return embeddings if not single else embeddings[0]

    def build_faiss_index(self):

        self.faiss_indexes = {}
        self.faiss_text_map = {}

        for key, vocab in GLOBAL_VOCABS.items():
            # 1 embedding vocab
            vecs = self.embedding(vocab)

            dim = vecs.shape[1]

            # 2 create faiss index
            index = faiss.IndexFlatIP(dim)  # cosine similarity

            # vectors already normalized
            index.add(vecs.astype(np.float32))

            # save
            self.faiss_indexes[key] = index
            self.faiss_text_map[key] = vocab

    def faiss_match(self, text, vocab_type, threshold):

        vec = self.embedding(text).astype(np.float32).reshape(1, -1)

        index = self.faiss_indexes[vocab_type]

        scores, ids = index.search(vec, 1)

        score = scores[0][0]

        if score >= threshold:
            return True, score
        else:
            return False, score

    @staticmethod
    def normalize_vectors(v: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> np.ndarray:
        """
        Normalize row vectors to unit L2 norm. Returns numpy array.
        Accepts either numpy arrays or torch tensors.
        """
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        v = np.asarray(v, dtype=np.float64)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return v / norms

    def _cosine_similarity_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix between rows of a and rows of b.
        Result shape: (a.shape[0], b.shape[0])
        """
        a_n = self.normalize_vectors(a)
        b_n = self.normalize_vectors(b)
        return np.matmul(a_n, b_n.T)

    def simplematch(self,
                    input_text: str,
                    target_text: str
                    ):
        input_text  = str(input_text)
        target_text = str(target_text)
        if re.search(rf"\b{re.escape(target_text)}\b", input_text):
            return 1
        else:
            return 0

    def cleaning(self, text: str):
        # 1. A clear dictionary of interjections (expandable as needed)
        filler_words = [
            "uh", "um", "umm", "hmm", "ah", "ahh", "er", "erm",
            "eh", "mm", "mmm", "enn", "ennn", "ann", "annn"
        ]

        # Construct regular expressions: Whole-word matching to avoid accidentally deleting normal words
        filler_pattern = r"\b(" + "|".join(filler_words) + r")\b"

        text = re.sub(filler_pattern, " ", text)

        # 2. Delete meaningless syllables with "elongated sounds"
        elongated_pattern = r"\b([aeioumn])\1{2,}\b"
        text = re.sub(elongated_pattern, " ", text)

        # 3. Remove unnecessary Spaces and Spaces before punctuation
        text = re.sub(r"\s+", " ", text).strip()

        # 4. All letter are changed to lowercase
        text = text.lower()
        clean_text = text

        has_turn = any(re.search(rf"\b{w}\b", clean_text) for w in TURN_WORDS)

        parts = re.split(SPLIT_PUNCT, text)

        segments = []
        for p in parts:
            sub = re.split(r"(?:(?<=^)|(?<=[,.!?]))\s*(" + "|".join(TURN_WORDS) + r")\b", p)
            for s in sub:
                s = s.strip()
                if s and s not in TURN_WORDS:
                    segments.append(s)
        # print(f"{clean_text}\n{segments}\n{has_turn}")
        return clean_text, segments, has_turn

    def match(
            self,
            input_text_or_vec: Union[str, List[str], np.ndarray, torch.Tensor],
            target_text_or_vec: Union[str, List[str], np.ndarray, torch.Tensor],
            threshold: float = 0.9,
            is_normalize: bool = False,
            aggregate: str = "max"
    ) -> Tuple[int, float]:
        """
        Returns (label, similarity_score)
          - label: 0 if similarity >= threshold (match), else 1
          - similarity_score: aggregated similarity (float between -1 and 1)
        Parameters:
          - input_text_or_vec / target_text_or_vec: can be raw text (str or list) OR precomputed vectors (numpy or torch)
          - is_normalize: if True, assumes passed arrays are already normalized (not used for text inputs)
          - aggregate: how to aggregate similarity matrix to single score: "max", "mean", or "min"
        """

        # prepare embeddings if strings/lists provided
        def to_vectors(x):
            if isinstance(x, (str, list)):
                return self.embedding(x)
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.asarray(x)

        a = to_vectors(input_text_or_vec)
        b = to_vectors(target_text_or_vec)

        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        if is_normalize:
            # assume already normalized; just compute dot-products
            sim_mat = np.matmul(a, b.T)
        else:
            sim_mat = self._cosine_similarity_matrix(a, b)

        # aggregate to single similarity score
        if aggregate == "max":
            sim_score = float(np.max(sim_mat))
        elif aggregate == "mean":
            sim_score = float(np.mean(sim_mat))
        elif aggregate == "min":
            sim_score = float(np.min(sim_mat))
        else:
            raise ValueError("aggregate must be one of 'max', 'mean', 'min'")

        label = 0 if sim_score >= threshold else 1
        return label, sim_score

    # ──────────────────────────────────────────────────────────────────
    # ENSEMBLE HELPERS
    # ──────────────────────────────────────────────────────────────────

    # Written-out number words → digit strings for common orientation answers
    _NUMBER_WORDS = {
        "zero":"0","one":"1","two":"2","three":"3","four":"4",
        "five":"5","six":"6","seven":"7","eight":"8","nine":"9",
        "ten":"10","eleven":"11","twelve":"12","thirteen":"13",
        "fourteen":"14","fifteen":"15","sixteen":"16","seventeen":"17",
        "eighteen":"18","nineteen":"19","twenty":"20","twenty one":"21",
        "twenty two":"22","twenty three":"23","twenty four":"24",
        "twenty five":"25","twenty six":"26","twenty seven":"27",
        "twenty eight":"28","twenty nine":"29","thirty":"30",
        "thirty one":"31",
        # Years spoken as pairs e.g. "twenty twenty six"
        "twenty twenty":"2020","twenty twenty one":"2021",
        "twenty twenty two":"2022","twenty twenty three":"2023",
        "twenty twenty four":"2024","twenty twenty five":"2025",
        "twenty twenty six":"2026","twenty twenty seven":"2027",
    }

    # Month name variants (abbreviations → full names)
    _MONTH_VARIANTS = {
        "jan":"january","feb":"february","mar":"march","apr":"april",
        "jun":"june","jul":"july","aug":"august","sep":"september",
        "sept":"september","oct":"october","nov":"november","dec":"december",
    }

    # Weekday variants
    _DAY_VARIANTS = {
        "mon":"monday","tue":"tuesday","tues":"tuesday",
        "wed":"wednesday","thu":"thursday","thur":"thursday","thurs":"thursday",
        "fri":"friday","sat":"saturday","sun":"sunday",
    }

    def _normalise_text(self, text: str) -> str:
        """
        Normalise spoken variants to their canonical form.
        e.g. "twenty twenty six" → "2026", "feb" → "february"
        """
        t = text.lower().strip()
        # replace written-out years / numbers
        for word, digit in sorted(self._NUMBER_WORDS.items(),
                                   key=lambda x: -len(x[0])):
            t = re.sub(rf"\b{re.escape(word)}\b", digit, t)
        # replace month abbreviations
        for abbr, full in self._MONTH_VARIANTS.items():
            t = re.sub(rf"\b{re.escape(abbr)}\b", full, t)
        # replace day abbreviations
        for abbr, full in self._DAY_VARIANTS.items():
            t = re.sub(rf"\b{re.escape(abbr)}\b", full, t)
        return t

    def _stage1_string(self, clean_input: str, target: str) -> bool:
        """
        Stage 1 — fast whole-word string match.
        Uses word boundaries so 'yes' does not match inside 'yesterday'.
        """
        return bool(re.search(rf"\b{re.escape(target)}\b", clean_input))

    def _stage2_keyword(self, clean_input: str, target: str) -> bool:
        """
        Stage 2 — normalised whole-word keyword match.
        Normalises both input and target (number words, abbrevs)
        then checks for whole-word match. No embeddings used.
        """
        norm_input  = self._normalise_text(clean_input)
        norm_target = self._normalise_text(target)
        return bool(re.search(rf"\b{re.escape(norm_target)}\b", norm_input))

    # ──────────────────────────────────────────────────────────────────

    def state_match(self, input_text: str, target_text: str,
                    context: list = None):
        """
        Classify a patient response using a 3-stage ensemble pipeline:

          Stage 1 — Exact string match    (instant, no model)
          Stage 2 — Normalised keyword    (fast, no model)
          Stage 3 — BGE + FAISS semantic  (full model, only if 1+2 fail)

        Special states (emergency, stop, dont_know, repeat) always go
        through FAISS — they cannot be safely string-matched.

        Args:
            input_text:  What the patient said.
            target_text: The expected correct answer.
            context:     Optional list of recent exchanges
                         [{"question": ..., "answer": ...}, ...]
                         Last 2 entries prepended for context-aware scoring.

        Returns:
            (state dict, score int, confidence float)
        """
        state = {
            "correct":   0,
            "dont_know": 0,
            "stop":      0,
            "emergency": 0,
            "repeat":    0,
            "require":   0,
            "free_talk": 0,
            "silence":   0,
        }
        confidence = 0.0
        target_low = target_text.lower()

        # ── Build context-enriched input ──────────────────────────────
        if context:
            recent = context[-2:]
            parts  = []
            for turn in recent:
                q = turn.get("question", "").strip()
                a = turn.get("answer",   "").strip()
                if a:
                    parts.append(f"Previously asked: {q} Patient said: {a}")
            prefix   = (" | ".join(parts) + " | Current: ") if parts else ""
            enriched = prefix + input_text
        else:
            enriched = input_text

        # ── Clean raw input (silence detection uses this) ─────────────
        raw_clean, _, _ = self.cleaning(input_text)

        # ── Clean enriched input for segment-level checks ─────────────
        _, segments, has_turn = self.cleaning(enriched)

        # ═══════════════════════════════════════════════════════════════
        # SPECIAL STATE CHECK — always via FAISS
        # ═══════════════════════════════════════════════════════════════
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            is_dk,      _ = self.faiss_match(seg, "dont_know", 0.85)
            is_stop,    _ = self.faiss_match(seg, "stop",      0.90)
            is_emer,    _ = self.faiss_match(seg, "emergency", 0.75)
            is_repeat,  _ = self.faiss_match(seg, "repeat",    0.80)
            # require checked after repeat — "could you repeat" must not trigger require
            is_require, _ = self.faiss_match(seg, "require",   0.82) if not is_repeat else (False, 0.0)

            # Single-word segments must not trigger emergency, stop, or require —
            # these states need meaningful context to be reliable.
            # dont_know and repeat are still allowed on single words ("pardon", "stop").
            seg_word_count = len(seg.split())
            if seg_word_count < 2:
                is_emer    = False
                is_stop    = False
                is_require = False

            if is_dk:      state["dont_know"] = 1
            if is_stop:    state["stop"]      = 1
            if is_emer:    state["emergency"] = 1
            if is_repeat:  state["repeat"]    = 1
            if is_require: state["require"]   = 1

            if any([is_dk, is_stop, is_emer, is_repeat, is_require]):
                return state, 2, 0.0

        # ═══════════════════════════════════════════════════════════════
        # STAGE 1 — Exact string match (zero cost)
        # ═══════════════════════════════════════════════════════════════
        if self._stage1_string(raw_clean, target_low):
            if not has_negation(raw_clean):
                print("[ENSEMBLE] Stage 1 hit — exact match")
                state["correct"] = 1
                return state, 1, 1.0

        # ═══════════════════════════════════════════════════════════════
        # STAGE 2 — Normalised keyword match (zero cost)
        # ═══════════════════════════════════════════════════════════════
        if self._stage2_keyword(raw_clean, target_low):
            if not has_negation(raw_clean):
                print("[ENSEMBLE] Stage 2 hit — normalised keyword match")
                state["correct"] = 1
                return state, 1, 0.95

        # ═══════════════════════════════════════════════════════════════
        # STAGE 3 — BGE + FAISS semantic (full model)
        # Only reached when stages 1 and 2 both fail
        # ═══════════════════════════════════════════════════════════════
        print("[ENSEMBLE] Stage 3 — semantic classification")

        score = 0
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            simple_hit = (self.simplematch(seg, target_low) == 1)
            neg        = has_negation(seg)

            if simple_hit and not neg:
                score += 1
                confidence = 1.0

            if has_turn:
                score = 0

        if score == 0:
            state["correct"] = 0

            # Silence — raw input only
            if not raw_clean or len(raw_clean.strip()) < 3:
                valid = [s for s in segments if s and len(s.strip()) >= 2]
                if not valid:
                    state["silence"] = 1

            # Semantic similarity via BGE (uses context-enriched text)
            _, sem_score = self.match(enriched, target_low, 0.55)
            confidence   = max(0.0, float(sem_score))
            if sem_score >= 0.55:
                state["free_talk"] = 1
        else:
            state["correct"] = 1

        if score >= 1:
            score = 1

        return state, score, round(confidence, 4)

    def is_negative(self, text: str) -> bool:
        negatives = [
            r"\bno(?![a-zA-Z])",
            r"not ready",
            r"not yet",
            r"i am not ready",
            r"i'm not ready",
            r"not now",
            r"wait",
            r"give me a minute"
        ]

        text_lower = text.lower().strip()

        return any(re.search(pattern, text_lower) for pattern in negatives)


if __name__ == "__main__":
    target = "February"
    input_list = [" it is January. NO,February. ", "It is January","It is February. It is  February", "This month is February", "NO, stop it",
                  "I'm feeling pain, help", "I have no idea.", "I have no idea what are you talking about.",
                  "I don't know.", "I don't know what you said."]
    classifier_0 = Classifier()
    # Test of the match function
    # for i in input_list:
    #     label, sim = classifier_0.match(i, target)
    #     print(f" the input is {i}, the label is {label}, the sim is {sim}")

    # Test of the simplematch function
    for i in input_list:
        # label = classifier_0.simplematch(i, target)
        # print(f" the input is {i}, the label is {label},")
        label_2 = classifier_0.state_match(i, target)
        print(f" the input is {i}, the label is {label_2},")
