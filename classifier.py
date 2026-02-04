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
    "I might die"
]

NEGATIONS = ["not", "no", "never", "n't", "cannot", "can't", "dont", "don't"]

TURN_WORDS = [
    "but", "however", "though", "although",
    "yet", "still", "instead", "no", 'No'
]

SPLIT_PUNCT = r"[,.!?;:]"


def has_negation(text: str, window: int = 5) -> bool:
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok in NEGATIONS:
            return True
    return False


class Classifier:
    def __init__(self, embedding_model: str = "bge-large-en-v1.5",
                 max_length: int = 1024,
                 device: str = None):

        if embedding_model == "bge-large-en-v1.5":

            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.model = AutoModel.from_pretrained(embedding_model)

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
        input_text = str(input_text)
        target_text = str(target_text)
        if target_text in input_text:
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
            sub = re.split(r"\b(" + "|".join(TURN_WORDS) + r")\b", p)
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

    def state_match(self, input_text: str, target_text: str):

        state = {
            "correct": 0,
            "dont_know": 0,
            "stop": 0,
            "emergency": 0,
        }
        score = 0
        target_text = target_text.lower()

        # 1. Cleaning
        clean_text, segments, has_turn = self.cleaning(input_text)

        # print(segments)
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            # 3. Special three types of semantic detection（embedding match）
            is_dk = self.match(seg, DONT_KNOW_VOCAB, 0.85)[0] == 0
            is_stop = self.match(seg, STOP_VOCAB, 0.90)[0] == 0
            is_emer = self.match(seg, EMERGENCY_VOCAB, 0.80)[0] == 0

            if is_dk:
                state["dont_know"] = 1
            if is_stop:
                state["stop"] = 1
            if is_emer:
                state["emergency"] = 1

            special_case = is_dk or is_stop or is_emer

            # 4. simple match + Negative detection
            simple_hit = (self.simplematch(seg, target_text) == 1)
            neg = has_negation(seg)

            if simple_hit and not neg:
                # print("Hit")
                score += 1

            # 5. f there are transitional words and the current paragraph is not one of the three special cases → before invalidation, make a judgment
            if has_turn and (not special_case):
                score = 0
            # print(f"the seg is {seg}, the score is {score}")

        # 6. Final detection
        if score == 0:
            state["correct"] = 0  # incorrect
        else:
            state["correct"] = 1  # correct

        return state, score


if __name__ == "__main__":
    target = "February"
    input_list = [" February. NO it is January.", "It is January", "This month is February", "NO, stop it",
                  "I'm feeling pain, help", "I don't know"]
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
