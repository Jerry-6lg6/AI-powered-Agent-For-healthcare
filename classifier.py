from transformers import AutoTokenizer, AutoModel
import torch
from typing import Union, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
class Classifier:
    def __init__(self, embedding_model: str = "bge-large-en-v1.5", max_length: int = 1024, device: str = None):
        if embedding_model == "bge-large-en-v1.5":
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.model = AutoModel.from_pretrained(embedding_model)
        if embedding_model == "Qwen/Qwen3-Embedding-0.6B":
            # Load the model
            self.model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
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
        """
        Encode text(s) and return a 2D numpy array of shape (n_items, hidden_size).
        Accepts a single string or a list of strings.
        """
        single = isinstance(input_text, str)
        texts = [input_text] if single else input_text

        encoded_input = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # use [CLS] token embedding (last_hidden_state[:, 0, :]) as in original code
            embeddings = model_output.last_hidden_state[:, 0, :]

        # move to CPU numpy
        embeddings = embeddings.cpu().numpy()
        return embeddings if not single else embeddings[0:1]  # keep 2D shape even for single input

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
            return 0
        else:
            return 1

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
if __name__ == "__main__":
    target = "Wednesday"
    input_list = [" Wednesday. So today is Wednesday.", "Thursday", "Today is  Tuesday,November the eighteenth, 2025.", "Today is November the eighteenth, Wednesday, 2025.","It is November the eighteenth", "Wednesday", " today is 2025","July", "I feel pain", "today is yellow"]
    classifier_0 = Classifier()
    # Test of the match function
    # for i in input_list:
    #     label, sim = classifier_0.match(i, target)
    #     print(f" the input is {i}, the label is {label}, the sim is {sim}")

    # Test of the simplematch function
    for i in input_list:
        label = classifier_0.simplematch(i, target)
        print(f" the input is {i}, the label is {label},")



