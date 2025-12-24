import os
import sys
from .. import BaseSearch
#from .elastic_search import ElasticSearch
import tqdm
import time
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import multiprocessing
import faiss
import torch

def sleep(seconds):
    if seconds: time.sleep(seconds) 

class EmbeddingSearch():
    def __init__(
        self,
        model_name="selected-model",
        HF_TOKEN="your-hf-token",
        retrieval = None,
    ):
        self.model_name = model_name
        self.HF_TOKEN = HF_TOKEN
        if retrieval is None:
            raise ValueError("Retrieval function must be provided.")
        self.retrieval = retrieval

    def search(
            self, 
            corpus: Dict[str, Dict[str, str]], 
            queries: Dict[str, str], 
            top_k: int, 
    ) -> Dict[str, Dict[str, float]]:
        
        model = SentenceTransformer(
            self.model_name, 
            token = self.HF_TOKEN,
            device="cpu"
        )

        results = dict()
        #retrieve results 
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
        
        document_embeddings = model.encode(
            corpus_texts,
            normalize_embeddings=True,
        )  # shape: (N, d)
        document_embeddings = document_embeddings.astype("float32")

        for query_id, query_code in tqdm.tqdm(queries.items(), total=len(queries), desc="Searching"):
            (
                scores_src,
                indices_src,
            ) = self.retrieval(
                document_embeddings=document_embeddings,
                query=query_code,
                model=model,
                HF_TOKEN=self.HF_TOKEN,
            )

            # Get top-k values
            scores_src = scores_src[:top_k]
            indices_src = indices_src[:top_k]
            scores = {}
            for (corpus_id, score) in zip(
                [corpus_ids[idx] for idx in indices_src],
                scores_src,
            ):
                scores[corpus_id] = float(score)
                
            results[query_id] = scores

        return results