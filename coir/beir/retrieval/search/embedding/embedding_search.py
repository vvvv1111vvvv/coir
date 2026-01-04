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
        backend=None,
        model_name="selected-model",
        HF_TOKEN="your-hf-token",
        max_cache_size=int(1e8),
        retrieval = None,
    ):
        self.model_name = model_name
        self.HF_TOKEN = HF_TOKEN
        self.max_cache_size = max_cache_size
        if retrieval is None:
            raise ValueError("Retrieval function must be provided.")
        self.retrieval = retrieval
        self.backend = backend
        self.shared_cache = None
        if backend in ["hungarian-cache"]:
            manager = multiprocessing.Manager()
            self.shared_cache = manager.dict()

    def search(
            self, 
            corpus: Dict[str, Dict[str, str]], 
            queries: Dict[str, str], 
            top_k: int, 
            corpus_pl: str, 
            queries_pl: str,
            delete_non_highlight_nodes=None,
    ) -> Dict[str, Dict[str, float]]:
        
        print(f"Building model: {self.model_name}")

        results = dict()
        #retrieve results 
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
        query_texts = [queries[qid] for qid in queries]

        cache_res = dict()
        cache_res['comp_sizes_total_main'] = 0
        cache_res['comp_sizes_total_others'] = 0
        cache_res['cache_hits_total_main'] = 0
        cache_res['cache_hits_total_others'] = 0

        if self.backend is None:
            # self.retrieval = compute_embeddingSimilarity_RAG
            (
                score_list, 
                indices_list            
            ) = self.retrieval(
                model_name=self.model_name,
                HF_TOKEN=self.HF_TOKEN,
                corpus_texts=corpus_texts,
                query_texts=query_texts,
                corpus_pl=corpus_pl,
                queries_pl=queries_pl,                
                top_k=top_k,
                delete_non_highlight_nodes=delete_non_highlight_nodes,
            )
            for scores_src, indices_src in zip(score_list, indices_list):
                scores = {}
                for (corpus_id, score) in zip(
                    [corpus_ids[idx] for idx in indices_src][:top_k],
                    scores_src[:top_k],
                ):
                    scores[corpus_id] = float(score)
                results[list(queries.keys())[len(results)]] = scores
        
        else:
            # self.retrieval = compute_embeddingSimilarity_hungarian_RAG
            (
                score_list, 
                indices_list,
                comp_size_main,
                comp_size_others,
                cache_hits_main,
                cache_hits_others,            
            ) = self.retrieval(
                model_name=self.model_name,
                HF_TOKEN=self.HF_TOKEN,
                corpus_texts=corpus_texts,
                query_texts=query_texts,
                corpus_pl=corpus_pl,
                queries_pl=queries_pl,
                backend=self.backend,
                shared_cache=self.shared_cache,
                max_cache_size=self.max_cache_size,
                top_k=top_k,
                delete_non_highlight_nodes=delete_non_highlight_nodes,
            )
            for scores_src, indices_src in zip(score_list, indices_list):
                scores = {}
                for (corpus_id, score) in zip(
                    [corpus_ids[idx] for idx in indices_src][:top_k],
                    scores_src[:top_k],
                ):
                    scores[corpus_id] = float(score)
                results[list(queries.keys())[len(results)]] = scores
            cache_res['comp_sizes_total_main'] += comp_size_main
            cache_res['comp_sizes_total_others'] += comp_size_others
            cache_res['cache_hits_total_main'] += cache_hits_main
            cache_res['cache_hits_total_others'] += cache_hits_others

        return results, cache_res