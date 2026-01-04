import os
import sys
from .. import BaseSearch
#from .elastic_search import ElasticSearch
import tqdm
import time
from typing import List, Dict
import multiprocessing
from transformers import AutoTokenizer

def sleep(seconds):
    if seconds: time.sleep(seconds) 

class TFIDFSearch():
    def __init__(
        self,
        retrieval = None,
        HF_TOKEN: str = None,
    ):
        if retrieval is None:
            raise ValueError("Retrieval function must be provided.")
        self.retrieval = retrieval
        self.HF_TOKEN = HF_TOKEN

    def search(
            self, 
            corpus: Dict[str, Dict[str, str]], 
            queries: Dict[str, str], 
            top_k: int, 
            tokenizer_name: str,
    ) -> Dict[str, Dict[str, float]]:
        
        results = dict()
        #retrieve results 
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]


        (
            score_list,
            indices_list,
        ) = self.retrieval(
            tokenizer_name=tokenizer_name,
            corpus_texts=corpus_texts,
            query_texts=query_texts,
            top_k=top_k,
        )

        for scores_src, indices_src in zip(score_list, indices_list):
            scores = {}
            for (corpus_id, score) in zip(
                [corpus_ids[idx] for idx in indices_src][:top_k],
                scores_src[:top_k],
            ):
                scores[corpus_id] = float(score)
            results[list(queries.keys())[len(results)]] = scores

        return results