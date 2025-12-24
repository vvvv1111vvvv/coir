import os
import sys
from .. import BaseSearch
#from .elastic_search import ElasticSearch
import tqdm
import time
from typing import List, Dict
import multiprocessing

def sleep(seconds):
    if seconds: time.sleep(seconds) 

class TSEDSearch():
    def __init__(
        self,
        retrieval = None,
    ):
        if retrieval is None:
            raise ValueError("Retrieval function must be provided.")
        self.retrieval = retrieval

    def search(
            self, 
            corpus: Dict[str, Dict[str, str]], 
            queries: Dict[str, str], 
            top_k: int, 
            db_pl: str, 
            query_pl: str,
    ) -> Dict[str, Dict[str, float]]:
        
        results = dict()
        #retrieve results 
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

        #query_ids = list(queries.keys())
        #query_texts = [queries[qid] for qid in query_ids]

        for query_id, query_code in tqdm.tqdm(queries.items(), total=len(queries), desc="Searching"):
            (
                scores_src,
                indices_src,
            ) = self.retrieval(
                pl1=db_pl,
                pl2=query_pl,
                code_db=corpus_texts,
                query=query_code,
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