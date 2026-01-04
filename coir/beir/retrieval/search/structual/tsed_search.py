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
            corpus_pl: str, 
            queries_pl: str,
            # ====================================================================
            part: int = None,
            # ====================================================================
    ) -> Dict[str, Dict[str, float]]:
        
        results = dict()
        #retrieve results 
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        # ====================================================================
        # partition queries
        if part is not None:
            query_texts = query_texts[part*30:(part+1)*30]
        # ====================================================================       
        
        (
            score_list,
            indices_list,
        ) = self.retrieval(
            corpus_pl=corpus_pl,
            queries_pl=queries_pl,
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
            #===================================================================
            if part is not None:
                results[list(queries.keys())[len(results) + part*30]] = scores
            else:
                results[list(queries.keys())[len(results)]] = scores
            #===================================================================

        return results