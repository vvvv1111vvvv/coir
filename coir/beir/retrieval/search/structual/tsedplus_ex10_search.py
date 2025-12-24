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

class TSEDPlusEx10Search():
    def __init__(
        self,
        backend="xted-cpu-hungarian-cache",
        n_threads=2,
        max_cache_size=5e6,
        retrieval = None,
    ):
        self.backend = backend
        self.n_threads = n_threads
        self.max_cache_size = max_cache_size
        self.shared_cache = None
        if retrieval is None:
            raise ValueError("Retrieval function must be provided.")
        self.retrieval = retrieval
        if backend in ["xted-cpu-cache", "xted-cpu-hungarian-cache"]:
            manager = multiprocessing.Manager()
            self.shared_cache = manager.dict()

    def search(
            self, 
            corpus: Dict[str, Dict[str, str]], 
            queries: Dict[str, str], top_k: int, 
            db_pl: str, 
            query_pl: str,
    ) -> Dict[str, Dict[str, float]]:
        
        results = dict()
        #retrieve results 
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

        #query_ids = list(queries.keys())
        #query_texts = [queries[qid] for qid in query_ids]
        cache_res = dict()
        cache_res['comp_sizes_total_main'] = 0
        cache_res['comp_sizes_total_others'] = 0
        cache_res['cache_hits_total_main'] = 0
        cache_res['cache_hits_total_others'] = 0
        for query_id, query_code in tqdm.tqdm(queries.items(), total=len(queries), desc="Searching"):
            (
                _,
                _,
                scores_src,
                indices_scores_src,
                comp_size_main,
                comp_size_others,
                cache_hits_main,
                cache_hits_others,
            ) = self.retrieval(
                db_pl=db_pl,
                query_pl=query_pl,
                code_db=corpus_texts,
                query=query_code,
                n_threads=self.n_threads,
                backend=self.backend,
                shared_cache=self.shared_cache,
                max_cache_size=self.max_cache_size,
            )

            # Get top-k values
            scores_src = scores_src[:top_k]
            indices_scores_src = indices_scores_src[:top_k]
            scores = {}
            for (corpus_id, score) in zip(
                [corpus_ids[idx] for idx in indices_scores_src],
                scores_src,
            ):
                scores[corpus_id] = float(score)
                
            results[query_id] = scores

            if self.backend in ["xted-cpu-cache", "xted-cpu-hungarian-cache"]:
                cache_res['comp_sizes_total_main'] += comp_size_main
                cache_res['comp_sizes_total_others'] += comp_size_others
                cache_res['cache_hits_total_main'] += cache_hits_main
                cache_res['cache_hits_total_others'] += cache_hits_others

        return results, cache_res