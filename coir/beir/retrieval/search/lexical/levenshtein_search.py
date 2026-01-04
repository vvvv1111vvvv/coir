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

class LevenshteinSearch():
    def __init__(
        self,
        backend=None,
        retrieval = None,
        max_cache_size: int = int(1e8),
    ):
        self.backend = backend
        self.max_cache_size = max_cache_size
        self.shared_cache = None    

        if retrieval is None:
            raise ValueError("Retrieval function must be provided.")
        self.retrieval = retrieval

        if backend in ["hungarian-cache"]:
            manager = multiprocessing.Manager()
            self.shared_cache = manager.dict()

    def search(
            self, 
            corpus: Dict[str, Dict[str, str]], 
            queries: Dict[str, str], 
            top_k: int, 
            corpus_pl,
            queries_pl,
            delete_non_highlight_nodes,
    ) -> Dict[str, Dict[str, float]]:
        
        results = dict()

        #retrieve results 
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        cache_res = dict()
        cache_res['comp_sizes_total_main'] = 0
        cache_res['comp_sizes_total_others'] = 0
        cache_res['cache_hits_total_main'] = 0
        cache_res['cache_hits_total_others'] = 0

        (
            distance_list,
            indices_list,
            cache_result
        ) = self.retrieval(
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
            

        for distance_src, indices_src in zip(distance_list, indices_list):
            distances = {}
            for (corpus_id, distance) in zip(
                [corpus_ids[idx] for idx in indices_src][:top_k],
                distance_src[:top_k],
            ):
                distances[corpus_id] = float(distance)
            results[list(queries.keys())[len(results)]] = distances

        if cache_result is not None:
            cache_res['comp_sizes_total_main'] += cache_result[0]
            cache_res['comp_sizes_total_others'] += cache_result[1]
            cache_res['cache_hits_total_main'] += cache_result[2]
            cache_res['cache_hits_total_others'] += cache_result[3]

        return results, cache_res
    