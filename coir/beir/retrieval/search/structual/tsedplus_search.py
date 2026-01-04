import os
import sys

import concurrent
from .. import BaseSearch
#from .elastic_search import ElasticSearch
import tqdm
import time
from typing import List, Dict
import multiprocessing
from tree_sitter import Language, Parser, Tree
from tree_sitter_language_pack import get_parser, SupportedLanguage

def sleep(seconds):
    if seconds: time.sleep(seconds) 


class TSEDPlusSearch():
    def __init__(
        self,
        backend="",
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
            queries: Dict[str, str], 
            top_k: int, 
            corpus_pl: str, 
            queries_pl: str,
    ) -> Dict[str, Dict[str, float]]:
        print("TSED_plus backend:", self.backend)
        results = dict()
        #retrieve results 
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
        query_texts = [queries[qid] for qid in queries]

        #query_ids = list(queries.keys())
        #query_texts = [queries[qid] for qid in query_ids]
        cache_res = dict()
        cache_res['comp_sizes_total_main'] = 0
        cache_res['comp_sizes_total_others'] = 0
        cache_res['cache_hits_total_main'] = 0
        cache_res['cache_hits_total_others'] = 0

        (
            score_list,
            indices_list,
            comp_size_main,
            comp_size_others,
            cache_hits_main,
            cache_hits_others,
        ) = self.retrieval(
            code_db_pl=corpus_pl,
            queries_pl=queries_pl,
            code_db=corpus_texts,
            queries=query_texts,
            top_k=top_k,
            n_threads=self.n_threads,
            backend=self.backend,
            shared_cache=self.shared_cache,
            max_cache_size=self.max_cache_size,
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
