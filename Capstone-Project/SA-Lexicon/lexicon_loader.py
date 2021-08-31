from collections import defaultdict
from typing import Tuple

import xml.etree.ElementTree as ET


class LexiconLoader:
    DEFAULT_LEXICON_PATHS = ('../ML-SentiCon/senticon.en.xml', '../ML-SentiCon/senticon.es.xml')

    MERGE_STRATEGY_AVERAGE = 0
    MERGE_STRATEGY_DROP_ALL = 1
    MERGE_STRATEGY_KEEP_FIRST = 2

    def __init__(self, lexicon_paths: Tuple[str] = DEFAULT_LEXICON_PATHS,
                 merge_strategy: int = MERGE_STRATEGY_DROP_ALL,
                 max_level: int = 8, max_std: float = 1.0):
        self.lexicon_paths = lexicon_paths
        self.merge_strategy = merge_strategy
        self.max_level = max_level
        self.max_std = max_std

    def load_single_lexicon(self, lexicon_path: str) -> dict:
        d = dict()
        tree = ET.parse(lexicon_path)
        root = tree.getroot()
        for layer in root:
            if int(layer.attrib['level']) > self.max_level:
                continue
            for sentiment_bucket in layer:
                for lemma in sentiment_bucket:
                    lemma_text = lemma.text.strip()
                    if float(lemma.attrib['std']) <= self.max_std:
                        d[lemma_text] = float(lemma.attrib['pol'])
        return d

    def load_all_and_merge(self) -> dict:
        dicts = []
        for lexicon_path in self.lexicon_paths:
            dicts.append(self.load_single_lexicon(lexicon_path))
        combined_dict = defaultdict(float)

        if self.merge_strategy == LexiconLoader.MERGE_STRATEGY_AVERAGE:
            counts = defaultdict(int)
            for d in dicts:
                for k in d:
                    combined_dict[k] += d[k]
                    counts[k] += 1
            for k in combined_dict:
                combined_dict[k] /= counts[k]
            return combined_dict
        elif self.merge_strategy == LexiconLoader.MERGE_STRATEGY_DROP_ALL:
            counts = defaultdict(int)
            for d in dicts:
                for k in d:
                    combined_dict[k] = d[k]
                    counts[k] += 1
            combined_dict = {k: combined_dict[k] for k in combined_dict if counts[k] == 1}
        elif self.merge_strategy == LexiconLoader.MERGE_STRATEGY_KEEP_FIRST:
            for d in dicts:
                for k in d:
                    if k in combined_dict:
                        continue
                    combined_dict[k] = d[k]

        return combined_dict
