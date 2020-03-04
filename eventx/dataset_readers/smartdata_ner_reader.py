import json
from typing import Iterable, Dict, List, Optional

from allennlp.data import DatasetReader, Instance, TokenIndexer, Token, Field
from allennlp.data.fields import TextField, MetadataField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


@DatasetReader.register('smartdata-ner-reader')
class SmartdataNerReader(DatasetReader):
    """A dataset reader for NER that is compatible with the crf tagger model
    and the sentence tagger as a predictor."""
    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for line in f.readlines():
                example = json.loads(line)
                yield self.text_to_instance(tokens=example['tokens'], tags=example['ner_tags'])

    def text_to_instance(self,
                         tokens: List[str],
                         tags: Optional[List[str]] = None) -> Instance:
        text_field = TextField([Token(t) for t in tokens],
                               token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {
            'tokens': text_field,
            'metadata': MetadataField({"words": tokens}),
        }
        if tags is not None:
            fields['tags'] = SequenceLabelField(labels=tags, sequence_field=text_field)
        return Instance(fields)
