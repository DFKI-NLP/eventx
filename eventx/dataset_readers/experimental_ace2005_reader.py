import json
from typing import Iterable, Dict, List, Tuple, Optional

from allennlp.data import DatasetReader, Instance, Token, TokenIndexer, Field
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField, LabelField, \
    ListField, SpanField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from eventx import NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL, NEGATIVE_ENTITY_LABEL


@DatasetReader.register('experimental-ace2005-reader')
class ExperimentalAce2005Reader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for line in f.readlines():
                example = json.loads(line)
                words = example['words']
                entities = example['entities']

                # If no entities are found the model can not learn anything from this instance,
                # so skip it
                if len(entities) == 0:
                    continue

                entity_spans = []
                # entity_labels = []
                ner_tags = [NEGATIVE_ENTITY_LABEL] * len(words)

                for entity in entities:
                    entity_span = (entity['start'], entity['end'])
                    entity_spans.append(entity_span)
                    entity_label = entity['entity-type']
                    # entity_labels.append(entity_label)
                    for idx in range(entity['start'], entity['end']):
                        # Encode triggers with IOB2 encoding scheme
                        if idx == entity['start']:
                            ner_tags[idx] = 'B-' + entity_label
                        else:
                            ner_tags[idx] = 'I-' + entity_label

                events = example['golden-event-mentions']
                trigger_labels = [NEGATIVE_TRIGGER_LABEL] * len(words)
                arg_role_labels = [[NEGATIVE_ARGUMENT_LABEL for _ in range(len(entity_spans))]
                                   for _ in range(len(words))]

                for event in events:
                    trigger = event['trigger']
                    trigger_start = trigger['start']
                    trigger_end = trigger['end']
                    for idx in range(trigger_start, trigger_end):
                        label = event['event_type']
                        # Encode triggers with IOB2 encoding scheme
                        if idx == trigger_start:
                            trigger_labels[idx] = 'B-' + label
                        else:
                            trigger_labels[idx] = 'I-' + label

                    # Every entity is a potential negative example for event arguments
                    for argument in event['arguments']:
                        entity_idx = next(idx
                                          for idx, entity in enumerate(entities)
                                          if entity['start'] == argument['start']
                                          and entity['end'] == argument['end']
                                          and entity['entity-type'] == argument['entity-type'])
                        for trigger_idx in range(trigger_start, trigger_end):
                            arg_role_labels[trigger_idx][entity_idx] = argument['role']

                yield self.text_to_instance(tokens=words,
                                            ner_tags=ner_tags,
                                            # entity_labels=entity_labels,
                                            entity_spans=entity_spans,
                                            trigger_labels=trigger_labels,
                                            arg_role_labels=arg_role_labels)

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         ner_tags: List[str],
                         # entity_labels: List[str],
                         entity_spans: List[Tuple[int, int]],
                         trigger_labels: Optional[List[str]] = None,
                         arg_role_labels: Optional[List[List[str]]] = None
                         ) -> Instance:
        assert len(entity_spans) > 0, 'Examples without entities are not supported'

        text_field = TextField([Token(t) for t in tokens], token_indexers=self._token_indexers)
        entity_spans_field = ListField([
            SpanField(span_start=span[0], span_end=span[1] - 1, sequence_field=text_field)
            for span in entity_spans
        ])
        # entity_labels_field = ListField([
        #     LabelField(label=entity_label, label_namespace='entity_labels')
        #     for entity_label in entity_labels
        # ])
        entity_tags_field = SequenceLabelField(labels=ner_tags,
                                               sequence_field=text_field,
                                               label_namespace='entity_tags')

        fields: Dict[str, Field] = {
            'metadata': MetadataField({"words": tokens}),
            'tokens': text_field,
            # 'entity_labels': entity_labels_field,
            'entity_tags': entity_tags_field,
            'entity_spans': entity_spans_field,
        }

        # Optionally add trigger labels
        if trigger_labels is not None:
            trigger_labels_field = SequenceLabelField(labels=trigger_labels,
                                                      sequence_field=text_field,
                                                      label_namespace='trigger_labels')
            fields['triggers'] = trigger_labels_field

        # Optionally add argument role labels
        if arg_role_labels is not None:
            arg_role_labels_field = ListField([
                ListField([LabelField(label=label, label_namespace='arg_role_labels')
                           for label in token_role_labels])
                for token_role_labels in arg_role_labels
            ])
            fields['arg_roles'] = arg_role_labels_field

        return Instance(fields)
