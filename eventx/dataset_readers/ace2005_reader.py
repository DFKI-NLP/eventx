import json
from typing import Iterable, Dict

from allennlp.data import DatasetReader, Instance, Token, TokenIndexer
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField, LabelField, \
    ListField, SpanField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from eventx import NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL


@DatasetReader.register('ace2005-reader')
class Ace2005Reader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for example in json.load(f):
                yield self.text_to_instance(example)

    @overrides
    def text_to_instance(self, example: Dict) -> Instance:
        words = example['words']
        text_field = TextField([Token(t) for t in words],
                               token_indexers=self._token_indexers)

        # These are required by allennlp for empty list fields
        # see: https://github.com/allenai/allennlp/issues/1391
        dummy_arg_roles_field = ListField([ListField([
            LabelField(label='a', label_namespace='arg_role_labels')
        ])])
        dummy_entity_labels_field = ListField([
            LabelField(label='a', label_namespace='entity_labels')
        ])
        dummy_span_list_field = ListField([SpanField(0, 0, text_field)])

        # Extract entities
        entity_labels = []
        entity_spans = []
        entities = example['golden-entity-mentions']
        if len(entities) > 0:
            for entity in entities:
                entity_labels.append(LabelField(label=entity['entity-type'],
                                                label_namespace='entity_labels'))
                entity_spans.append(SpanField(span_start=entity['start'],
                                              span_end=entity['end'] - 1,
                                              sequence_field=text_field))
            entity_labels_field = ListField(entity_labels)
            entity_spans_field = ListField(entity_spans)
        else:
            entity_labels_field = dummy_entity_labels_field.empty_field()
            entity_spans_field = dummy_span_list_field.empty_field()

        triggers = [NEGATIVE_TRIGGER_LABEL] * len(words)
        events = example['golden-event-mentions']

        if len(entity_spans) > 0:
            arg_roles = [[NEGATIVE_ARGUMENT_LABEL for _ in range(len(entity_spans))]
                         for _ in range(len(words))]
        else:
            arg_roles = None

        for event in events:
            trigger = event['trigger']
            trigger_start = trigger['start']
            trigger_end = trigger['end']
            for idx in range(trigger_start, trigger_end):
                label = event['event_type']
                # Encode triggers with IOB2 encoding scheme
                if idx == trigger['start']:
                    triggers[idx] = 'B-' + label
                else:
                    triggers[idx] = 'I-' + label

            if arg_roles:
                # Every entity is a potential negative example for event arguments
                for argument in event['arguments']:
                    entity_idx = next(idx
                                      for idx, entity in enumerate(entities)
                                      if entity['start'] == argument['start']
                                      and entity['end'] == argument['end']
                                      and entity['entity-type'] == argument['entity-type'])
                    for trigger_idx in range(trigger_start, trigger_end):
                        arg_roles[trigger_idx][entity_idx] = argument['role']

        if arg_roles:
            arg_roles_field = ListField([
                ListField([LabelField(label=label, label_namespace='arg_role_labels')
                           for label in token_role_labels])
                for token_role_labels in arg_roles
            ])
        else:
            arg_roles_field = dummy_arg_roles_field.empty_field()

        fields = {
            'metadata': MetadataField({"words": example['words']}),
            'tokens': text_field,
            'entity_labels': entity_labels_field,
            'entity_spans': entity_spans_field,
            'triggers': SequenceLabelField(labels=triggers,
                                           sequence_field=text_field,
                                           label_namespace='trigger_labels'),
            'arg_roles': arg_roles_field,
        }
        return Instance(fields)

