import json
import io

import numpy as np
from typing import Iterable, Dict
from allennlp.data import DatasetReader, Instance, Token, TokenIndexer
from allennlp.data.fields import MetadataField, TextField, ListField, SpanField, \
    SequenceLabelField, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from eventx import NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL, SD4M_RELATION_TYPES, ROLE_LABELS
from eventx.util.utils import one_hot_encode


@DatasetReader.register('snorkel-reader')
class SnorkelReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with io.open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                example = json.loads(line)
                yield self.text_to_instance(example)

    @overrides
    def text_to_instance(self, example: Dict) -> Instance:
        words = example['tokens']
        text_field = TextField([Token(t) for t in words],
                               token_indexers=self._token_indexers)

        # These are required by allennlp for empty list fields
        # see: https://github.com/allenai/allennlp/issues/1391
        dummy_arg_roles_field = ListField([ListField([
            ArrayField(array=np.asarray([0.0]*len(SD4M_RELATION_TYPES)))
        ])])
        dummy_trigger_labels_field = ListField([
            ArrayField(array=np.asarray([0.0]*len(ROLE_LABELS)))
        ])
        dummy_span_list_field = ListField([SpanField(0, 0, text_field)])

        # Extract entity spans
        entities = example['entities']
        entity_ids = [e['id'] for e in entities]
        if len(entities) > 0:
            entity_spans = []
            for entity in entities:
                entity_spans.append(SpanField(span_start=entity['start'],
                                              span_end=entity['end'] - 1,
                                              sequence_field=text_field))
            entity_spans_field = ListField(entity_spans)
        else:
            entity_spans_field = dummy_span_list_field.empty_field()

        entity_tags_field = SequenceLabelField(labels=example['ner_tags'],
                                               sequence_field=text_field,
                                               label_namespace='entity_tags')

        # Extract triggers
        event_triggers = example['event_triggers']
        triggers = [e for e in entities if e['entity_type'] in ['TRIGGER', 'trigger']]
        trigger_ids = [t['id'] for t in triggers]

        if len(triggers) > 0:
            id_to_label_pairs = [(event_trigger['id'], event_trigger['event_type_probs'])
                                 for event_trigger in event_triggers]
            trigger_id_to_label = dict(id_to_label_pairs)
            trigger_labels = []
            trigger_spans = []
            for trigger in triggers:
                trigger_id = trigger['id']
                if trigger_id in trigger_id_to_label:
                    trigger_label = trigger_id_to_label[trigger_id]
                else:
                    trigger_label = one_hot_encode(NEGATIVE_TRIGGER_LABEL, SD4M_RELATION_TYPES)
                trigger_labels.append(ArrayField(array=np.asarray(trigger_label)))
                trigger_spans.append(SpanField(span_start=trigger['start'],
                                               span_end=trigger['end'] - 1,
                                               sequence_field=text_field))

            trigger_labels_field = ListField(trigger_labels)
            trigger_spans_field = ListField(trigger_spans)
        else:
            trigger_labels_field = dummy_trigger_labels_field.empty_field()
            trigger_spans_field = dummy_span_list_field.empty_field()

        event_roles = example['event_roles']
        # Extract argument role labels
        if len(entities) > 0 and len(triggers) > 0:
            # Initialize the argument roles to be the negative class by default
            arg_roles = [[one_hot_encode(NEGATIVE_ARGUMENT_LABEL, ROLE_LABELS)
                          for _ in range(len(entities))]
                         for _ in range(len(triggers))]

            for event_role in event_roles:
                trigger_idx = trigger_ids.index(event_role['trigger'])
                entity_idx = entity_ids.index(event_role['argument'])
                # Set positive event argument roles overwriting the default
                arg_roles[trigger_idx][entity_idx] = event_role['event_argument_probs']

            arg_roles_field = ListField([
                ListField([ArrayField(array=np.asarray(label))
                           for label in token_role_labels])
                for token_role_labels in arg_roles
            ])
        else:
            arg_roles_field = dummy_arg_roles_field.empty_field()

        fields = {
            'metadata': MetadataField({"words": words}),
            'tokens': text_field,
            'entity_tags': entity_tags_field,
            'entity_spans': entity_spans_field,
            'trigger_labels': trigger_labels_field,
            'trigger_spans': trigger_spans_field,
            'arg_roles': arg_roles_field,
        }
        return Instance(fields)
