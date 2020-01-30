import json
from typing import Iterable, Dict

from allennlp.data import DatasetReader, Instance, Token, TokenIndexer
from allennlp.data.fields import MetadataField, TextField, LabelField, ListField, SpanField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from eventx import NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL


@DatasetReader.register('daystream-reader')
class DaystreamReader(DatasetReader):
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
                yield self.text_to_instance(example)

    @overrides
    def text_to_instance(self, example: Dict) -> Instance:
        words = example['tokens']
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
        dummy_trigger_labels_field = ListField([
            LabelField(label='a', label_namespace='trigger_labels')
        ])
        dummy_span_list_field = ListField([SpanField(0, 0, text_field)])

        # Extract entities
        entities = example['entities']
        entity_ids = [e['id'] for e in entities]
        if len(entities) > 0:
            entity_labels = []
            entity_spans = []
            for entity in entities:
                entity_labels.append(LabelField(label=entity['entity_type'],
                                                label_namespace='entity_labels'))
                entity_spans.append(SpanField(span_start=entity['start'],
                                              span_end=entity['end'] - 1,
                                              sequence_field=text_field))
            entity_labels_field = ListField(entity_labels)
            entity_spans_field = ListField(entity_spans)
        else:
            entity_labels_field = dummy_entity_labels_field.empty_field()
            entity_spans_field = dummy_span_list_field.empty_field()

        # Extract triggers
        events = example['events']
        triggers = [e for e in entities if e['entity_type'] == 'TRIGGER']
        trigger_ids = [t['id'] for t in triggers]

        if len(triggers) > 0:
            id_to_label_pairs = [(event['trigger']['id'], event['event_type'])
                                   for event in events]
            trigger_id_to_label = dict(id_to_label_pairs)
            trigger_labels = []
            trigger_spans = []
            for trigger in triggers:
                trigger_id = trigger['id']
                if trigger_id in trigger_id_to_label:
                    trigger_label = trigger_id_to_label[trigger_id]
                else:
                    trigger_label = NEGATIVE_TRIGGER_LABEL
                trigger_labels.append(LabelField(label=trigger_label,
                                                 label_namespace='trigger_labels'))
                trigger_spans.append(SpanField(span_start=trigger['start'],
                                               span_end=trigger['end'] - 1,
                                               sequence_field=text_field))

            trigger_labels_field = ListField(trigger_labels)
            trigger_spans_field = ListField(trigger_spans)
        else:
            trigger_labels_field = dummy_trigger_labels_field.empty_field()
            trigger_spans_field = dummy_span_list_field.empty_field()

        # Extract argument role labels
        if len(entities) > 0 and len(triggers) > 0:
            # Initialize the argument roles to be the negative class by default
            arg_roles = [[NEGATIVE_ARGUMENT_LABEL for _ in range(len(entities))]
                         for _ in range(len(triggers))]

            for event in events:
                trigger_idx = trigger_ids.index(event['trigger']['id'])
                for argument in event['arguments']:
                    entity_idx = entity_ids.index(argument['id'])
                    # Set positive event argument roles overwriting the default
                    arg_roles[trigger_idx][entity_idx] = argument['role']

            arg_roles_field = ListField([
                ListField([LabelField(label=label, label_namespace='arg_role_labels')
                           for label in token_role_labels])
                for token_role_labels in arg_roles
            ])
        else:
            arg_roles_field = dummy_arg_roles_field.empty_field()

        fields = {
            'metadata': MetadataField({"words": words}),
            'tokens': text_field,
            'entity_labels': entity_labels_field,
            'entity_spans': entity_spans_field,
            'trigger_labels': trigger_labels_field,
            'trigger_spans': trigger_spans_field,
            'arg_roles': arg_roles_field,
        }
        return Instance(fields)

