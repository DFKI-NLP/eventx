import json
from typing import Iterable, Dict, List, Optional, Tuple

from allennlp.data import DatasetReader, Instance, Token, TokenIndexer, Field
from allennlp.data.fields import MetadataField, TextField, LabelField, ListField, SpanField, \
    SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from eventx import NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL


@DatasetReader.register('smartdata-eventx-reader')
class SmartdataEventxReader(DatasetReader):
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

                entities = example['entities']
                triggers = [e for e in entities if e['entity_type'].lower() == 'trigger']
                entity_spans = [(e['start'], e['end']) for e in entities]
                trigger_spans = [(t['start'], t['end']) for t in triggers]
                events = example['events']
                entity_ids = [e['id'] for e in entities]
                trigger_ids = [t['id'] for t in triggers]

                # If no triggers are found the model can not learn anything from this instance, so skip it
                if len(triggers) == 0:
                    continue

                # Extract event trigger labels
                trigger_labels = []
                id_to_label_pairs = [(event['trigger']['id'], event['event_type'])
                                     for event in events]
                trigger_id_to_label = dict(id_to_label_pairs)
                for trigger in triggers:
                    trigger_id = trigger['id']
                    if trigger_id in trigger_id_to_label:
                        trigger_label = trigger_id_to_label[trigger_id]
                    else:
                        trigger_label = NEGATIVE_TRIGGER_LABEL
                    trigger_labels.append(trigger_label)

                # Extract argument role labels
                # Initialize the argument roles to be the negative class by default
                arg_role_labels = [[NEGATIVE_ARGUMENT_LABEL for _ in range(len(entity_spans))]
                                   for _ in range(len(trigger_spans))]
                for event in events:
                    trigger_idx = trigger_ids.index(event['trigger']['id'])
                    for argument in event['arguments']:
                        entity_idx = entity_ids.index(argument['id'])
                        # Set positive event argument roles overwriting the default
                        arg_role_labels[trigger_idx][entity_idx] = argument['role']

                yield self.text_to_instance(tokens=example['tokens'],
                                            ner_tags=example['ner_tags'],
                                            entity_spans=entity_spans,
                                            trigger_spans=trigger_spans,
                                            trigger_labels=trigger_labels,
                                            arg_role_labels=arg_role_labels)

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         ner_tags: List[str],
                         entity_spans: List[Tuple[int, int]],
                         trigger_spans: List[Tuple[int, int]],
                         trigger_labels: Optional[List[str]] = None,
                         arg_role_labels: Optional[List[List[str]]] = None) -> Instance:
        assert len(trigger_spans) > 0, 'Examples without triggers are not supported'

        text_field = TextField([Token(t) for t in tokens], token_indexers=self._token_indexers)
        entity_spans_field = ListField([
            SpanField(span_start=span[0], span_end=span[1] - 1, sequence_field=text_field)
            for span in entity_spans
        ])
        entity_tags_field = SequenceLabelField(labels=ner_tags,
                                               sequence_field=text_field,
                                               label_namespace='entity_tags')
        trigger_spans_field = ListField([
            SpanField(span_start=span[0], span_end=span[1] - 1, sequence_field=text_field)
            for span in trigger_spans
        ])

        fields: Dict[str, Field] = {
            'metadata': MetadataField({"words": tokens}),
            'tokens': text_field,
            'entity_tags': entity_tags_field,
            'entity_spans': entity_spans_field,
            'trigger_spans': trigger_spans_field,
        }

        # Optionally add trigger labels
        if trigger_labels is not None:
            trigger_labels_field = ListField([
                LabelField(label=trigger_label, label_namespace='trigger_labels')
                for trigger_label in trigger_labels
            ])
            fields['trigger_labels'] = trigger_labels_field

        # Optionally add argument role labels
        if arg_role_labels is not None:
            arg_role_labels_field = ListField([
                ListField([LabelField(label=label, label_namespace='arg_role_labels')
                           for label in trigger_role_labels])
                for trigger_role_labels in arg_role_labels
            ])
            fields['arg_roles'] = arg_role_labels_field

        return Instance(fields)

