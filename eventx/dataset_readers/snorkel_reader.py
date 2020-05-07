import json
import io

import numpy as np
from typing import Iterable, Dict, List, Tuple, Optional
from allennlp.data import DatasetReader, Instance, Token, TokenIndexer, Field
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

                entities = example['entities']
                entity_ids = [e['id'] for e in entities]
                entity_spans = [(e['start'], e['end']) for e in entities]
                triggers = [e for e in entities if e['entity_type'] in ['TRIGGER', 'trigger']]
                trigger_ids = [t['id'] for t in triggers]
                trigger_spans = [(t['start'], t['end']) for t in triggers]
                event_triggers = example['event_triggers']
                event_roles = example['event_roles']

                # If no triggers are found the model can not learn anything from this instance,
                # so skip it
                if len(triggers) == 0:
                    continue

                # Extract event trigger labels
                trigger_labels = []
                id_to_label_pairs = [(event_trigger['id'],
                                      np.asarray(event_trigger['event_type_probs']))
                                     for event_trigger in event_triggers]
                trigger_id_to_label = dict(id_to_label_pairs)
                for trigger in triggers:
                    trigger_id = trigger['id']
                    if trigger_id in trigger_id_to_label:
                        trigger_label = trigger_id_to_label[trigger_id]
                    else:
                        # TODO: every trigger should be labeled in the data, use ignore instead
                        # ABSTAIN instances that are filtered out, are set to negative class
                        # instead they should be ignored or there might be false labels
                        # If they are kept, they have equal probabilities for every class
                        # mark and ignore them during metrics calculation
                        trigger_label = one_hot_encode(NEGATIVE_TRIGGER_LABEL, SD4M_RELATION_TYPES)
                    trigger_labels.append(trigger_label)

                # Extract argument role labels
                # Initialize the argument roles to be the negative class by default
                # TODO: same as event triggers, mark ABSTAINs and ignore during metrics calculation
                arg_role_labels = [[one_hot_encode(NEGATIVE_ARGUMENT_LABEL, ROLE_LABELS)
                                    for _ in range(len(entity_spans))]
                                   for _ in range(len(trigger_spans))]
                for event_role in event_roles:
                    trigger_idx = trigger_ids.index(event_role['trigger'])
                    entity_idx = entity_ids.index(event_role['argument'])
                    # Set positive event argument roles overwriting the default
                    arg_role_labels[trigger_idx][entity_idx] = np.asarray(
                        event_role['event_argument_probs'])

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
                         trigger_labels: Optional[List[np.ndarray]] = None,
                         arg_role_labels: Optional[List[List[np.ndarray]]] = None
                         ) -> Instance:
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
                ArrayField(array=np.asarray(trigger_label))
                for trigger_label in trigger_labels
            ])
            fields['trigger_labels'] = trigger_labels_field

        # Optionally add argument role labels
        if arg_role_labels is not None:
            arg_role_labels_field = ListField([
                ListField([ArrayField(array=np.asarray(label))
                           for label in trigger_role_labels])
                for trigger_role_labels in arg_role_labels
            ])
            fields['arg_roles'] = arg_role_labels_field

        return Instance(fields)
