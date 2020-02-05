from pathlib import Path

import numpy as np
from allennlp.common.util import ensure_list
from allennlp.tests.data.dataset_readers.dataset_reader_test import DatasetReaderTest

from eventx import NEGATIVE_ARGUMENT_LABEL, SD4M_RELATION_TYPES, ROLE_LABELS
from eventx.dataset_readers.snorkel_reader import SnorkelReader


class SnorkelReaderTest(DatasetReaderTest):
    def tearDown(self):
        # Fixes an issue in the basic dataset reader
        Path(self.cache_directory).mkdir(exist_ok=True)
        super().tearDown()

    @staticmethod
    def test_read_sample():
        """Tests parsing the sample file"""
        reader = SnorkelReader()
        instances = ensure_list(reader._read(
            '/Users/phuc/develop/python/eventx/tests/fixtures/snorkel_sample_data.jsonl'))
        assert len(instances) == 5
        instance = instances[0]

        expected_tokens = [
            "Unfall", "Abschnitt", ":", "Marzahn", "(", "Berlin", ")",
            "Gültig", "ab", ":", "09.02.2016", "20:06", "gesperrt", ",", "Unfall"
        ]
        metadata_field = instance.fields.get('metadata')
        assert metadata_field['words'] == expected_tokens

        instance_tokens = [t.text for t in instance.fields.get("tokens")]
        assert instance_tokens == expected_tokens

        def extract_spans(ll):
            return list(map(lambda x: (x.span_start, x.span_end), ll))

        def extract_labels(ll):
            return list(map(lambda x: x.array, ll))

        expected_entity_tags = [
            "B-TRIGGER", "O", "O", "B-LOCATION", "O", "B-LOCATION_CITY",
            "O", "O", "O", "O", "B-DATE", "B-TIME", "B-TRIGGER", "O", "O"
        ]
        assert instance.fields.get('entity_tags').labels == expected_entity_tags

        expected_entity_start_spans = [0, 3, 5, 10, 11, 12]
        expected_entity_end_spans = [0, 3, 5, 10, 11, 12]
        expected_entity_spans = list(zip(expected_entity_start_spans,
                                         expected_entity_end_spans))
        assert extract_spans(instance.fields.get('entity_spans')) == expected_entity_spans

        def one_hot_encode(label, label_names, negative_label='O'):
            label = label if label in label_names else negative_label
            class_probs = np.asarray([1.0 if label_name == label else 0.0
                                      for label_name in label_names])
            return class_probs

        def one_hot_decode(class_probs, label_names):
            class_probs_array = np.asarray(class_probs)
            class_name = label_names[class_probs_array.argmax()]
            return class_name

        expected_trigger_labels = ["Accident", "Obstruction"]
        expected_trigger_labels_hot = np.asarray([one_hot_encode(label, SD4M_RELATION_TYPES)
                                                  for label in expected_trigger_labels])
        assert np.array_equal(extract_labels(instance.fields.get('trigger_labels')),
                              expected_trigger_labels_hot)

        expected_trigger_spans = [
            (0, 0), (12, 12)
        ]
        assert extract_spans(instance.fields.get('trigger_spans')) == expected_trigger_spans

        # The cross product of "number of triggers" and "number of entities" (2 x 6)
        num_triggers = 2
        num_entities = 6
        expected_arg_roles = []
        for _ in range(num_triggers):
            expected_arg_roles.append([one_hot_encode(NEGATIVE_ARGUMENT_LABEL, ROLE_LABELS)]
                                      * num_entities)
        # Argument roles of the accident event: Unfall
        expected_arg_roles[0][2] = one_hot_encode('location', ROLE_LABELS)  # -> Berlin
        expected_arg_roles[0][1] = one_hot_encode('location', ROLE_LABELS)  # -> Marzahn
        expected_arg_roles[0][3] = one_hot_encode('start_date', ROLE_LABELS)  # -> 09.02.2016

        # Argument roles of the obstruction event: gesperrt
        expected_arg_roles[1][2] = one_hot_encode('location', ROLE_LABELS)  # -> Berlin
        expected_arg_roles[1][1] = one_hot_encode('location', ROLE_LABELS)  # -> Marzahn
        expected_arg_roles[1][3] = one_hot_encode('start_date', ROLE_LABELS)  # -> 09.02.2016
        expected_arg_roles = np.asarray(expected_arg_roles)
        # print("\n")
        # for x in expected_arg_roles:
        #     print(x)

        instance_arg_roles = np.asarray([extract_labels(event_arg_roles)
                                        for event_arg_roles in instance.fields.get('arg_roles')])
        # print("\n")
        # for x in instance_arg_roles:
        #     print(x)
        assert len(instance_arg_roles) == 2  # The example has 2 triggers
        for token_arg_roles in instance_arg_roles:
            assert len(token_arg_roles) == 6  # Each trigger can have 6 potential arguments
        assert np.array_equal(instance_arg_roles, expected_arg_roles)
