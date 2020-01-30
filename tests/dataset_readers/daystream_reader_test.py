from pathlib import Path

from allennlp.common.util import ensure_list
from allennlp.tests.data.dataset_readers.dataset_reader_test import DatasetReaderTest

from eventx import NEGATIVE_ARGUMENT_LABEL
from eventx.dataset_readers.daystream_reader import DaystreamReader


class DaystreamReaderTest(DatasetReaderTest):
    def tearDown(self):
        # Fixes an issue in the basic dataset reader
        Path(self.cache_directory).mkdir(exist_ok=True)
        super().tearDown()

    @staticmethod
    def test_read_sample():
        """Tests parsing the sample file"""
        reader = DaystreamReader()
        instances = ensure_list(reader._read('tests/fixtures/daystream_sample_data.jsonl'))
        assert len(instances) == 5
        instance = instances[0]

        expected_tokens = [
            "Unfall\n", "Abschnitt:", ": ", "Marzahn ", "(B", "Berlin)", ")\n",
            "Gültig ", "ab:", ": ", "09.02.2016 ", "20:06\n", "gesperrt,", ", ", "Unfall\n"
        ]
        metadata_field = instance.fields.get('metadata')
        assert metadata_field['words'] == expected_tokens

        instance_tokens = [t.text for t in instance.fields.get("tokens")]
        assert instance_tokens == expected_tokens

        def extract_spans(ll):
            return list(map(lambda x: (x.span_start, x.span_end), ll))

        def extract_labels(ll):
            return list(map(lambda x: x.label, ll))

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

        expected_trigger_labels = ["Accident", "Obstruction"]
        assert extract_labels(instance.fields.get('trigger_labels')) == expected_trigger_labels

        expected_trigger_spans = [
            (0, 0), (12, 12)
        ]
        assert extract_spans(instance.fields.get('trigger_spans')) == expected_trigger_spans

        # The cross product of "number of triggers" and "number of entities" (2 x 6)
        num_triggers = 2
        num_entities = 6
        expected_arg_roles = []
        for _ in range(num_triggers):
            expected_arg_roles.append([NEGATIVE_ARGUMENT_LABEL] * num_entities)

        # Argument roles of the accident event
        expected_arg_roles[0][2] = 'location'    # Unfall -> Berlin
        expected_arg_roles[0][1] = 'location'    # Unfall -> Marzahn
        expected_arg_roles[0][3] = 'start_date'  # Unfall -> 09.02.2016

        # Argument roles of the obstruction event
        expected_arg_roles[1][2] = 'location'    # gesperrt -> Berlin
        expected_arg_roles[1][1] = 'location'    # gesperrt -> Marzahn
        expected_arg_roles[1][3] = 'start_date'  # gesperrt -> 09.02.2016

        instance_arg_roles = [extract_labels(event_arg_roles)
                              for event_arg_roles in instance.fields.get('arg_roles')]
        assert len(instance_arg_roles) == 2  # The example has 2 triggers
        for token_arg_roles in instance_arg_roles:
            assert len(token_arg_roles) == 6  # Each trigger can have 6 potential arguments
        assert instance_arg_roles == expected_arg_roles
