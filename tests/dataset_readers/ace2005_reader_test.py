from pathlib import Path

from allennlp.common.util import ensure_list
from allennlp.tests.data.dataset_readers.dataset_reader_test import DatasetReaderTest

from eventx import NEGATIVE_ARGUMENT_LABEL
from eventx.dataset_readers.ace2005_reader import Ace2005Reader


class Ace2005ReaderTest(DatasetReaderTest):
    def tearDown(self):
        # Fixes an issue in the basic dataset reader
        Path(self.cache_directory).mkdir(exist_ok=True)
        super().tearDown()

    @staticmethod
    def test_read_sample():
        """Tests parsing the sample file"""
        reader = Ace2005Reader()
        instances = ensure_list(reader._read('tests/fixtures/ace2005_sample_data.json'))
        assert len(instances) == 5
        instance = instances[3]

        expected_tokens = [
            "Even", "as", "the", "secretary", "of", "homeland", "security", "was", "putting",
            "his", "people", "on", "high", "alert", "last", "month", ",", "a", "30-foot",
            "Cuban", "patrol", "boat", "with", "four", "heavily", "armed", "men", "landed",
            "on", "American", "shores", ",", "utterly", "undetected", "by", "the", "Coast",
            "Guard", "Secretary", "Ridge", "now", "leads", "."
        ]
        metadata_field = instance.fields.get('metadata')
        assert metadata_field['words'] == expected_tokens

        instance_tokens = [t.text for t in instance.fields.get("tokens")]
        assert instance_tokens == expected_tokens

        def extract_spans(ll):
            return list(map(lambda x: [x.span_start, x.span_end], ll))

        def extract_labels(ll):
            return list(map(lambda x: x.label, ll))

        expected_entity_labels = [
            "PER:Individual", "PER:Individual", "PER:Individual", "PER:Individual", "GPE:Nation",
            "ORG:Government", "PER:Group", "VEH:Water", "GPE:Nation", "PER:Group",
            "LOC:Region-General", "ORG:Government", "TIM:time", "TIM:time",
        ]
        assert extract_labels(instance.fields.get('entity_labels')) == expected_entity_labels

        expected_entity_spans = [
            [38, 38], [2, 6], [9, 9], [38, 39], [29, 29], [5, 6], [9, 10], [17, 26],
            [19, 19], [23, 26], [29, 30], [35, 37], [14, 15], [40, 40],
        ]
        assert extract_spans(instance.fields.get('entity_spans')) == expected_entity_spans

        expected_triggers = ["O"] * 27 + ["B-Movement:Transport"] + ["O"] * 15
        assert instance.fields.get('triggers').labels == expected_triggers

        # The cross product of sequence length x number of entities (33 * 14)
        # sequence length: 27 + 1 + 15

        no_event_arg_roles = [NEGATIVE_ARGUMENT_LABEL] * 14  # no-event tokens do not have arguments
        event_arg_roles = [NEGATIVE_ARGUMENT_LABEL] * 7 + [
            'Vehicle',  # "a 30-foot Cuban patrol boat with four heavily armed men"
            NEGATIVE_ARGUMENT_LABEL,
            'Artifact',  # "four heavily armed men"
            'Destination'  # "American shores"
        ] + [NEGATIVE_ARGUMENT_LABEL] * 3
        expected_arg_roles = [no_event_arg_roles] * 27 +\
                             [event_arg_roles] +\
                             [no_event_arg_roles] * 15
        instance_arg_roles = [extract_labels(event_arg_roles)
                              for event_arg_roles in instance.fields.get('arg_roles')]
        assert len(instance_arg_roles) == 43  # The word has 27 tokens
        for token_arg_roles in instance_arg_roles:
            assert len(token_arg_roles) == 14  # Each token can have 14 potential arguments
        assert instance_arg_roles == expected_arg_roles
