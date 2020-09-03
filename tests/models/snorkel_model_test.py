from allennlp.common.testing import ModelTestCase

import eventx  # Import custom classes


class SnorkelModelTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/snorkel_model_config.jsonnet',
                          'tests/fixtures/snorkel_sample_data.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
