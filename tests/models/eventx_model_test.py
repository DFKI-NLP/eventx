from allennlp.common.testing import ModelTestCase

import eventx  # Import custom classes


class EventxModelTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/eventx_model_config.jsonnet',
                          'tests/fixtures/ace2005_sample_data.json')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
