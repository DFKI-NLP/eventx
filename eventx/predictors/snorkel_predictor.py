from typing import List, Tuple

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans
from allennlp.predictors import Predictor


@Predictor.register('snorkel-eventx-predictor')
class SnorkelEventxPredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        ner_tags = json_dict['ner_tags']
        trigger_spans, entity_spans = self._extract_trigger_and_entity_spans(ner_tags)
        return self._dataset_reader.text_to_instance(tokens=json_dict['tokens'],
                                                     ner_tags=ner_tags,
                                                     entity_spans=entity_spans,
                                                     trigger_spans=trigger_spans)

    @staticmethod
    def _extract_trigger_and_entity_spans(ner_tags):
        entity_types_with_spans = bio_tags_to_spans(ner_tags)
        triggers_with_spans = [type_span
                               for type_span in entity_types_with_spans
                               if type_span[0].lower() == 'trigger']
        trigger_spans = SnorkelEventxPredictor._extract_exclusive_spans(triggers_with_spans)
        entity_spans = SnorkelEventxPredictor._extract_exclusive_spans(entity_types_with_spans)
        return trigger_spans, entity_spans

    @staticmethod
    def _extract_exclusive_spans(
        types_with_spans: List[Tuple[str, Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        return [(t[1][0], t[1][1] + 1) for t in types_with_spans]
