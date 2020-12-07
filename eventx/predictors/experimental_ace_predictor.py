from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from eventx import NEGATIVE_ENTITY_LABEL


@Predictor.register('experimental-ace-eventx-predictor')
class ExperimentalAcePredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        words = json_dict['words']
        entities = json_dict['golden-entity-mentions']

        entity_spans = []
        ner_tags = [NEGATIVE_ENTITY_LABEL] * len(words)

        for entity in entities:
            entity_span = (entity['start'], entity['end'])
            entity_spans.append(entity_span)
            entity_label = entity['entity-type']
            for idx in range(entity['start'], entity['end']):
                # Encode triggers with IOB2 encoding scheme
                if idx == entity['start']:
                    ner_tags[idx] = 'B-' + entity_label
                else:
                    ner_tags[idx] = 'I-' + entity_label
        return self._dataset_reader.text_to_instance(tokens=words,
                                                     ner_tags=ner_tags,
                                                     entity_spans=entity_spans)
