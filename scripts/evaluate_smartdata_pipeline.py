import argparse
import json

from allennlp.predictors import Predictor
from sklearn.metrics import classification_report

import eventx  # Register classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str,
                        help="A jsonl file in the smartdata format")
    parser.add_argument("--ner-model", type=str, required=True,
                        help="Path to a trained ner model")
    parser.add_argument("--eventx-model", type=str, required=True,
                        help="Path to a trained eventx model")
    args = parser.parse_args()

    print('Loading ner model')
    ner_predictor = Predictor.from_path(args.ner_model,
                                        predictor_name='sentence-tagger')
    print('Loading eventx model')
    eventx_predictor = Predictor.from_path(args.eventx_model,
                                           predictor_name='smartdata-eventx-predictor')

    all_gold_ner_tags = []
    all_predicted_ner_tags = []

    with open(args.input_file) as f:
        for line in f.readlines():
            example = json.loads(line)

            gold_ner_tags = example['ner_tags']
            all_gold_ner_tags.extend(gold_ner_tags)

            tokens = example['tokens']
            instance = ner_predictor._dataset_reader.text_to_instance(tokens)
            predicted_ner_tags = ner_predictor.predict_instance(instance)['tags']
            all_predicted_ner_tags.extend(predicted_ner_tags)

            if _tags_include_trigger(gold_ner_tags):
                predicted_events = []
                if _tags_include_trigger(predicted_ner_tags):
                    # Extend the input by NER tags and extract events
                    eventx_prediction = eventx_predictor.predict_json({
                        'tokens': tokens,
                        'ner_tags': predicted_ner_tags
                    })
                    predicted_events = eventx_prediction['events']

                if len(example['events']+predicted_events) > 0:
                    print('*' * 80)
                    print(example['text'])
                    print('Gold:')
                    for event in example['events']:
                        event_string = f'  {event["event_type"]}: {event["trigger"]["text"]} (trigger)'
                        for arg in event['arguments']:
                            event_string += f', {arg["text"]} ({arg["role"]})'
                        print(event_string)
                    print('Prediction:')
                    for event in predicted_events:
                        event_string = f'  {event["event_type"]}: {event["trigger"]["text"]} (trigger)'
                        for arg in event['arguments']:
                            event_string += f', {arg["text"]} ({arg["role"]})'
                        print(event_string)
                    print('*' * 80)

    positive_gold_labels = set(all_gold_ner_tags)
    positive_gold_labels.remove('O')
    print(classification_report(y_true=all_gold_ner_tags,
                                y_pred=all_predicted_ner_tags,
                                labels=list(positive_gold_labels)))


def _tags_include_trigger(tags):
    return any([tag != eventx.NEGATIVE_TRIGGER_LABEL and tag.lower()[2:] == 'trigger'
                for tag in tags])


if __name__ == '__main__':
    main()
