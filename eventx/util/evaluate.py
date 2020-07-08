import argparse
import json
import io
import logging
import copy
import sklearn
from typing import List

import numpy as np
from pathlib import Path

from allennlp.common import JsonDict
from eventx.predictors.predictor_utils import load_predictor
from eventx.models.model_utils import batched_predict_json
from eventx import SD4M_RELATION_TYPES, ROLE_LABELS, NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

PREDICTOR_NAME = "snorkel-eventx-predictor"


def load_test_data(input_path) -> List[JsonDict]:
    docs_without_triggers = 0
    docs_with_triggers = 0
    test_documents = []
    with io.open(input_path) as test_file:
        for line in test_file.readlines():
            example = json.loads(line)
            if any(e['entity_type'].lower() == 'trigger' for e in example['entities']):
                test_documents.append(example)
                docs_with_triggers += 1
            else:
                logging.debug(f"Document {example['id']} does not contain triggers and is "
                              f"therefore not supported.")
                docs_without_triggers += 1
    logging.debug(f"Keeping {docs_with_triggers}/{docs_with_triggers + docs_without_triggers} "
                  f"for evaluation")
    return test_documents


def get_label_arrays(gold_docs, predicted_docs):
    """
    Construct label arrays in order to use sklearn.metrics

    Parameters
    ----------
    gold_docs
    predicted_docs

    Returns
    -------
    Label arrays for trigger identification, classification, argument identification, classification
    """
    trigger_labels = []
    pred_trigger_labels = []
    arg_role_labels = []
    pred_arg_role_labels = []
    for gold_doc, pred_doc in zip(gold_docs, predicted_docs):
        entities = gold_doc['entities']
        events = gold_doc['events']
        entity_spans = [(e['start'], e['end']) for e in entities]
        triggers = [e for e in entities if e['entity_type'].lower() == 'trigger']
        trigger_spans = [(t['start'], t['end']) for t in triggers]

        # Extract gold & predicted event trigger labels
        span_to_label_pairs = [((event['trigger']['start'], event['trigger']['end']),
                                event['event_type'])
                               for event in events]
        trigger_span_to_label = dict(span_to_label_pairs)

        pred_events = pred_doc['events']
        pred_span_to_label_pairs = [((event['trigger']['start'], event['trigger']['end']),
                                     event['event_type'])
                                    for event in pred_events]
        pred_trigger_span_to_label = dict(pred_span_to_label_pairs)

        for trigger_span in trigger_spans:
            # Gold label
            if trigger_span in trigger_span_to_label:
                trigger_label = trigger_span_to_label[trigger_span]
            else:
                trigger_label = NEGATIVE_TRIGGER_LABEL
            trigger_labels.append(trigger_label)
            # Predicted label
            if trigger_span in pred_trigger_span_to_label:
                trigger_label = pred_trigger_span_to_label[trigger_span]
            else:
                trigger_label = NEGATIVE_TRIGGER_LABEL
            pred_trigger_labels.append(trigger_label)

        # Extract gold & predicted argument role labels
        doc_arg_role_labels = [[NEGATIVE_ARGUMENT_LABEL for _ in range(len(entity_spans))]
                               for _ in range(len(trigger_spans))]
        doc_pred_arg_role_labels = copy.deepcopy(doc_arg_role_labels)

        # Construct span->role_label map for predicted events
        pred_span_to_role_label = {}
        for event in pred_events:
            trigger_start = event['trigger']['start']
            trigger_end = event['trigger']['end']
            trigger_label = event['event_type']
            for argument in event['arguments']:
                key_tuple = trigger_start, trigger_end, trigger_label, \
                            argument['start'], argument['end']
                pred_span_to_role_label[key_tuple] = argument['role']

        for event in events:
            trigger_span = event['trigger']['start'], event['trigger']['end']
            trigger_idx = trigger_spans.index(trigger_span)
            for argument in event['arguments']:
                # Gold role label
                entity_idx = entity_spans.index((argument['start'], argument['end']))
                # Set positive event argument roles overwriting the default
                doc_arg_role_labels[trigger_idx][entity_idx] = argument['role']
                # Predicted role label
                key_tuple = event['trigger']['start'], event['trigger']['end'], \
                    event['event_type'], argument['start'], argument['end']
                if key_tuple in pred_span_to_role_label:
                    doc_pred_arg_role_labels[trigger_idx][entity_idx] = \
                        pred_span_to_role_label[key_tuple]

        doc_arg_role_labels = [role_label for args in doc_arg_role_labels for role_label in args]
        doc_pred_arg_role_labels = [role_label for args in doc_pred_arg_role_labels
                                    for role_label in args]
        arg_role_labels += doc_arg_role_labels
        pred_arg_role_labels += doc_pred_arg_role_labels

    trigger_id_labels = [label if label == NEGATIVE_TRIGGER_LABEL else 'Event Trigger'
                         for label in trigger_labels]
    pred_trigger_id_labels = [label if label == NEGATIVE_TRIGGER_LABEL else 'Event Trigger'
                              for label in pred_trigger_labels]
    arg_role_id_labels = [label if label == NEGATIVE_ARGUMENT_LABEL else 'Event Argument'
                          for label in arg_role_labels]
    pred_arg_role_id_labels = [label if label == NEGATIVE_ARGUMENT_LABEL else 'Event Argument'
                               for label in pred_arg_role_labels]

    return {
        "trigger_id_y_true": np.asarray(trigger_id_labels),
        "trigger_id_y_pred": np.asarray(pred_trigger_id_labels),
        "trigger_class_y_true": np.asarray(trigger_labels),
        "trigger_class_y_pred": np.asarray(pred_trigger_labels),
        "arg_role_id_y_true": np.asarray(arg_role_id_labels),
        "arg_role_id_y_pred": np.asarray(pred_arg_role_id_labels),
        "arg_role_class_y_true": np.asarray(arg_role_labels),
        "arg_role_class_y_pred": np.asarray(pred_arg_role_labels)
    }


def main(args):
    input_path = Path(args.input_path)
    assert input_path.exists(), 'Input not found: %s'.format(args.input_path)
    # output_path = Path(args.output_path)
    model_path = Path(args.model_path)
    assert model_path.exists(), 'Input not found: %s'.format(args.model_path)

    test_docs = load_test_data(input_path)

    # Constructs instance only using tokens and ner tags
    predictor = load_predictor(model_dir=model_path, predictor_name=PREDICTOR_NAME)
    predicted_docs = batched_predict_json(predictor=predictor, examples=test_docs)
    label_arrays = get_label_arrays(test_docs, predicted_docs)
    print('Trigger identification')
    print(sklearn.metrics.classification_report(y_true=label_arrays['trigger_id_y_true'],
                                                y_pred=label_arrays['trigger_id_y_pred']))
    # TODO find a way to take out the negative classes for the averaging of the scores
    print('Trigger classification')
    print(sklearn.metrics.classification_report(y_true=label_arrays['trigger_class_y_true'],
                                                y_pred=label_arrays['trigger_class_y_pred'],
                                                labels=SD4M_RELATION_TYPES[:-1]))
    print('Argument identification')
    print(sklearn.metrics.classification_report(y_true=label_arrays['arg_role_id_y_true'],
                                                y_pred=label_arrays['arg_role_id_y_pred']))
    print('Role classification')
    print(sklearn.metrics.classification_report(y_true=label_arrays['arg_role_class_y_true'],
                                                y_pred=label_arrays['arg_role_class_y_pred'],
                                                labels=ROLE_LABELS[:-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Corpus statistics')
    parser.add_argument('--input_path', type=str, help='Path to test file')
    # parser.add_argument('--output_path', type=str, help='Path to output file')
    parser.add_argument('--model_path', type=str, help='Path to model')
    arguments = parser.parse_args()
    main(arguments)
