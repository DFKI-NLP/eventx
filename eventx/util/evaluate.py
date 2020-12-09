import argparse
import json
import io
import logging
import copy
import traceback
from typing import List

import numpy as np
import pandas as pd
from pathlib import Path

from allennlp.common import JsonDict
from eventx.predictors.predictor_utils import load_predictor
from eventx.models.model_utils import batched_predict_json
from eventx.util import scorer
from eventx import SD4M_RELATION_TYPES, ROLE_LABELS, SDW_RELATION_TYPES, SDW_ROLE_LABELS,\
    NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL

logger = logging.getLogger('eventx')
logger.setLevel(level=logging.INFO)


def load_test_data(input_path) -> List[JsonDict]:
    """
    Filters out documents that do not contain any triggers.
    Parameters
    ----------
    input_path: Path to test file

    Returns
    -------
    Filtered document list
    """
    docs_without_triggers = 0
    docs_with_triggers = 0
    test_documents = []
    with io.open(input_path, encoding='utf8') as test_file:
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
    Construct label arrays in order to use sklearn.metrics for confusion matrices etc.
    Keep in mind that these are not in compliance with the correctness criteria by
    Ji and Grishman, 2008 (https://www.aclweb.org/anthology/P08-1030.pdf) as we pay no attention
    to the correctness of the trigger label when constructing the argument role label array.

    Parameters
    ----------
    gold_docs: Documents with gold event annotation
    predicted_docs: Documents with predicted events

    Returns
    -------
    Label arrays for trigger & argument role classification
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

        for event in pred_events:
            trigger_span = event['trigger']['start'], event['trigger']['end']
            trigger_idx = trigger_spans.index(trigger_span)
            for argument in event['arguments']:
                entity_idx = entity_spans.index((argument['start'], argument['end']))
                doc_pred_arg_role_labels[trigger_idx][entity_idx] = argument['role']

        for event in events:
            trigger_span = event['trigger']['start'], event['trigger']['end']
            trigger_idx = trigger_spans.index(trigger_span)
            for argument in event['arguments']:
                # Gold role label
                entity_idx = entity_spans.index((argument['start'], argument['end']))
                # Set positive event argument roles overwriting the default
                doc_arg_role_labels[trigger_idx][entity_idx] = argument['role']

        doc_arg_role_labels = [role_label for args in doc_arg_role_labels for role_label in args]
        doc_pred_arg_role_labels = [role_label for args in doc_pred_arg_role_labels
                                    for role_label in args]
        arg_role_labels += doc_arg_role_labels
        pred_arg_role_labels += doc_pred_arg_role_labels

    return {
        "trigger_y_true": np.asarray(trigger_labels),
        "trigger_y_pred": np.asarray(pred_trigger_labels),
        "arg_y_true": np.asarray(arg_role_labels),
        "arg_y_pred": np.asarray(pred_arg_role_labels)
    }


def summize_multiple_runs(model_paths, test_docs, remove_duplicates=True,
                          predictor_name="snorkel-eventx-predictor", events_key="events",
                          include_class_metrics=True, relation_types=SD4M_RELATION_TYPES,
                          role_classes=ROLE_LABELS):
    trigger_id_metrics = []
    trigger_class_metrics = []
    argument_id_metrics = []
    argument_class_metrics = []

    gold_triggers = scorer.get_triggers(test_docs, events_key)
    gold_arguments = scorer.get_arguments(test_docs, events_key)
    if remove_duplicates:
        # Remove duplicates that are due to events sharing the same trigger
        gold_triggers = list(set(gold_triggers))
        gold_arguments = list(set(gold_arguments))
    logger.info(f'Collecting metrics for the {len(model_paths)} models')
    for idx, model_path in enumerate(model_paths):
        logger.info(f'Working on {idx+1}. model.')
        try:
            predictor = load_predictor(model_dir=model_path, predictor_name=predictor_name)
            predicted_docs = batched_predict_json(predictor=predictor, examples=test_docs)

            predicted_triggers = scorer.get_triggers(predicted_docs)
            predicted_arguments = scorer.get_arguments(predicted_docs)

            trigger_id_metrics.append(
                scorer.get_trigger_identification_metrics(
                    gold_triggers, predicted_triggers)
            )
            trigger_class_metrics.append(
                scorer.get_trigger_classification_metrics(
                    gold_triggers, predicted_triggers,
                    include_class_metrics=include_class_metrics, labels=relation_types
                )
            )
            argument_id_metrics.append(
                scorer.get_argument_identification_metrics(gold_arguments, predicted_arguments)
            )
            argument_class_metrics.append(
                scorer.get_argument_classification_metrics(
                    gold_arguments, predicted_arguments,
                    include_class_metrics=include_class_metrics, labels=role_classes)
            )
        except KeyError as err:
            logger.warning(f'The model {model_path} encountered an error ({err}). '
                           f'Skip and continue.')
            logger.warning(traceback.format_exc())
    trigger_id_metrics = pd.DataFrame(trigger_id_metrics)
    trigger_class_metrics = pd.DataFrame(trigger_class_metrics)
    argument_id_metrics = pd.DataFrame(argument_id_metrics)
    argument_class_metrics = pd.DataFrame(argument_class_metrics)

    trigger_metrics = [('Trigger identification',
                        collect_values('Trigger identification', trigger_id_metrics)),
                       ('Trigger classification',
                        collect_values('Trigger classification', trigger_class_metrics))]
    if include_class_metrics:
        trigger_metrics += [
            (trigger_label, collect_values(trigger_label, trigger_class_metrics))
            for trigger_label in relation_types[:-1]]
    trigger_metrics = dict(trigger_metrics)

    argument_metrics = [('Argument identification',
                         collect_values('Argument identification', argument_id_metrics)),
                        ('Argument classification',
                         collect_values('Argument classification', argument_class_metrics))]
    if include_class_metrics:
        argument_metrics += [
            (role_label, collect_values(role_label, argument_class_metrics))
            for role_label in role_classes[:-1]]
    argument_metrics = dict(argument_metrics)

    return trigger_metrics, argument_metrics


def get_median_std(metrics_dictionary):
    """
    Calculates median and standard deviation across repeated runs
    Parameters
    ----------
    metrics_dictionary

    Returns
    -------

    """
    metrics_dictionary_copy = copy.deepcopy(metrics_dictionary)
    for label, label_metrics in metrics_dictionary_copy.items():
        for metric in ['precision', 'recall', 'f1-score']:
            values = np.asarray(label_metrics[metric])
            median = np.median(values)
            std = values.std()
            metrics_dictionary_copy[label][metric] = [median, std]
    return metrics_dictionary_copy


def get_mean_std(metrics_dictionary):
    """
    Calculates mean and standard deviation across repeated runs
    Parameters
    ----------
    metrics_dictionary

    Returns
    -------

    """
    metrics_dictionary_copy = copy.deepcopy(metrics_dictionary)
    for label, label_metrics in metrics_dictionary_copy.items():
        for metric in ['precision', 'recall', 'f1-score']:
            values = np.asarray(label_metrics[metric])
            mean = values.mean()
            std = values.std()
            metrics_dictionary_copy[label][metric] = [mean, std]
    return metrics_dictionary_copy


def collect_values(label, data_frame):
    metric_values = {}
    label_column = data_frame[label]
    for metric in ['precision', 'recall', 'f1-score']:
        values = np.asarray([row[metric] for row in label_column])
        metric_values[metric] = values
    metric_values['support'] = label_column[0]['support']
    return metric_values


def format_classification_report(classification_report, make_string=False):
    """
    Helper to format the classification report in a way that is easier to export as csv
    Assumes either list of mean/median and standard deviation for each metric or single value
    for each metric.
    Parameters
    ----------
    make_string: Median/Mean with standard deviation in one string
    classification_report

    Returns
    -------

    """
    rows = []
    for k, v in classification_report.items():
        row = {'row_name': k}
        mean_row = {'row_name': k}
        std_row = {'row_name': k}
        if all(isinstance(v[metric], list) for metric in ['precision', 'recall', 'f1-score']):
            for metric in ['precision', 'recall', 'f1-score']:
                mean_row[metric] = '{:.{prec}f}'.format(v[metric][0] * 100, prec=1)
                std_row[metric] = '+/- {:.{prec}f}'.format(v[metric][1] * 100, prec=1)
                if make_string:
                    mean_row[metric] = mean_row[metric] + std_row[metric]
            mean_row['support'] = v['support']
            std_row['support'] = v['support']
            rows.append(mean_row)
            if not make_string:
                rows.append(std_row)
        else:
            for metric in ['precision', 'recall', 'f1-score']:
                row[metric] = '{:.{prec}f}'.format(v[metric] * 100, prec=1)
            row['support'] = v['support']
            rows.append(row)

    return pd.DataFrame(rows)


def evaluate_random_repeats(models_base_path, test_data_path, output_path, runs=5,
                            configs=None):
    models_base_path = Path(models_base_path)
    test_docs = load_test_data(test_data_path)
    output_path = Path(output_path)
    if configs is None:
        configs = ['snorkel_bert_gold', 'snorkel_bert_daystream', 'snorkel_bert_merged']
    for config in configs:
        model_paths = [models_base_path.joinpath(f'run_{run + 1}/{config}') for run in range(runs)]
        trigger_metrics, argument_metrics = summize_multiple_runs(model_paths, test_docs)
        trigger_metrics = get_mean_std(trigger_metrics)
        argument_metrics = get_mean_std(argument_metrics)
        formatted_trigger = format_classification_report(trigger_metrics)
        formatted_argument = format_classification_report(argument_metrics)
        formatted_metrics = pd.concat([formatted_trigger, formatted_argument])
        formatted_metrics.set_index('row_name', inplace=True)
        formatted_metrics.to_csv(output_path.joinpath(f'{config}_results.csv'))


def evaluate_increasing_train_data(models_base_path, test_data_path, output_path, runs=5,
                                   configs=None):
    models_base_path = Path(models_base_path)
    test_docs = load_test_data(test_data_path)
    output_path = Path(output_path)
    if configs is None:
        configs = [f'daystream{percentage}_snorkeled' for percentage in range(50, 101, 10)]
    for config in configs:
        model_paths = [models_base_path.joinpath(f'samples_{run + 1}/{config}')
                       for run in range(runs)]
        trigger_metrics, argument_metrics = summize_multiple_runs(model_paths, test_docs)
        trigger_metrics = get_mean_std(trigger_metrics)
        argument_metrics = get_mean_std(argument_metrics)
        formatted_trigger = format_classification_report(trigger_metrics)
        formatted_argument = format_classification_report(argument_metrics)
        formatted_metrics = pd.concat([formatted_trigger, formatted_argument])
        formatted_metrics.set_index('row_name', inplace=True)
        formatted_metrics.to_csv(output_path.joinpath(f'{config}_results.csv'))


def main(args):
    input_path = Path(args.input_path)
    assert input_path.exists(), 'Input not found: %s'.format(args.input_path)
    # output_path = Path(args.output_path)
    model_path = Path(args.model_path)
    assert model_path.exists(), 'Input not found: %s'.format(args.model_path)

    test_docs = load_test_data(input_path)
    gold_triggers = scorer.get_triggers(test_docs)
    gold_arguments = scorer.get_arguments(test_docs)

    if not args.keep_duplicates:
        # Remove duplicates that are due to events sharing the same trigger
        gold_triggers = list(set(gold_triggers))
        gold_arguments = list(set(gold_arguments))

    # Constructs instance only using tokens and ner tags
    predictor = load_predictor(model_dir=model_path, predictor_name=args.predictor_name)
    predicted_docs = batched_predict_json(predictor=predictor, examples=test_docs)
    predicted_triggers = scorer.get_triggers(predicted_docs)
    predicted_arguments = scorer.get_arguments(predicted_docs)

    scorer.get_trigger_identification_metrics(
        gold_triggers, predicted_triggers, output_string=True)
    scorer.get_trigger_classification_metrics(
        gold_triggers, predicted_triggers, output_string=True)
    scorer.get_argument_identification_metrics(
        gold_arguments, predicted_arguments, output_string=True)
    scorer.get_argument_classification_metrics(
        gold_arguments, predicted_arguments, output_string=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Corpus statistics')
    parser.add_argument('--input_path', type=str, help='Path to test file')
    # parser.add_argument('--output_path', type=str, help='Path to output file')
    parser.add_argument('--model_path', type=str, help='Path to model')
    parser.add_argument('--predictor_name', type=str, default='snorkel-eventx-predictor',
                        help='Name of the predictor class')
    parser.add_argument('--keep_duplicates', default=False, action='store_true',
                        help='Whether to keep duplicates caused by events sharing the same trigger')
    arguments = parser.parse_args()
    main(arguments)
