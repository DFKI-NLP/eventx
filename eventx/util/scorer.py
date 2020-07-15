import copy
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Union
from allennlp.data import Instance


class Result:
    def __init__(self, tp: int = 0, fp: int = 0, fn: int = 0):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def add_counts(self, tp: int = 0, fp: int = 0, fn: int = 0):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def precision(self):
        if self.tp + self.fp > 0:
            return self.tp / (self.tp + self.fp)
        else:
            return 1.0

    def recall(self):
        if self.tp + self.fn > 0:
            return self.tp / (self.tp + self.fn)
        else:
            return 1.0

    def f1(self):
        p = self.precision()
        r = self.recall()
        if p + r > 0:
            return 2 * (p * r) / (p + r)
        else:
            return 0.0

    def get_metrics(self):
        return {
            'precision': self.precision(),
            'recall': self.recall(),
            'f1-score': self.f1(),
            'support': self.tp + self.fn
        }


def entity_equals(a, b):
    if a == b:
        return True
    if 'id' in a and 'id' in b and a['id'] == b['id']:
        return True
    if a['start'] != b['start'] or a['end'] != b['end']:
        return False
    if 'entity_type' in a and 'entity_type' in b and a['entity_type'] != b['entity_type']:
        return False
    # Check text: predicted entity text is roughly reconstructed from tokens
    return True


def get_event_span(event):
    start = event['trigger']['start']
    end = event['trigger']['end']
    for arg in event['arguments']:
        arg_start = arg['start']
        arg_end = arg['end']
        if arg_start < start:
            start = arg_start
        if arg_end > end:
            end = arg_end
    return start, end


def event_equals(pred_event, gold_event, ignore_span=False, ignore_args=False,
                 ignore_optional_args=False):
    if pred_event == gold_event:
        return True
    if pred_event['event_type'] != gold_event['event_type']:
        return False
    if not entity_equals(pred_event['trigger'], gold_event['trigger']):
        return False
    if not ignore_span:
        pred_event_start, pred_event_end = get_event_span(pred_event)
        gold_event_start, gold_event_end = get_event_span(gold_event)
        if pred_event_start != gold_event_start or pred_event_end != gold_event_end:
            return False
    if not ignore_args:
        if len(pred_event['arguments']) != len(gold_event['arguments']):
            return False
        if ignore_optional_args:
            gold_args = [arg for arg in gold_event['arguments'] if arg['role'] == 'location']
        else:
            gold_args = gold_event['arguments']
        for gold_arg in gold_args:
            found_arg = False
            if any(gold_arg['role'] == pred_arg['role'] and entity_equals(gold_arg, pred_arg)
                   for pred_arg in pred_event['arguments']):
                found_arg = True
            if not found_arg:
                return False
    return True


def event_subsumes(subsumed_event, subsuming_event, ignore_span=False, ignore_args=False):
    if subsumed_event == subsuming_event:
        return True
    if subsumed_event['event_type'] != subsuming_event['event_type']:
        return False
    if not entity_equals(subsumed_event['trigger'], subsuming_event['trigger']):
        return False
    if not ignore_span:
        subsumed_event_start, subsumed_event_end = get_event_span(subsumed_event)
        subsuming_event_start, subsuming_event_end = get_event_span(subsuming_event)
        if subsumed_event_start < subsuming_event_start or subsumed_event_end > subsuming_event_end:
            return False
    if not ignore_args:
        if len(subsumed_event['arguments']) > len(subsuming_event['arguments']):
            return False
        # all required roles should be found, if found optional roles have to be correct
        required_subsuming_args = [arg for arg in subsuming_event['arguments']
                                   if arg['role'] == 'location']
        # Check required args
        for required_arg in required_subsuming_args:
            found_arg = False
            if any(subsumed_arg['role'] == required_arg['role'] and
                   entity_equals(subsumed_arg, required_arg) for subsumed_arg in
                   subsumed_event['arguments']):
                found_arg = True
            if not found_arg:
                return False

        # Check for falsely labeled args in subsumed (predicted)
        for subsumed_arg in subsumed_event['arguments']:
            found_arg = False
            if any(subsuming_arg['role'] == subsumed_arg['role'] and
                   entity_equals(subsuming_arg, subsumed_arg) for subsuming_arg in
                   subsuming_event['arguments']):
                found_arg = True
            if not found_arg:
                return False
    return True


def event_scorer(pred_events, gold_events, ignore_args=False, ignore_span=False,
                 allow_subsumption=False, keep_event_matches=False,
                 ignore_optional_args=False) -> Tuple[int, int, int]:
    """
    Counts true positives, false positives and false negatives.

    Parameters
    ----------
    pred_events: Predicted events
    gold_events: Gold events
    ignore_args: Ignore event arguments during comparison
    ignore_span: Ignore event span during comparison
    ignore_optional_args: Only look at required arguments for comparison, i.e. location arg
    allow_subsumption: Allows for a gold event to subsume a predicted event and vice versa
    keep_event_matches: Keeps predicted events that were matched with gold events

    Returns
    -------
    TP, FP, FN
    """
    pred_events_copy = copy.deepcopy(pred_events) if pred_events else []
    gold_events_copy = copy.deepcopy(gold_events) if gold_events else []
    # sort event lists by event span start
    pred_events_copy.sort(key=lambda event: get_event_span(event)[0])
    gold_events_copy.sort(key=lambda event: get_event_span(event)[0])
    tp = 0
    fn = 0

    pred_events_no_match = np.ones(len(pred_events_copy))

    for gold_event in gold_events_copy:
        found_idx = -1
        for idx, pred_event in enumerate(pred_events_copy):
            if event_equals(pred_event, gold_event,
                            ignore_args=ignore_args, ignore_span=ignore_span,
                            ignore_optional_args=ignore_optional_args) or \
                (allow_subsumption and
                 event_subsumes(subsumed_event=pred_event, subsuming_event=gold_event,
                                ignore_args=ignore_args, ignore_span=ignore_span)):
                tp += 1
                found_idx = idx
        if found_idx < 0:
            fn += 1
        else:
            # pred_event might match multiple gold_events because
            # Eventx model does not distinguish between events sharing the same trigger
            if keep_event_matches:
                pred_events_no_match[found_idx] *= 0
            else:
                del pred_events_copy[found_idx]
    fp = int(pred_events_no_match.sum()) if keep_event_matches else len(pred_events_copy)
    return tp, fp, fn


def event_by_class_scorer(pred_events, gold_events, ignore_args=False, ignore_span=False,
                          allow_subsumption=False, keep_event_matches=False,
                          ignore_optional_args=False) -> Dict[str, Result]:
    results: Dict[str, Result] = {}
    event_types = list(set([event['event_type'] for event in pred_events] +
                           [event['event_type'] for event in gold_events]))
    for event_type in event_types:
        class_pred_events = [event for event in pred_events if event['event_type'] == event_type]
        class_gold_events = [event for event in gold_events if event['event_type'] == event_type]
        tp, fp, fn = event_scorer(
            class_pred_events, class_gold_events,
            ignore_args=ignore_args, ignore_span=ignore_span,
            allow_subsumption=allow_subsumption, keep_event_matches=keep_event_matches,
            ignore_optional_args=ignore_optional_args)
        results[event_type] = Result(tp, fp, fn)
    return results


def score_document(pred_doc, gold_doc, ignore_args=False, ignore_span=False,
                   allow_subsumption=False, keep_event_matches=False,
                   ignore_optional_args=False):
    results = event_by_class_scorer(pred_doc['events'], gold_doc['events'],
                                    ignore_args=ignore_args, ignore_span=ignore_span,
                                    allow_subsumption=allow_subsumption,
                                    keep_event_matches=keep_event_matches,
                                    ignore_optional_args=ignore_optional_args)
    return results


def score_documents(pred_events, gold_events,
                    ignore_args=False, ignore_span=False,
                    allow_subsumption=False, keep_event_matches=False,
                    ignore_optional_args=False):
    results: Dict[str, Result] = {'micro avg': Result()}
    for pred_doc_events, gold_doc_events in zip(pred_events,
                                                list(gold_events)):
        doc_results = event_by_class_scorer(pred_doc_events, gold_doc_events,
                                            ignore_args=ignore_args, ignore_span=ignore_span,
                                            allow_subsumption=allow_subsumption,
                                            keep_event_matches=keep_event_matches,
                                            ignore_optional_args=ignore_optional_args)
        for event_type, result in doc_results.items():
            results['micro avg'].add_counts(result.tp, result.fp, result.fn)
            if event_type in results:
                results[event_type].add_counts(result.tp, result.fp, result.fn)
            else:
                results[event_type] = result

    classification_report = {}
    for event_type, result in results.items():
        classification_report[event_type] = result.get_metrics()
    return classification_report


def score_files(pred_file_path, gold_file_path, ignore_args=False, ignore_span=False,
                allow_subsumption=False, keep_event_matches=False,
                ignore_optional_args=False):
    pred_file = pd.read_json(pred_file_path, lines=True, encoding='utf8')
    gold_file = pd.read_json(gold_file_path, lines=True, encoding='utf8')

    pred_events = pred_file['events']
    gold_events = gold_file['events']
    return score_documents(pred_events, gold_events, ignore_args=ignore_args,
                           ignore_span=ignore_span,
                           allow_subsumption=allow_subsumption,
                           keep_event_matches=keep_event_matches,
                           ignore_optional_args=ignore_optional_args)


def get_triggers(documents: List[Union[Dict, Instance]]):
    """
    Retrieves triggers from list of documents

    Parameters
    ----------
    documents: List of documents containing events

    Returns
    -------
    List of tuples for each trigger containing document index, trigger start, trigger end and
    trigger type

    """
    triggers = []
    for doc_idx, doc in enumerate(documents):
        for event in doc['events']:
            trigger = event['trigger']
            triggers.append((doc_idx, trigger['start'], trigger['end'], event['event_type']))
    return triggers


def get_arguments(documents: List[Union[Dict, Instance]]):
    """
    Retrieves arguments from list of documents

    Parameters
    ----------
    documents: List of documents containing events

    Returns
    -------
    List of tuples for each argument containing document index, trigger start, trigger end,
    trigger type, argument start, argument end, argument role

    """
    arguments = []
    for doc_idx, doc in enumerate(documents):
        for event in doc['events']:
            trigger = event['trigger']
            for arg in event['arguments']:
                arguments.append((doc_idx,
                                  trigger['start'], trigger['end'], event['event_type'],
                                  arg['start'], arg['end'], arg['role']))
    return arguments


def calc_metric(y_true, y_pred):
    # Copied from:
    # https://github.com/nlpcl-lab/bert-event-extraction/blob/c35caea08269d6143cb988366c91a664b60b4106/utils.py#L20
    y_true_set = set(y_true)  # Remove duplicates due to events sharing the same trigger
    num_predicted = len(y_pred)  # TP + FP
    num_gold = len(y_true_set)  # TP + FN
    num_correct = 0  # TP

    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    precision = num_correct / num_predicted if num_predicted > 0 else 1.0
    recall = num_correct / num_gold if num_gold > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    support = len(y_true_set)

    return {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': support
    }


def get_metrics_by_class(y_true: List[Tuple], y_pred: List[Tuple]):
    """
    Assumes the label to be the last element of the tuple.

    Parameters
    ----------
    y_true: List with examples with gold labels
    y_pred: List with examples with predicted labels

    Returns
    -------
    Dictionary with precision, recall, f1 for every class

    """
    classes = [item[-1] for item in y_true]
    classes += [item[-1] for item in y_pred]
    classes = set(classes)

    metrics_by_class = {}

    for clazz in classes:
        filtered_y_true = [item for item in y_true if item[-1] == clazz]
        filtered_y_pred = [item for item in y_pred if item[-1] == clazz]
        metrics_by_class[clazz] = calc_metric(filtered_y_true, filtered_y_pred)

    return metrics_by_class


def get_trigger_identification_metrics(gold_triggers: List[Tuple], pred_triggers: List[Tuple]):
    """
    Gets metrics for trigger identification. A trigger is identified correctly
    according to Ji and Grishman, 2008 (https://www.aclweb.org/anthology/P08-1030.pdf)
    if its spans match any of the reference triggers.

    Parameters
    ----------
    gold_triggers: Gold triggers with document idx, trigger spans & type
    pred_triggers: Predicted triggers with document idx, trigger spans & type

    Returns
    -------
    Precision, recall, f1, support for trigger identification

    """
    gold_trigger_spans = [(trigger[0], trigger[1], trigger[2]) for trigger in gold_triggers]
    pred_trigger_spans = [(trigger[0], trigger[1], trigger[2]) for trigger in pred_triggers]
    return calc_metric(gold_trigger_spans, pred_trigger_spans)


def get_trigger_classification_metrics(gold_triggers: List[Tuple], pred_triggers: List[Tuple]):
    """
    Gets metrics for trigger classification. A trigger is correct
    according to Ji and Grishman, 2008 (https://www.aclweb.org/anthology/P08-1030.pdf)
    if its event subtype and offsets match those of a reference trigger.

    Parameters
    ----------
    gold_triggers: Gold triggers with document idx, trigger spans & type
    pred_triggers: Predicted triggers with document idx, trigger spans & type

    Returns
    -------
    Precision, recall, f1, support for trigger classification

    """
    micro_avg = calc_metric(gold_triggers, pred_triggers)
    classification_report = get_metrics_by_class(gold_triggers, pred_triggers)
    classification_report['micro avg'] = micro_avg
    return classification_report


def get_argument_identification_metrics(gold_arguments: List[Tuple], pred_arguments: List[Tuple]):
    """
    Gets metrics for argument identification. An argument is identified correctly
    according to Ji and Grishman, 2008 (https://www.aclweb.org/anthology/P08-1030.pdf)
    if its event subtype and offsets match those of any of the reference argument mentions.

    Parameters
    ----------
    gold_arguments: Gold arguments with document idx, trigger spans & type, arg spans & type
    pred_arguments: Predicted arguments with document idx, trigger spans & type, arg spans & type

    Returns
    -------
    Precision, recall, f1, support for argument identification

    """
    gold_argument_spans = [(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5])
                           for arg in gold_arguments]
    pred_argument_spans = [(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5])
                           for arg in pred_arguments]
    return calc_metric(gold_argument_spans, pred_argument_spans)


def get_argument_classification_metrics(gold_arguments: List[Tuple], pred_arguments: List[Tuple]):
    """
    Gets metrics for argument classification. An argument is classified correctly
    according to Ji and Grishman, 2008 (https://www.aclweb.org/anthology/P08-1030.pdf)
    if its event subtype, offsets and argument role match those of any of the
    reference argument mentions.

    Parameters
    ----------
    gold_arguments: Gold arguments with document idx, trigger spans & type, arg spans & type
    pred_arguments: Predicted arguments with document idx, trigger spans & type, arg spans & type

    Returns
    -------
    Precision, recall, f1, support for argument classification

    """
    micro_avg = calc_metric(gold_arguments, pred_arguments)
    classification_report = get_metrics_by_class(gold_arguments, pred_arguments)
    classification_report['micro avg'] = micro_avg
    return classification_report
