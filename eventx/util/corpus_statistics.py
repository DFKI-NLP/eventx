import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from eventx import SD4M_RELATION_TYPES, ROLE_LABELS, NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL
from eventx.util import scorer


def is_rss(doc_id: str):
    return doc_id.startswith("http")


def is_twitter(doc_id: str):
    return doc_id[0].isdigit()


def get_docs_tokens_entities_triggers(dataset: pd.DataFrame):
    """
    Retrieves numbers about the Smartdata/Daystream dataset

    Parameters
    ----------
    dataset

    Returns
    -------
    Counts for documents, tokens, entities, triggers
    """
    num_of_docs = len(dataset)
    num_of_tokens = 0
    for doc_tokens in dataset['tokens']:
        num_of_tokens += len(doc_tokens)
    num_of_entities = 0
    num_of_triggers = 0
    for doc_entities in dataset['entities']:
        num_of_entities += len(doc_entities)
        doc_triggers = [entity for entity in doc_entities if entity['entity_type'] == 'trigger']
        num_of_triggers += len(doc_triggers)
    return {
        "# Docs": num_of_docs,
        "# Tokens": num_of_tokens,
        "# Entities": num_of_entities,
        "# Triggers": num_of_triggers
    }


def get_docs_tokens_entities_events(dataset: pd.DataFrame):
    """
    Retrieves numbers about the dataset for ACE

    Parameters
    ----------
    dataset

    Returns
    -------
    Counts for documents, tokens, entities, triggers
    """
    num_of_docs = len(dataset)
    num_of_tokens = 0
    for doc_tokens in dataset['words']:
        num_of_tokens += len(doc_tokens)
    num_of_entities = 0
    num_of_triggers = 0
    num_of_arguments = 0
    num_of_unique_triggers = 0
    for doc_entities in dataset['golden-entity-mentions']:
        num_of_entities += len(doc_entities)
    for doc_events in dataset['golden-event-mentions']:
        num_of_triggers += len(doc_events)
        doc_triggers = [(event['trigger']['start'], event['trigger']['end'])
                        for event in doc_events]
        num_of_unique_triggers += len(set(doc_triggers))
        doc_args = [len(event['arguments']) for event in doc_events]
        num_of_arguments += sum(doc_args)
    return {
        "# Sentences": num_of_docs,
        "# Tokens": num_of_tokens,
        "# Entities": num_of_entities,
        "# Triggers": num_of_triggers,
        "# Unique Triggers": num_of_unique_triggers,
        "# Arguments": num_of_arguments
    }


def has_triggers(doc):
    """
    Parameters
    ----------
    doc: Document

    Returns
    -------
    Whether the document contains any triggers
    """
    entities = doc['entities']
    return any(entity['entity_type'] == 'trigger' for entity in entities)


def has_events(doc, include_negatives=False):
    """
    Parameters
    ----------
    doc: Document
    include_negatives: Count document as having events when at least one trigger is not an abstain

    Returns
    -------
    Whether the document contains any (positive) events
    """
    if 'events' in doc and doc['events']:
        return True
    elif 'golden-event-mentions' in doc and doc['golden-event-mentions']:
        return True
    elif 'event_triggers' in doc and doc['event_triggers']:
        trigger_probs = np.asarray(
            [trigger['event_type_probs'] for trigger in doc['event_triggers']]
        )
        if include_negatives:
            return trigger_probs.sum() > 0.0
        labeled_triggers = trigger_probs.sum(axis=1) > 0.0
        trigger_labels = trigger_probs[labeled_triggers].argmax(axis=1)
        if any(label < len(SD4M_RELATION_TYPES)-1 for label in trigger_labels):
            return True
    return False


def has_multiple_events(doc, include_negatives=False):
    """
    Parameters
    ----------
    doc: Document
    include_negatives: Count document as having events when at least one trigger is not an abstain

    Returns
    -------
    Whether the document contains multiple (positive) events
    """
    if 'events' in doc and len(doc['events']) > 1:
        return True
    elif 'golden-event-mentions' in doc and len(doc['golden-event-mentions']) > 1:
        return True
    elif 'event_triggers' in doc and doc['event_triggers']:
        trigger_probs = np.asarray(
            [trigger['event_type_probs'] for trigger in doc['event_triggers']]
        )
        if include_negatives:
            return trigger_probs.sum() > 0.0
        labeled_triggers = trigger_probs.sum(axis=1) > 0.0
        trigger_labels = trigger_probs[labeled_triggers].argmax(axis=1)
        events_positives = [label < len(SD4M_RELATION_TYPES)-1 for label in trigger_labels]
        if np.asarray(events_positives).sum() > 1:
            return True
    return False


def has_multiple_same_events(doc, include_negatives=False):
    """
    Parameters
    ----------
    doc: Document
    include_negatives: Count document as having events when at least one trigger is not an abstain

    Returns
    -------
    Whether the document contains multiple (positive) events of the same type
    """
    if 'events' in doc and len(doc['events']) > 1:
        event_types = [event['event_type'] for event in doc['events']]
        uniques, counts = np.unique(event_types, return_counts=True)
        return any(count > 1 for count in counts)
    elif 'golden-event-mentions' in doc and len(doc['golden-event-mentions']) > 1:
        event_types = [event['event_type'] for event in doc['golden-event-mentions']]
        uniques, counts = np.unique(event_types, return_counts=True)
        return any(count > 1 for count in counts)
    elif 'event_triggers' in doc and doc['event_triggers']:
        trigger_probs = np.asarray(
            [trigger['event_type_probs'] for trigger in doc['event_triggers']]
        )
        if include_negatives:
            return trigger_probs.sum() > 0.0
        labeled_triggers = trigger_probs.sum(axis=1) > 0.0
        trigger_labels = trigger_probs[labeled_triggers].argmax(axis=1)
        events_positives = [label < len(SD4M_RELATION_TYPES)-1 for label in trigger_labels]
        trigger_labels_np = np.asarray(trigger_labels)[events_positives]
        uniques, counts = np.unique(trigger_labels_np, return_counts=True)
        return any(count > 1 for count in counts)
    return False


def has_roles(doc, include_negatives=False):
    """
    Parameters
    ----------
    doc: Document
    include_negatives: Count document as having roles when at least one role is not an abstain

    Returns
    -------
    Whether the document contains any (positive) roles
    """
    if 'events' in doc and doc['events']:
        return any(len(event['arguments']) > 0 for event in doc['events'])
    elif 'golden-event-mentions' in doc and doc['golden-event-mentions']:
        return any(len(event['arguments']) > 0 for event in doc['golden-event-mentions'])
    elif 'event_roles' in doc and doc['event_roles']:
        role_probs = np.asarray(
            [role['event_argument_probs'] for role in doc['event_roles']]
        )
        if include_negatives:
            return role_probs.sum() > 0.0
        labeled_roles = role_probs.sum(axis=1) > 0.0
        role_labels = role_probs[labeled_roles].argmax(axis=1)
        if any(label < len(ROLE_LABELS)-1 for label in role_labels):
            return True
    return False


def get_snorkel_event_stats(dataset: pd.DataFrame):
    # Positive (Labeled positive vs. Abstains+negative), Documents, DataPoints
    assert 'event_triggers' in dataset and 'event_roles' in dataset
    event_doc_triggers = list(dataset['event_triggers'])
    event_doc_roles = list(dataset['event_roles'])
    trigger_class_freqs = {}
    for trigger_class in SD4M_RELATION_TYPES:
        trigger_class_freqs[trigger_class] = 0
    role_class_freqs = {}
    for role_class in ROLE_LABELS:
        role_class_freqs[role_class] = 0
    # Positive, Negative, Abstain

    trigger_probs = np.asarray([trigger['event_type_probs'] for triggers in event_doc_triggers
                                for trigger in triggers])
    docs_with_events = sum(dataset.apply(lambda document: has_events(document), axis=1))
    docs_with_multiple_events = sum(
        dataset.apply(lambda document: has_multiple_events(document), axis=1))
    docs_with_multiple_events_same_type = sum(
        dataset.apply(lambda document: has_multiple_same_events(document), axis=1))
    labeled_triggers = trigger_probs.sum(axis=-1) > 0.0
    trigger_a = len(trigger_probs) - sum(labeled_triggers)
    trigger_labels = trigger_probs[labeled_triggers].argmax(axis=-1)
    unique, counts = np.unique(trigger_labels, return_counts=True)
    for u, c in zip(unique, counts):
        trigger_class_freqs[SD4M_RELATION_TYPES[u]] = c
    trigger_n = trigger_class_freqs[NEGATIVE_TRIGGER_LABEL]
    trigger_p = len(trigger_probs) - trigger_a - trigger_n

    role_probs = np.asarray([role['event_argument_probs'] for roles in event_doc_roles
                             for role in roles])
    docs_with_roles = sum(dataset.apply(lambda document: has_roles(document), axis=1))
    labeled_roles = role_probs.sum(axis=-1) > 0.0
    role_a = len(role_probs) - sum(labeled_roles)
    role_labels = role_probs[labeled_roles].argmax(axis=-1)
    unique, counts = np.unique(role_labels, return_counts=True)
    for u, c in zip(unique, counts):
        role_class_freqs[ROLE_LABELS[u]] = c
    role_n = role_class_freqs[NEGATIVE_ARGUMENT_LABEL]
    role_p = len(role_probs) - role_a - role_n

    return {
        "# Docs": len(dataset),
        "# Docs with event triggers": docs_with_events,
        "# Docs with multiple event triggers": docs_with_multiple_events,
        "# Docs with multiple event triggers with same type": docs_with_multiple_events_same_type,
        "Average events per document": trigger_p/len(dataset),
        "# Event triggers with positive label": trigger_p,
        "# Event triggers with negative label": trigger_n,
        "# Event triggers with abstain": trigger_a,
        "Trigger class frequencies": trigger_class_freqs,
        "# Docs with event roles": docs_with_roles,
        "# Event role with positive label": role_p,
        "# Event roles with negative label": role_n,
        "# Event roles with abstain": role_a,
        "Role class frequencies": role_class_freqs
    }


def get_event_stats(dataset: pd.DataFrame):
    if 'event_triggers' in dataset and 'event_roles' in dataset:
        return get_snorkel_event_stats(dataset)
    assert 'events' in dataset
    docs_with_events = 0
    docs_with_roles = 0
    num_event_triggers = 0
    num_event_roles = 0
    trigger_class_freqs = {}
    for trigger_class in SD4M_RELATION_TYPES:
        trigger_class_freqs[trigger_class] = 0
    role_class_freqs = {}
    for role_class in ROLE_LABELS:
        role_class_freqs[role_class] = 0

    doc_events = list(dataset['events'])
    for events in doc_events:
        has_annotated_events = False
        has_annotated_roles = False
        for event in events:
            trigger_class_freqs[event['event_type']] += 1
            if event['event_type'] in SD4M_RELATION_TYPES[:-1]:
                num_event_triggers += 1
                has_annotated_events = True
                for arg in event['arguments']:
                    role_class_freqs[arg['role']] += 1
                    if arg['role'] in ROLE_LABELS[:-1]:
                        num_event_roles += 1
                        has_annotated_roles = True
        if has_annotated_events:
            docs_with_events += 1
        if has_annotated_roles:
            docs_with_roles += 1

    docs_with_multiple_events = sum(
        dataset.apply(lambda document: has_multiple_events(document), axis=1))
    docs_with_multiple_events_same_type = sum(
        dataset.apply(lambda document: has_multiple_same_events(document), axis=1))

    return {
        "# Docs": len(dataset),
        "# Docs with event triggers": docs_with_events,
        "# Docs with multiple event triggers": docs_with_multiple_events,
        "# Docs with multiple event triggers with same type": docs_with_multiple_events_same_type,
        "Average events per sentence": num_event_triggers / len(dataset),
        "# Event triggers": num_event_triggers,
        "Trigger class frequencies": trigger_class_freqs,
        "# Docs with event roles": docs_with_roles,
        "# Event roles": num_event_roles,
        "Role class frequencies": role_class_freqs
    }


def get_ace_event_stats(dataset: pd.DataFrame):
    ace_triggers = scorer.get_triggers(dataset.to_dict('records'), 'golden-event-mentions')
    ace_arguments = scorer.get_arguments(dataset.to_dict('records'), 'golden-event-mentions')
    num_event_triggers = len(ace_triggers)
    num_unique_event_triggers = len(set(ace_triggers))
    num_event_roles = len(ace_arguments)
    docs_with_events = sum(
        dataset.apply(lambda document: has_events(document), axis=1))
    docs_with_multiple_events = sum(
        dataset.apply(lambda document: has_multiple_events(document), axis=1))
    docs_with_multiple_events_same_type = sum(
        dataset.apply(lambda document: has_multiple_same_events(document), axis=1))
    docs_with_roles = sum(
        dataset.apply(lambda document: has_roles(document), axis=1))
    event_types = [ace_trigger[3] for ace_trigger in ace_triggers]
    uniques, counts = np.unique(event_types, return_counts=True)
    trigger_class_freqs = dict(zip(uniques, counts))
    arg_roles = [ace_arg[6] for ace_arg in ace_arguments]
    uniques, counts = np.unique(arg_roles, return_counts=True)
    role_class_freqs = dict(zip(uniques, counts))
    return {
        "# Sentences": len(dataset),
        "# Sentences with event triggers": docs_with_events,
        "# Sentences with multiple event triggers": docs_with_multiple_events,
        "# Sentences with multiple event triggers with same type":
            docs_with_multiple_events_same_type,
        "Average events per sentence": num_event_triggers / len(dataset),
        "# Event triggers": num_event_triggers,
        "# Unique Event triggers": num_unique_event_triggers,
        "Trigger class frequencies": trigger_class_freqs,
        "# Sentences with event roles": docs_with_roles,
        "# Event roles": num_event_roles,
        "Role class frequencies": role_class_freqs
    }


def add_doc_type(document: pd.Series):
    doc_type = 'Other'
    if 'docType' in document:
        return document
    elif is_rss(document['id']):
        doc_type = 'RSS_XML'
    elif is_twitter(document['id']):
        doc_type = 'TWITTER_JSON'
    document['docType'] = doc_type
    return document


def get_dataset_stats(dataset: pd.DataFrame):
    dataset = dataset.apply(add_doc_type, axis=1)
    dataset_stats = {'docType': 'MIXED'}
    dataset_stats.update(get_docs_tokens_entities_triggers(dataset))
    dataset_stats.update(get_event_stats(dataset))

    rss_dataset = dataset[dataset['docType'] == 'RSS_XML']
    twitter_dataset = dataset[dataset['docType'] == 'TWITTER_JSON']
    rss_stats = {'docType': 'RSS_XML'}
    rss_stats.update(get_docs_tokens_entities_triggers(rss_dataset))
    rss_stats.update(get_event_stats(rss_dataset))

    twitter_stats = {'docType': 'TWITTER_JSONL'}
    twitter_stats.update(get_docs_tokens_entities_triggers(twitter_dataset))
    twitter_stats.update(get_event_stats(twitter_dataset))

    stats = pd.DataFrame([dataset_stats, rss_stats, twitter_stats])
    return stats


def main(args):
    input_path = Path(args.input_path)
    assert input_path.exists(), 'Input not found: %s'.format(args.input_path)
    output_path = Path(args.output_path)

    dataset = pd.read_json(input_path, lines=True, encoding='utf8')

    stats = get_dataset_stats(dataset)
    stats.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print(stats.to_json(orient='records', lines=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Corpus statistics')
    parser.add_argument('--input_path', type=str, help='Path to corpus file')
    parser.add_argument('--output_path', type=str, help='Path to output file')
    arguments = parser.parse_args()
    main(arguments)
