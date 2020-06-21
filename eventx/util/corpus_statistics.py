import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from eventx import SD4M_RELATION_TYPES, ROLE_LABELS, NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL


def is_rss(doc_id: str):
    return doc_id.startswith("http")


def is_twitter(doc_id: str):
    return doc_id[0].isdigit()


def get_docs_tokens_entities_triggers(dataset: pd.DataFrame):
    """
    Retrieves numbers about the dataset

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
        return True
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
    return {
        "# Docs": len(dataset),
        "# Docs with event triggers": docs_with_events,
        "# Event triggers": num_event_triggers,
        "Trigger class frequencies": trigger_class_freqs,
        "# Docs with event roles": docs_with_roles,
        "# Event roles": num_event_roles,
        "Role class frequencies": role_class_freqs
    }


def get_doc_type(document: pd.Series):
    if 'docType' in document:
        return document['docType']
    elif is_rss(document['id']):
        return 'RSS_XML'
    elif is_twitter(document['id']):
        return 'TWITTER_JSON'
    else:
        return 'Other'


def get_doc_types(dataframe: pd.DataFrame):
    return dataframe.apply(lambda document: get_doc_type(document), axis=1)


def main(args):
    input_path = Path(args.input_path)
    assert input_path.exists(), 'Input not found: %s'.format(args.input_path)
    output_path = Path(args.output_path)

    dataset = pd.read_json(input_path, lines=True, encoding='utf8')
    dataset['docType'] = get_doc_types(dataset)
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
    stats.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print(stats.to_json(orient='records', lines=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Corpus statistics')
    parser.add_argument('--input_path', type=str, help='Path to corpus file')
    parser.add_argument('--output_path', type=str, help='Path to output file')
    arguments = parser.parse_args()
    main(arguments)
