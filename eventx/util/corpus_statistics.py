import argparse
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np

from eventx import SD4M_RELATION_TYPES, ROLE_LABELS, SDW_RELATION_TYPES, SDW_ROLE_LABELS, \
    NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL
from eventx.util import scorer


def is_rss(doc_id: str):
    return doc_id.startswith("http")


def is_twitter(doc_id: str):
    return doc_id[0].isdigit()


def get_num_of_tokens(dataset: pd.DataFrame, token_keyword="tokens"):
    return sum(dataset.apply(lambda x: len(x[token_keyword]), axis=1))


def get_num_of_entities(dataset: pd.DataFrame, entity_keyword="entities"):
    return sum(dataset.apply(lambda x: len(x[entity_keyword]), axis=1))


def get_num_of_trigger_entities(dataset: pd.DataFrame, entity_keyword="entities"):
    return sum(dataset.apply(
        lambda x: len([e for e in x[entity_keyword] if e["entity_type"] == "trigger"]), axis=1)
    )


def get_num_of_event_triggers(dataset: pd.DataFrame, snorkel_format=False, event_keyword="events",
                              unique_triggers=False):
    if snorkel_format and "event_triggers" in dataset:
        trigger_probs = np.asarray(
            [trigger["event_type_probs"] for doc_triggers in dataset["event_triggers"]
             for trigger in doc_triggers]
        )
        labeled_triggers = trigger_probs.sum(axis=1) > 0.0
        trigger_labels = trigger_probs[labeled_triggers].argmax(axis=1)
        negative_label_idx = trigger_probs.shape[-1] - 1
        num_of_triggers = np.sum(trigger_labels < negative_label_idx)
        return num_of_triggers
    elif unique_triggers:
        num_of_unique_triggers = 0
        for doc_events in dataset[event_keyword]:
            doc_triggers = [(event["trigger"]["start"], event["trigger"]["end"])
                            for event in doc_events]
            num_of_unique_triggers += len(set(doc_triggers))
        return num_of_unique_triggers
    else:
        return sum(dataset.apply(
            lambda x: len([e["trigger"] for e in x[event_keyword]]), axis=1)
        )


def get_num_of_event_arguments(dataset: pd.DataFrame, snorkel_format=False, event_keyword="events"):
    if snorkel_format and "event_roles" in dataset:
        role_probs = np.asarray(
            [role["event_argument_probs"] for doc_roles in dataset["event_roles"]
             for role in doc_roles]
        )
        labeled_roles = role_probs.sum(axis=1) > 0.0
        role_labels = role_probs[labeled_roles].argmax(axis=1)
        negative_label_idx = role_probs.shape[-1] - 1
        num_of_roles = np.sum(role_labels < negative_label_idx)
        return num_of_roles
    else:
        num_of_arguments = 0
        for doc_events in dataset[event_keyword]:
            doc_args = [len(event["arguments"]) for event in doc_events]
            num_of_arguments += sum(doc_args)
        return num_of_arguments


def has_triggers(doc):
    """
    Parameters
    ----------
    doc: Document

    Returns
    -------
    Whether the document contains any triggers
    """
    entities = doc["entities"]
    return any(entity["entity_type"] == "trigger" for entity in entities)


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
    if "events" in doc and doc["events"]:
        return True
    elif "golden-event-mentions" in doc and doc["golden-event-mentions"]:
        return True
    elif "event_triggers" in doc and doc["event_triggers"]:
        trigger_probs = np.asarray(
            [trigger["event_type_probs"] for trigger in doc["event_triggers"]]
        )
        if include_negatives:
            return trigger_probs.sum() > 0.0
        labeled_triggers = trigger_probs.sum(axis=1) > 0.0
        trigger_labels = trigger_probs[labeled_triggers].argmax(axis=1)
        negative_label_idx = trigger_probs.shape[1] - 1
        if any(label < negative_label_idx for label in trigger_labels):
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
    if "events" in doc and len(doc["events"]) > 1:
        return True
    elif "golden-event-mentions" in doc and len(doc["golden-event-mentions"]) > 1:
        return True
    elif "event_triggers" in doc and doc["event_triggers"]:
        trigger_probs = np.asarray(
            [trigger["event_type_probs"] for trigger in doc["event_triggers"]]
        )
        if include_negatives:
            return trigger_probs.sum() > 0.0
        labeled_triggers = trigger_probs.sum(axis=1) > 0.0
        trigger_labels = trigger_probs[labeled_triggers].argmax(axis=1)
        negative_label_idx = trigger_probs.shape[-1] - 1
        events_positives = [label < negative_label_idx for label in trigger_labels]
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
    if "events" in doc and len(doc["events"]) > 1:
        event_types = [event["event_type"] for event in doc["events"]]
        uniques, counts = np.unique(event_types, return_counts=True)
        return any(count > 1 for count in counts)
    elif "golden-event-mentions" in doc and len(doc["golden-event-mentions"]) > 1:
        event_types = [event["event_type"] for event in doc["golden-event-mentions"]]
        uniques, counts = np.unique(event_types, return_counts=True)
        return any(count > 1 for count in counts)
    elif "event_triggers" in doc and doc["event_triggers"]:
        trigger_probs = np.asarray(
            [trigger["event_type_probs"] for trigger in doc["event_triggers"]]
        )
        if include_negatives:
            return trigger_probs.sum() > 0.0
        labeled_triggers = trigger_probs.sum(axis=1) > 0.0
        trigger_labels = trigger_probs[labeled_triggers].argmax(axis=1)
        negative_label_idx = trigger_probs.shape[-1] - 1
        events_positives = [label < negative_label_idx for label in trigger_labels]
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
    if "events" in doc and doc["events"]:
        return any(len(event["arguments"]) > 0 for event in doc["events"])
    elif "golden-event-mentions" in doc and doc["golden-event-mentions"]:
        return any(len(event["arguments"]) > 0 for event in doc["golden-event-mentions"])
    elif "event_roles" in doc and doc["event_roles"]:
        role_probs = np.asarray(
            [role["event_argument_probs"] for role in doc["event_roles"]]
        )
        if include_negatives:
            return role_probs.sum() > 0.0
        labeled_roles = role_probs.sum(axis=1) > 0.0
        role_labels = role_probs[labeled_roles].argmax(axis=1)
        negative_label_idx = role_probs.shape[-1] - 1
        if any(label < negative_label_idx for label in role_labels):
            return True
    return False


def get_snorkel_labeling_info(dataset: pd.DataFrame, relation_types=SD4M_RELATION_TYPES,
                              role_classes=ROLE_LABELS, return_frequencies=True):
    assert "event_triggers" in dataset and "event_roles" in dataset
    event_doc_triggers = list(dataset["event_triggers"])
    event_doc_roles = list(dataset["event_roles"])
    trigger_class_freqs = {}
    for trigger_class in relation_types:
        trigger_class_freqs[trigger_class] = 0
    role_class_freqs = {}
    for role_class in role_classes:
        role_class_freqs[role_class] = 0
    # Positive, Negative, Abstain
    trigger_probs = np.asarray([trigger["event_type_probs"] for triggers in event_doc_triggers
                                for trigger in triggers])
    labeled_triggers = trigger_probs.sum(axis=-1) > 0.0
    trigger_a = len(trigger_probs) - sum(labeled_triggers)
    trigger_labels = trigger_probs[labeled_triggers].argmax(axis=-1)
    unique, counts = np.unique(trigger_labels, return_counts=True)
    for u, c in zip(unique, counts):
        trigger_class_freqs[relation_types[u]] = c
    trigger_n = trigger_class_freqs[NEGATIVE_TRIGGER_LABEL]
    trigger_p = len(trigger_probs) - trigger_a - trigger_n

    role_probs = np.asarray([role["event_argument_probs"] for roles in event_doc_roles
                             for role in roles])
    labeled_roles = role_probs.sum(axis=-1) > 0.0
    role_a = len(role_probs) - sum(labeled_roles)
    role_labels = role_probs[labeled_roles].argmax(axis=-1)
    unique, counts = np.unique(role_labels, return_counts=True)
    for u, c in zip(unique, counts):
        role_class_freqs[role_classes[u]] = c
    role_n = role_class_freqs[NEGATIVE_ARGUMENT_LABEL]
    role_p = len(role_probs) - role_a - role_n
    labeling_info = {
        "# Event triggers with positive label": trigger_p,
        "# Event triggers with negative label": trigger_n,
        "# Event triggers with abstain": trigger_a,
        "# Event role with positive label": role_p,
        "# Event roles with negative label": role_n,
        "# Event roles with abstain": role_a,
    }
    if return_frequencies:
        labeling_info["trigger_class_freqs"] = trigger_class_freqs
        labeling_info["role_class_freqs"] = role_class_freqs
    return labeling_info


def get_general_stats(dataset: pd.DataFrame, include_trigger_entities=False,
                      token_keyword="tokens", entity_keyword="entities",
                      event_keyword="events", sentence_level=False,
                      snorkel_format=False):
    unit_name = "Sentences" if sentence_level else "Docs"
    general_stats = {
        f"# {unit_name}": len(dataset),
        "# Tokens": get_num_of_tokens(dataset, token_keyword=token_keyword),
        "# Entities": get_num_of_entities(dataset, entity_keyword=entity_keyword)
    }
    if include_trigger_entities:
        general_stats["# Trigger Entities"] = get_num_of_trigger_entities(
            dataset, entity_keyword=entity_keyword)
    general_stats["# Event Triggers"] = get_num_of_event_triggers(
        dataset, event_keyword=event_keyword, snorkel_format=snorkel_format)
    general_stats["# Unique Event Triggers"] = get_num_of_event_triggers(
        dataset, event_keyword=event_keyword, unique_triggers=True, snorkel_format=snorkel_format)
    general_stats["# Arguments"] = get_num_of_event_arguments(
        dataset, event_keyword=event_keyword, snorkel_format=snorkel_format)
    general_stats[f"# {unit_name} with event triggers"] = sum(
        dataset.apply(lambda document: has_events(document), axis=1))
    general_stats[f"# {unit_name} with multiple event triggers"] = sum(
        dataset.apply(lambda document: has_multiple_events(document), axis=1))
    general_stats[f"# {unit_name} with multiple event triggers with same type"] = sum(
        dataset.apply(lambda document: has_multiple_same_events(document), axis=1))
    general_stats[f"# {unit_name} with event roles"] = sum(
        dataset.apply(lambda document: has_roles(document), axis=1))
    return general_stats


def get_ace_general_stats(dataset: pd.DataFrame):
    return get_general_stats(
        dataset=dataset,
        include_trigger_entities=False,
        token_keyword="words",
        entity_keyword="golden-entity-mentions",
        event_keyword="golden-event-mentions",
        sentence_level=True,
        snorkel_format=False
    )


def get_snorkel_general_stats(dataset: pd.DataFrame):
    return get_general_stats(
        dataset=dataset,
        include_trigger_entities=True,
        token_keyword="tokens",
        entity_keyword="entities",
        sentence_level=False,
        snorkel_format=True
    )


def get_smartdata_general_stats(dataset: pd.DataFrame):
    return get_general_stats(
        dataset=dataset,
        include_trigger_entities=True,
        token_keyword="tokens",
        entity_keyword="entities",
        event_keyword="events",
        sentence_level=False,
        snorkel_format=False
    )


def get_trigger_class_frequencies(dataset: pd.DataFrame, event_keyword="events",
                                  class_names: Optional[List[str]] = None):
    assert event_keyword in dataset, f"{event_keyword} is not a valid keyword for the dataset"
    assert event_keyword in ["events", "event_triggers", "golden-event-mentions"], \
        f"{event_keyword} is not a valid event keyword"
    if event_keyword == "event_triggers":
        # Snorkel Daystream
        event_doc_triggers = list(dataset["event_triggers"])
        trigger_probs = np.asarray([trigger["event_type_probs"]
                                    for triggers in event_doc_triggers
                                    for trigger in triggers])
        labeled_triggers = trigger_probs.sum(axis=-1) > 0.0
        event_types = trigger_probs[labeled_triggers].argmax(axis=-1)
        uniques, counts = np.unique(event_types, return_counts=True)
        trigger_class_freqs = dict(zip(uniques, counts))
        if class_names and len(class_names) >= len(uniques):
            named_trigger_class_freqs = {}
            for idx, class_name in enumerate(class_names):
                count = 0
                if idx in uniques:
                    count_idx = list(uniques).index(idx)
                    count = trigger_class_freqs[count_idx]
                named_trigger_class_freqs[class_name] = count
            return named_trigger_class_freqs
        else:
            return trigger_class_freqs
    else:
        # SD4M/SDW, ACE2005
        triggers = scorer.get_triggers(dataset.to_dict("records"), event_keyword)
        event_types = [trigger[3] for trigger in triggers]
    uniques, counts = np.unique(event_types, return_counts=True)
    trigger_class_freqs = dict(zip(uniques, counts))
    if class_names and len(class_names) >= len(uniques):
        named_trigger_class_freqs = {}
        for class_name in class_names:
            count = 0
            if class_name in uniques:
                count = trigger_class_freqs[class_name]
            named_trigger_class_freqs[class_name] = count
        return named_trigger_class_freqs
    else:
        return trigger_class_freqs


def get_role_class_frequencies(dataset: pd.DataFrame, event_keyword="events",
                               class_names: Optional[List[str]] = None):
    assert event_keyword in dataset, f"{event_keyword} is not a valid keyword for the dataset"
    assert event_keyword in ["events", "event_roles", "golden-event-mentions"], \
        f"{event_keyword} is not a valid event keyword"
    if event_keyword == "event_roles":
        # Snorkel Daystream
        event_doc_roles = list(dataset["event_roles"])
        role_probs = np.asarray([role["event_argument_probs"]
                                 for roles in event_doc_roles
                                 for role in roles])
        labeled_roles = role_probs.sum(axis=-1) > 0.0
        arg_roles = role_probs[labeled_roles].argmax(axis=-1)
        uniques, counts = np.unique(arg_roles, return_counts=True)
        role_class_freqs = dict(zip(uniques, counts))
        if class_names and len(class_names) >= len(uniques):
            named_role_class_freqs = {}
            for idx, class_name in enumerate(class_names):
                count = 0
                if idx in uniques:
                    count_idx = list(uniques).index(idx)
                    count = role_class_freqs[count_idx]
                named_role_class_freqs[class_name] = count
            return named_role_class_freqs
        else:
            return role_class_freqs
    else:
        # SD4M/SDW, ACE2005
        arguments = scorer.get_arguments(dataset.to_dict("records"), event_keyword)
        arg_roles = [arg[6] for arg in arguments]
    uniques, counts = np.unique(arg_roles, return_counts=True)
    role_class_freqs = dict(zip(uniques, counts))
    if class_names and len(class_names) >= len(uniques):
        named_role_class_freqs = {}
        for class_name in class_names:
            count = 0
            if class_name in uniques:
                count = role_class_freqs[class_name]
            named_role_class_freqs[class_name] = count
        return named_role_class_freqs
    else:
        return role_class_freqs


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
    return {
        "# Docs": len(dataset),
        "# Tokens": get_num_of_tokens(dataset, token_keyword="tokens"),
        "# Entities": get_num_of_entities(dataset, entity_keyword="entities"),
        "# Trigger Entities": get_num_of_trigger_entities(dataset, entity_keyword="entities")
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
    return {
        "# Sentences": len(dataset),
        "# Tokens": get_num_of_tokens(dataset, token_keyword="words"),
        "# Entities": get_num_of_entities(dataset, entity_keyword="golden-entity-mentions"),
        "# Triggers": get_num_of_event_triggers(dataset, event_keyword="golden-event-mentions"),
        "# Unique Triggers": get_num_of_event_triggers(dataset,
                                                       event_keyword="golden-event-mentions",
                                                       unique_triggers=True),
        "# Arguments": get_num_of_event_arguments(dataset,
                                                  event_keyword="golden-event-mentions")
    }


def add_doc_type(document: pd.Series):
    doc_type = "Other"
    if "docType" in document:
        return document
    elif is_rss(document["id"]):
        doc_type = "RSS_XML"
    elif is_twitter(document["id"]):
        doc_type = "TWITTER_JSON"
    document["docType"] = doc_type
    return document
