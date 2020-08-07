from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

from eventx import SD4M_RELATION_TYPES, NEGATIVE_TRIGGER_LABEL, ROLE_LABELS, NEGATIVE_ARGUMENT_LABEL


def one_hot_encode(label, label_names, negative_label='O'):
    label = label if label in label_names else negative_label
    class_probs = np.asarray([1.0 if label_name == label else 0.0 for label_name in label_names])
    return class_probs


def one_hot_decode(class_probs, label_names):
    class_probs_array = np.asarray(class_probs)
    class_name = label_names[class_probs_array.argmax()]
    return class_name


def get_entity(entity_id: str, entities: List[Dict[str, Any]]) -> Dict:
    """
    Retrieves entity from list of entities given the entity id.

    Parameters
    ----------
    entity_id: String identifier of the relevant entity.
    entities: List of entities

    Returns
    -------
    Entity from entity list with matching entity id

    """
    entity: Optional[Dict] = next((x for x in entities if x['id'] == entity_id), None)
    if entity:
        return entity
    else:
        raise Exception(f'The entity_id {entity_id} was not found in:\n {entities}')


def snorkel_to_ace_format(doc_df):
    """
    Takes list of documents with event triggers and event roles in the Snorkel format and creates
    events in the ACE format.

    Parameters
    ----------
    doc_df: List of documents/ pd.DataFrame with event triggers and event roles

    Returns
    -------
    pd.DataFrame of documents with events
    """
    df = pd.DataFrame(doc_df)
    assert 'event_triggers' in df
    assert 'event_roles' in df
    converted_df = df.apply(create_events, axis=1)\
        .drop(labels=['event_triggers', 'event_roles'], axis=1)
    return converted_df


def create_events(document):
    """
    Takes a row (document) and creates events in the ACE format using event triggers and
    event arguments.

    Parameters
    ----------
    document: Document containing among others event triggers and event arguments

    Returns
    -------
    Row (document) with events
    """
    formatted_events = []
    if 'entities' in document and 'event_triggers' in document and 'event_roles' in document:
        for event_trigger in document['event_triggers']:
            event_type_probs = np.asarray(event_trigger['event_type_probs'])
            if event_type_probs.sum() == 0.0:
                continue
            trigger_label_idx = event_type_probs.argmax()
            trigger_label = SD4M_RELATION_TYPES[trigger_label_idx]
            if trigger_label == NEGATIVE_TRIGGER_LABEL:
                continue
            trigger_entity = get_entity(event_trigger['id'], document['entities'])
            relevant_args = [arg for arg in document['event_roles']
                             if arg['trigger'] == event_trigger['id']]
            formatted_args = []
            for event_arg in relevant_args:
                event_argument_probs = np.asarray(event_arg['event_argument_probs'])
                if event_argument_probs.sum() == 0.0:
                    continue
                role_label_idx = np.asarray(event_arg['event_argument_probs']).argmax()
                role_label = ROLE_LABELS[role_label_idx]
                if role_label == NEGATIVE_ARGUMENT_LABEL:
                    continue
                event_arg_entity = get_entity(event_arg['argument'], document['entities'])
                event_arg_entity['role'] = role_label
                formatted_args.append(event_arg_entity)
            formatted_event = {
                'event_type': trigger_label,
                'trigger': trigger_entity,
                'arguments': formatted_args
            }
            formatted_events.append(formatted_event)
    document['events'] = formatted_events
    return document


def ace_to_snorkel_format(doc_df: List[Dict[str, Any]]):
    """
    Takes list of documents with events in the ACE format and creates
    events in the Snorkel format with event triggers and event roles.

    Parameters
    ----------
    doc_df: List of documents/ pd.DataFrame with event triggers and event roles

    Returns
    -------
    pd.DataFrame of documents with events
    """
    df = pd.DataFrame(doc_df)
    assert 'events' in df
    converted_df = df.apply(convert_events, axis=1).drop(labels=['events'], axis=1)
    return converted_df


def convert_events(document):
    """
    Takes a document (document) and constructs event triggers and event roles from
    events in the ACE format.

    Parameters
    ----------
    document: Document containing events

    Returns
    -------
    Row (document) with event triggers and event roles
    """
    event_triggers = []
    event_roles = []
    for event in document['events']:
        event_type = event['event_type']
        trigger = event['trigger']
        event_triggers.append({
            'id': trigger['id'],
            'event_type_probs': one_hot_encode(event_type, SD4M_RELATION_TYPES,
                                               NEGATIVE_TRIGGER_LABEL)
        })
        for argument in event['arguments']:
            event_roles.append({
                'trigger': trigger['id'],
                'argument': argument['id'],
                'event_argument_probs': one_hot_encode(argument['role'], ROLE_LABELS,
                                                       NEGATIVE_ARGUMENT_LABEL)
            })
    document['event_triggers'] = event_triggers
    document['event_roles'] = event_roles
    return document
