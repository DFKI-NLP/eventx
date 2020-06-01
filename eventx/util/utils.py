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


def snorkel_to_ace_format(doc_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes list of documents with event triggers and event roles in the Snorkel format and creates
    events in the ACE format.

    Parameters
    ----------
    doc_list: List of documents with event triggers and event roles

    Returns
    -------
    List of documents with events
    """
    df = pd.DataFrame(doc_list)
    assert 'event_triggers' in df
    assert 'event_roles' in df
    converted_df = df.apply(create_events, axis=1)\
        .drop(labels=['event_triggers', 'event_roles'], axis=1)
    return converted_df.to_dict('records')


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
        # TODO: save string labels instead of redoing it later on
        filtered_triggers = [t for t in document['event_triggers']
                             if SD4M_RELATION_TYPES[np.asarray(t['event_type_probs']).argmax()]
                             != NEGATIVE_TRIGGER_LABEL]
        filtered_roles = [r for r in document['event_roles']
                          if ROLE_LABELS[np.asarray(r['event_argument_probs']).argmax()]
                          != NEGATIVE_ARGUMENT_LABEL]

        for event_trigger in filtered_triggers:
            trigger_entity = get_entity(event_trigger['id'], document['entities'])
            event_type = SD4M_RELATION_TYPES[np.asarray(event_trigger['event_type_probs']).argmax()]
            formatted_trigger = {
                'id': trigger_entity['id'],
                'text': trigger_entity['text'],
                'entity_type': trigger_entity['entity_type'],
                'start': trigger_entity['start'],
                'end': trigger_entity['end']
            }
            relevant_args = [arg for arg in filtered_roles if arg['trigger'] == event_trigger['id']]
            formatted_args = []
            for event_arg in relevant_args:
                event_arg_entity = get_entity(event_arg['argument'], document['entities'])
                arg_role = ROLE_LABELS[np.asarray(event_arg['event_argument_probs']).argmax()]
                formatted_arg = {
                    'id': event_arg_entity['id'],
                    'text': event_arg_entity['text'],
                    'entity_type': event_arg_entity['entity_type'],
                    'start': event_arg_entity['start'],
                    'end': event_arg_entity['end'],
                    'role': arg_role
                }
                formatted_args.append(formatted_arg)
            formatted_event = {
                'event_type': event_type,
                'trigger': formatted_trigger,
                'arguments': formatted_args
            }
            formatted_events.append(formatted_event)
    document['events'] = formatted_events
    return document


def ace_to_snorkel_format(doc_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes list of documents with events in the ACE format and creates
    events in the Snorkel format with event triggers and event roles.

    Parameters
    ----------
    doc_list: List of documents with event triggers and event roles

    Returns
    -------
    List of documents with events
    """
    df = pd.DataFrame(doc_list)
    assert 'events' in df
    converted_df = df.apply(convert_events, axis=1)\
        .drop(labels=['events'], axis=1)
    return converted_df.to_dict('records')


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
