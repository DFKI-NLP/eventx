from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, TokenEmbedder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch.nn import Linear, Parameter

from eventx import NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL, SD4M_RELATION_TYPES, ROLE_LABELS
from eventx.util import MicroFBetaMeasure
from eventx.util.loss import cross_entropy_with_probs


@Model.register('snorkel-eventx-model')
class SnorkelEventxModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 span_extractor: SpanExtractor,
                 entity_embedder: TokenEmbedder,
                 hidden_dim: int,
                 loss_weight: float = 1.0,
                 trigger_gamma: float = None,
                 role_gamma: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)
        self.num_trigger_classes = len(SD4M_RELATION_TYPES)
        self.num_role_classes = len(ROLE_LABELS)
        self.hidden_dim = hidden_dim
        self.loss_weight = loss_weight
        self.trigger_gamma = trigger_gamma
        self.role_gamma = role_gamma
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.entity_embedder = entity_embedder
        self.span_extractor = span_extractor
        self.trigger_projection = Linear(self.encoder.get_output_dim(), self.num_trigger_classes)
        self.trigger_to_hidden = Linear(self.encoder.get_output_dim(), self.hidden_dim)
        self.entities_to_hidden = Linear(self.encoder.get_output_dim(), self.hidden_dim)
        self.hidden_bias = Parameter(torch.Tensor(self.hidden_dim))
        torch.nn.init.normal_(self.hidden_bias)
        self.hidden_to_roles = Linear(self.hidden_dim,
                                      self.num_role_classes)
        self.trigger_accuracy = CategoricalAccuracy()
        # TODO check whether this works
        # trigger_labels_to_idx = self.vocab.
        # get_token_to_index_vocabulary(namespace=triggers_namespace)
        trigger_labels_to_idx = dict([(label, idx) for idx, label in enumerate(SD4M_RELATION_TYPES)])
        evaluated_trigger_idxs = list(trigger_labels_to_idx.values())
        evaluated_trigger_idxs.remove(trigger_labels_to_idx[NEGATIVE_TRIGGER_LABEL])
        self.trigger_f1 = MicroFBetaMeasure(average='micro',  # Macro averaging in get_metrics
                                            labels=evaluated_trigger_idxs)
        # TODO check whether this works
        # role_labels_to_idx = self.vocab.get_token_to_index_vocabulary(namespace=roles_namespace)
        role_labels_to_idx = dict([(label, idx) for idx, label in enumerate(ROLE_LABELS)])
        evaluated_role_idxs = list(role_labels_to_idx.values())
        evaluated_role_idxs.remove(role_labels_to_idx[NEGATIVE_ARGUMENT_LABEL])
        self.role_accuracy = CategoricalAccuracy()
        self.role_f1 = MicroFBetaMeasure(average='micro',  # Macro averaging in get_metrics
                                         labels=evaluated_role_idxs)
        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                entity_tags: torch.LongTensor,
                entity_spans: torch.LongTensor,
                trigger_spans: torch.LongTensor,
                trigger_labels: torch.LongTensor = None,
                arg_roles: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        embedded_tokens = self.text_field_embedder(tokens)
        text_mask = get_text_field_mask(tokens)
        embedded_entity_tags = self.entity_embedder(entity_tags)
        embedded_input = torch.cat([embedded_tokens, embedded_entity_tags], dim=-1)

        encoded_input = self.encoder(embedded_input, text_mask)

        ###########################
        # Trigger type prediction #
        ###########################

        # Extract the spans of the triggers
        trigger_spans_mask = (trigger_spans[:, :, 0] >= 0).long()
        encoded_triggers = self.span_extractor(sequence_tensor=encoded_input,
                                               span_indices=trigger_spans,
                                               sequence_mask=text_mask,
                                               span_indices_mask=trigger_spans_mask)

        # Pass the extracted triggers through a projection for classification
        trigger_logits = self.trigger_projection(encoded_triggers)

        # Add the trigger predictions to the output
        trigger_probabilities = F.softmax(trigger_logits, dim=-1)
        trigger_predictions = trigger_logits.argmax(dim=-1)
        output_dict = {"trigger_logits": trigger_logits,
                       "trigger_probabilities": trigger_probabilities,
                       "trigger_predictions": trigger_predictions}

        if trigger_labels is not None:
            # Compute loss and metrics using the given trigger labels
            dummy = torch.tensor([0.0] * len(SD4M_RELATION_TYPES),
                                 dtype=trigger_labels.dtype, device=trigger_labels.device)
            # TODO find more efficient method to get target mask
            trigger_mask = torch.tensor([[not array.equal(dummy) for array in batch]
                                         for batch in trigger_labels],
                                        device=trigger_labels.device)  # B x T
            # TODO probably not necessary since the dummy fields for trigger labels and argument
            #  roles already are tensors containing only zeros
            trigger_labels = trigger_labels * trigger_mask[..., None]  # B x T x Event Classes
            decoded_trigger_labels = trigger_labels.argmax(dim=2)
            self.trigger_accuracy(trigger_logits, decoded_trigger_labels, trigger_mask.float())
            self.trigger_f1(trigger_logits, decoded_trigger_labels, trigger_mask.float())

            trigger_logits_t = trigger_logits.permute(0, 2, 1)
            trigger_loss = self._cross_entropy_loss(logits=trigger_logits_t,
                                                    target=trigger_labels,
                                                    target_mask=trigger_mask)

            output_dict["triggers_loss"] = trigger_loss
            output_dict["loss"] = trigger_loss

        ########################################
        # Argument detection and role labeling #
        ########################################

        # Extract the spans of the encoded entities
        entity_spans_mask = (entity_spans[:, :, 0] >= 0).long()
        encoded_entities = self.span_extractor(sequence_tensor=encoded_input,
                                               span_indices=entity_spans,
                                               sequence_mask=text_mask,
                                               span_indices_mask=entity_spans_mask)

        # Project both triggers and entities/args into a 'hidden' comparison space
        triggers_hidden = self.trigger_to_hidden(encoded_triggers)
        args_hidden = self.entities_to_hidden(encoded_entities)

        # Create the cross-product of triggers and args via broadcasting
        trigger = triggers_hidden.unsqueeze(2)  # B x T x 1 x H
        args = args_hidden.unsqueeze(1)  # B x T x E x H
        trigger_arg = trigger + args + self.hidden_bias  # B x T x E x H

        # Pass through activation and projection for classification
        role_activations = F.relu(trigger_arg)
        role_logits = self.hidden_to_roles(role_activations)  # B x T x E x R

        # Add the role predictions to the output
        role_probabilities = torch.softmax(role_logits, dim=-1)
        output_dict['role_logits'] = role_logits
        output_dict['role_probabilities'] = role_probabilities

        # Compute loss and metrics using the given role labels
        if arg_roles is not None:
            arg_roles = self._assert_target_shape(logits=role_logits, target=arg_roles,
                                                  fill_value=0)
            dummy = torch.tensor([0.0] * len(ROLE_LABELS),
                                 dtype=arg_roles.dtype, device=arg_roles.device)
            # TODO find more efficient method to get target mask
            target_mask = torch.tensor([[[not array.equal(dummy) for array in trigger]
                                         for trigger in batch]
                                        for batch in arg_roles],
                                       device=arg_roles.device)  # B x T x E
            # TODO probably not necessary since the dummy fields for trigger labels and argument
            #  roles already are tensors containing only zeros
            target = arg_roles * target_mask[..., None]  # B x T x E x R
            decoded_target = target.argmax(dim=3)
            self.role_accuracy(role_logits, decoded_target, target_mask.float())
            self.role_f1(role_logits, decoded_target, target_mask.float())

            # Masked batch-wise cross entropy loss, optionally with focal-loss
            role_logits_t = role_logits.permute(0, 3, 1, 2)
            role_loss = self._cross_entropy_loss(logits=role_logits_t,
                                                 target=target,
                                                 target_mask=target_mask)

            output_dict['role_loss'] = role_loss
            output_dict['loss'] += self.loss_weight * role_loss

        # Append the original tokens for visualization
        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'trigger_acc': self.trigger_accuracy.get_metric(reset=reset),
            'trigger_f1': self.trigger_f1.get_metric(reset=reset)['fscore'],
            'role_acc': self.role_accuracy.get_metric(reset=reset),
            'role_f1': self.role_f1.get_metric(reset=reset)['fscore']
        }

    @staticmethod
    def _assert_target_shape(logits, target, fill_value=0):
        """
        Asserts that target tensors are always of the same size of logits. This is not always
        the case since some batches are not completely filled.
        """
        expected_shape = logits.shape
        if target.shape == expected_shape:
            return target
        else:
            new_target = torch.full(size=expected_shape,
                                    fill_value=fill_value,
                                    dtype=target.dtype,
                                    device=target.device)
            batch_size, triggers_len, arguments_len, _ = target.shape
            new_target[:, :triggers_len, :arguments_len] = target
            return new_target

    @staticmethod
    def _cross_entropy_loss(logits, target, target_mask) -> torch.Tensor:
        loss_unreduced = cross_entropy_with_probs(logits, target, reduction="none")
        masked_loss = loss_unreduced * target_mask
        batch_size = target.size(0)
        loss_per_batch = masked_loss.view(batch_size, -1).sum(dim=1)
        mask_per_batch = target_mask.view(batch_size, -1).sum()
        return (loss_per_batch / mask_per_batch).sum() / batch_size
