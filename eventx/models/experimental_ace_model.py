from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import numpy as np
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, TimeDistributed, TokenEmbedder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from overrides import overrides
from torch.nn import Linear, Parameter

from eventx.util import MicroFBetaMeasure
from eventx import NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL


@Model.register('experimental-ace-model')
class ExperimentalAceModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 span_extractor: SpanExtractor,
                 entity_embedder: TokenEmbedder,
                 trigger_embedder: TokenEmbedder,
                 hidden_dim: int,
                 loss_weight: float = 1.0,
                 trigger_gamma: float = None,
                 role_gamma: float = None,
                 triggers_namespace: str = 'trigger_labels',
                 roles_namespace: str = 'arg_role_labels',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)
        self._triggers_namespace = triggers_namespace
        self._roles_namespace = roles_namespace
        self.num_trigger_classes = self.vocab.get_vocab_size(triggers_namespace)
        self.num_role_classes = self.vocab.get_vocab_size(roles_namespace)
        self.hidden_dim = hidden_dim
        self.loss_weight = loss_weight
        self.trigger_gamma = trigger_gamma
        self.role_gamma = role_gamma
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.entity_embedder = entity_embedder
        self.trigger_embedder = trigger_embedder
        self.span_extractor = span_extractor
        self.trigger_projection = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                         self.num_trigger_classes))
        self.trigger_to_hidden = Linear(
            self.encoder.get_output_dim() + self.trigger_embedder.get_output_dim(),
            self.hidden_dim)
        self.entities_to_hidden = Linear(self.encoder.get_output_dim(), self.hidden_dim)
        self.hidden_bias = Parameter(torch.Tensor(self.hidden_dim))
        torch.nn.init.normal_(self.hidden_bias)
        self.hidden_to_roles = Linear(self.hidden_dim,
                                      self.num_role_classes)
        self.trigger_accuracy = CategoricalAccuracy()
        self.trigger_f1 = SpanBasedF1Measure(vocab,
                                             tag_namespace=triggers_namespace,
                                             label_encoding="BIO",
                                             ignore_classes=[NEGATIVE_TRIGGER_LABEL])
        role_labels_to_idx = self.vocab.get_token_to_index_vocabulary(namespace=roles_namespace)
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
                triggers: torch.LongTensor = None,
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

        # Pass the extracted triggers through a projection for classification
        trigger_logits = self.trigger_projection(encoded_input)

        # Add the trigger predictions to the output
        trigger_probabilities = F.softmax(trigger_logits, dim=-1)
        trigger_predictions = trigger_logits.argmax(dim=-1)
        output_dict = {"trigger_logits": trigger_logits,
                       "trigger_probabilities": trigger_probabilities}

        if triggers is not None:
            # Compute loss and metrics using the given trigger labels
            self.trigger_accuracy(trigger_logits, triggers, text_mask.float())
            self.trigger_f1(trigger_logits, triggers, text_mask.float())
            loss = sequence_cross_entropy_with_logits(logits=trigger_logits,
                                                      targets=triggers,
                                                      weights=text_mask,
                                                      gamma=self.trigger_gamma)
            output_dict["triggers_loss"] = loss
            output_dict["loss"] = loss

        ########################################
        # Argument detection and role labeling #
        ########################################

        # Extract the spans of the encoded entities
        entity_spans_mask = (entity_spans[:, :, 0] >= 0).squeeze(-1).long()
        encoded_entities = self.span_extractor(sequence_tensor=encoded_input,
                                               span_indices=entity_spans,
                                               sequence_mask=text_mask,
                                               span_indices_mask=entity_spans_mask)

        # Project both triggers and entities/args into a 'hidden' comparison space
        embedded_trigger_labels = self.trigger_embedder(trigger_predictions)
        triggers_hidden = self.trigger_to_hidden(
            torch.cat([encoded_input, embedded_trigger_labels], dim=-1))  # B x L x H
        entities_hidden = self.entities_to_hidden(encoded_entities)  # B x E x H

        # Create the cross-product of triggers and args via broadcasting
        trigger = triggers_hidden.unsqueeze(2)  # Shape: B x L x 1 x H
        args = entities_hidden.unsqueeze(1)  # Shape: B x 1 x E x H
        trigger_arg = trigger + args + self.hidden_bias  # B x L x E x H

        # Pass through activation and projection for classification
        role_activations = F.relu(trigger_arg)
        role_logits = self.hidden_to_roles(role_activations)  # B x L x E x R

        # Add the role predictions to the output
        role_probabilities = torch.softmax(role_logits, dim=-1)
        output_dict['role_logits'] = role_logits
        output_dict['role_probabilities'] = role_probabilities

        # Compute loss and metrics using the given role labels
        if arg_roles is not None:
            arg_roles = self._assert_target_seq_len(seq_len=embedded_tokens.shape[1],
                                                    target=arg_roles)
            target_mask = (arg_roles != -1)
            target = arg_roles * target_mask  # remove negative indices

            self.role_accuracy(role_logits, target, target_mask.float())
            self.role_f1(role_logits, target, target_mask.float())

            # Masked batch-wise cross entropy loss, optionally with focal-loss
            role_logits_t = role_logits.permute(0, 3, 1, 2)
            role_loss = self._cross_entropy_focal_loss(logits=role_logits_t,
                                                       target=target,
                                                       target_mask=target_mask,
                                                       gamma=self.role_gamma)

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
            'trigger_f1': self.trigger_f1.get_metric(reset=reset)['f1-measure-overall'],
            'role_acc': self.role_accuracy.get_metric(reset=reset),
            'role_f1': self.role_f1.get_metric(reset=reset)['fscore']
        }

    def retrieve_trigger_spans_from_probabilities(self, trigger_probabilities):
        # Retrieve predicted iob2 trigger tags, get exclusive spans
        batch_trigger_spans = []
        trigger_predictions = trigger_probabilities.cpu().data.numpy()
        for example in np.argmax(trigger_predictions, axis=-1):
            trigger_tags = [self.vocab.get_token_from_index(trigger_idx, self._triggers_namespace)
                            for trigger_idx in example]
            trigger_labels_with_spans = bio_tags_to_spans(trigger_tags)
            trigger_spans = [(t[1][0], t[1][1] + 1) for t in trigger_labels_with_spans]
            batch_trigger_spans.append(trigger_spans)

        return batch_trigger_spans

    @staticmethod
    def _assert_target_seq_len(seq_len, target):
        """
        In some batches the longest sentence does not include any entities.
        This results in a target tensor, which is not padded to the full seq length.
        """
        batch_size, target_seq_len, num_spans = target.size()
        if seq_len == target_seq_len:
            return target
        else:
            missing_padding = seq_len - target_seq_len
            padding_size = (batch_size, missing_padding, num_spans)
            padding_tensor = torch.full(size=padding_size,
                                        fill_value=-1,
                                        dtype=target.dtype,
                                        device=target.device)
            return torch.cat([target, padding_tensor], dim=1)

    @staticmethod
    def _cross_entropy_focal_loss(logits, target, target_mask, gamma=None) -> torch.Tensor:
        if gamma:
            log_probs = torch.log_softmax(logits, dim=1)
            true_probs = log_probs.gather(dim=1, index=target.unsqueeze(1)).exp()
            true_probs = true_probs.view(*target.size())
            focal_factor = (1.0 - true_probs) ** gamma
            loss_unreduced = F.nll_loss(log_probs, target, reduction='none')
            loss_unreduced *= focal_factor
        else:
            loss_unreduced = F.cross_entropy(logits, target, reduction='none')
        masked_loss = loss_unreduced * target_mask
        batch_size = target.size(0)
        loss_per_batch = masked_loss.view(batch_size, -1).sum(dim=1)
        mask_per_batch = target_mask.view(batch_size, -1).sum()
        return (loss_per_batch / mask_per_batch).sum() / batch_size
