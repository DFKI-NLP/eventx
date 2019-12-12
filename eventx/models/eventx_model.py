from statistics import mean
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, TimeDistributed, TokenEmbedder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, FBetaMeasure
from overrides import overrides
from torch.nn import Linear, Parameter

from eventx.util import MicroFBetaMeasure
from eventx import NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL


@Model.register('eventx-model')
class EventxModel(Model):

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
        self.trigger_to_hidden = Linear(self.encoder.get_output_dim() + self.trigger_embedder.get_output_dim(),
                                        self.hidden_dim)
        self.entities_to_hidden = Linear(self.encoder.get_output_dim() + self.entity_embedder.get_output_dim(),
                                         self.hidden_dim)
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
                entity_labels: torch.LongTensor,
                entity_spans: torch.LongTensor,
                triggers: torch.LongTensor = None,
                arg_roles: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        embedded_tokens = self.text_field_embedder(tokens)
        text_mask = get_text_field_mask(tokens)
        encoded_tokens = self.encoder(embedded_tokens, text_mask)

        trigger_logits = self.trigger_projection(encoded_tokens)
        trigger_probabilities = F.softmax(trigger_logits, dim=-1)
        trigger_predictions = trigger_logits.argmax(dim=-1)
        output_dict = {"trigger_logits": trigger_logits,
                       "trigger_probabilities": trigger_probabilities}

        if triggers is not None:
            self.trigger_accuracy(trigger_logits, triggers, text_mask.float())
            self.trigger_f1(trigger_logits, triggers, text_mask.float())
            loss = sequence_cross_entropy_with_logits(logits=trigger_logits,
                                                      targets=triggers,
                                                      weights=text_mask,
                                                      gamma=self.trigger_gamma)
            output_dict["triggers_loss"] = loss
            output_dict["loss"] = loss

        embedded_trigger_labels = self.trigger_embedder(trigger_predictions)
        triggers_hidden = self.trigger_to_hidden(torch.cat([encoded_tokens, embedded_trigger_labels], dim=-1))  # B x L x H

        entity_spans_mask = (entity_spans[:, :, 0] >= 0).squeeze(-1).long()
        encoded_entities = self.span_extractor(sequence_tensor=encoded_tokens,
                                               span_indices=entity_spans,
                                               sequence_mask=text_mask,
                                               span_indices_mask=entity_spans_mask)
        entity_label_mask = (entity_labels != -1)
        entity_labels = entity_labels * entity_label_mask
        embedded_entity_labels = self.entity_embedder(entity_labels)
        entities_hidden = self.entities_to_hidden(torch.cat([encoded_entities, embedded_entity_labels], dim=-1))  # B x E x H

        trigger = triggers_hidden.unsqueeze(2)  # Shape: B x L x 1 x H
        args = entities_hidden.unsqueeze(1)  # Shape: B x 1 x E x H

        trigger_arg = trigger + args + self.hidden_bias  # B x L x E x H
        role_activations = F.relu(trigger_arg)

        role_logits = self.hidden_to_roles(role_activations)  # B x L x E x R
        role_probabilities = torch.softmax(role_logits, dim=-1)
        output_dict['role_logits'] = role_logits
        output_dict['role_probabilities'] = role_probabilities

        if arg_roles is not None:
            arg_roles = self._assert_target_seq_len(seq_len=embedded_tokens.shape[1],
                                                    target=arg_roles)
            target_mask = (arg_roles != -1)
            target = arg_roles * target_mask
            self.role_accuracy(role_logits, target, target_mask.float())
            self.role_f1(role_logits, target, target_mask.float())

            # Masked batch-wise cross entropy loss, optionally with focal-loss
            role_logits_t = role_logits.permute(0, 3, 1, 2)
            if self.role_gamma:
                log_probs = torch.log_softmax(role_logits_t, dim=1)
                true_probs = log_probs.gather(dim=1, index=target.unsqueeze(1)).exp()
                true_probs = true_probs.view(*target.size())
                focal_factor = (1.0 - true_probs) ** self.role_gamma
                loss_unreduced = F.nll_loss(log_probs, target, reduction='none')
                loss_unreduced *= focal_factor
            else:
                loss_unreduced = F.cross_entropy(role_logits_t, target, reduction='none')
            masked_loss = loss_unreduced * target_mask
            batch_size = text_mask.shape[0]
            loss_per_batch = masked_loss.view(batch_size, -1).sum(dim=1)
            mask_per_batch = target_mask.view(batch_size, -1).sum()
            role_loss = (loss_per_batch / mask_per_batch).sum() / batch_size

            output_dict['role_loss'] = role_loss
            output_dict['loss'] += self.loss_weight * role_loss

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
