# from typing import Dict, List, Tuple
#
# import numpy
# from overrides import overrides
# import torch
# import torch.nn.functional as F
# from torch.nn.modules.linear import Linear
# from torch.nn.modules.rnn import LSTMCell
#
# from allennlp.common.util import START_SYMBOL, END_SYMBOL
# from allennlp.data.vocabulary import Vocabulary
# from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
# from allennlp.models.model import Model
# from allennlp.nn import util
# from allennlp.nn.beam_search import BeamSearch
# from allennlp.training.metrics import BLEU
#
#
# @Model.register(name="masked_copynet")
# class MaskedCopyNet(Model):
#
#     def __init__(self,
#                  vocab: Vocabulary,
#                  embedder: TextFieldEmbedder,
#                  encoder: Seq2SeqEncoder,
#                  max_decoding_steps: int,
#                  attention: Attention = None,
#                  mask_embedder: TextFieldEmbedder = None,
#                  mask_attention: Attention = None,
#                  beam_size: int = None,
#                  target_namespace: str = "tokens",
#                  scheduled_sampling_ratio: float = 0.,
#                  use_bleu: bool = True) -> None:
#         super().__init__(vocab)
#         self._target_namespace = target_namespace
#         self._scheduled_sampling_ratio = scheduled_sampling_ratio
#
#         # We need the start symbol to provide as the input at the first timestep of decoding, and
#         # end symbol as a way to indicate the end of the decoded sequence.
#         self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
#         self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
#
#         if use_bleu:
#             pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)
#             self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
#         else:
#             self._bleu = None
#
#         # At prediction time, we use a beam search to find the most likely sequence of target tokens.
#         beam_size = beam_size or 1
#         self._max_decoding_steps = max_decoding_steps
#         self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)
#
#         # Dense embedding of source vocab tokens.
#         self._embedder = embedder
#         self._mask_embedder = mask_embedder
#
#         # Encodes the sequence of source embeddings into a sequence of hidden states.
#         self._encoder = encoder
#
#         num_classes = self.vocab.get_vocab_size(self._target_namespace)
#
#         # Attention mechanism applied to the encoder output for each step.
#         self._attention = attention
#         self._mask_attention = mask_attention
#
#         # Dense embedding of vocab words in the target space.
#         target_embedding_dim = self._embedder.get_output_dim()
#
#         # Decoder output dim needs to be the same as the encoder output dim since we initialize the
#         # hidden state of the decoder with the final hidden state of the encoder.
#         self._encoder_output_dim = self._encoder.get_output_dim()
#         self._decoder_output_dim = self._encoder_output_dim
#
#         if self._attention:
#             # If using attention, a weighted average over encoder outputs will be concatenated
#             # to the previous target embedding to form the input to the decoder at each
#             # time step.
#             self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
#         else:
#             # Otherwise, the input to the decoder is just the previous target embedding.
#             self._decoder_input_dim = target_embedding_dim
#
#         if self._mask_attention:
#             self._decoder_input_dim += self._mask_embedder.get_output_dim()
#
#         # We'll use an LSTM cell as the recurrent cell that produces a hidden state
#         # for the decoder at each time step.
#         self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
#
#         # We project the hidden state from the decoder into the output vocabulary space
#         # in order to get log probabilities of each target token, at each time step.
#         self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)
#
#     def take_step(self,
#                   last_predictions: torch.Tensor,
#                   state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         # shape: (group_size, num_classes)
#         output_projections, state = self._prepare_output_projections(last_predictions, state)
#
#         # shape: (group_size, num_classes)
#         class_log_probabilities = F.log_softmax(output_projections, dim=-1)
#
#         return class_log_probabilities, state
#
#     @overrides
#     def forward(self,  # type: ignore
#                 source_tokens: Dict[str, torch.LongTensor],
#                 target_tokens: Dict[str, torch.LongTensor] = None,
#                 mask_tokens: Dict[str, torch.LongTensor] = None,
#                 **kwargs) -> Dict[str, torch.Tensor]:
#         del kwargs
#         assert mask_tokens is not None or self._mask_embedder is None, \
#             'You must pass `mask_tokens` when `mask_embedder` is not None'
#         state = self.encode(source_tokens, mask_tokens)
#
#         if target_tokens:
#             state = self.init_decoder_state(state)
#             output_dict = self._forward_loop(state, target_tokens)
#         else:
#             output_dict = {}
#
#         if not self.training:
#             state = self.init_decoder_state(state)
#             predictions = self.beam_search(state)
#             output_dict.update(predictions)
#             if target_tokens and self._bleu:
#                 # shape: (batch_size, beam_size, max_sequence_length)
#                 top_k_predictions = output_dict["predictions"]
#                 # shape: (batch_size, max_predicted_sequence_length)
#                 best_predictions = top_k_predictions[:, 0, :]
#                 self._bleu(best_predictions, target_tokens["tokens"])
#
#         return output_dict
#
#     @overrides
#     def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         predicted_indices = output_dict["predictions"]
#         if not isinstance(predicted_indices, numpy.ndarray):
#             predicted_indices = predicted_indices.detach().cpu().numpy()
#         all_predicted_tokens = []
#         for i, indices in enumerate(predicted_indices):
#             curr_predictions = []
#             for ind in indices:
#                 ind = list(ind)
#                 # Collect indices till the first end_symbol
#                 if self._end_index in ind:
#                     ind = ind[:ind.index(self._end_index)]
#                 predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
#                                     for x in ind]
#                 curr_predictions.append(predicted_tokens)
#             all_predicted_tokens.append(curr_predictions)
#         output_dict["predicted_tokens"] = all_predicted_tokens  # [batch_size, k, num_decoding_steps]
#         return output_dict
#
#     def encode(self, source_tokens: Dict[str, torch.Tensor],
#                mask_tokens: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
#         # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
#         embedded_input = self._embedder(source_tokens)
#         # shape: (batch_size, max_input_sequence_length)
#         source_mask = util.get_text_field_mask(source_tokens)
#         # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
#         encoder_outputs = self._encoder(embedded_input, source_mask)
#         state = {
#             "source_mask": source_mask,
#             "encoder_outputs": encoder_outputs
#         }
#
#         if mask_tokens is not None and self._mask_embedder is not None:
#             embedded_input = self._mask_embedder(mask_tokens)
#             masker_mask = util.get_text_field_mask(mask_tokens)
#             state.update(
#                 {
#                     "mask_source_mask": masker_mask,
#                     "mask_encoder_outputs": embedded_input
#                 }
#             )
#         return state
#
#     def init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         batch_size = state["source_mask"].size(0)
#         # shape: (batch_size, encoder_output_dim)
#         final_encoder_output = util.get_final_encoder_states(
#                 state["encoder_outputs"],
#                 state["source_mask"],
#                 self._encoder.is_bidirectional())
#         # Initialize the decoder hidden state with the final output of the encoder.
#         # shape: (batch_size, decoder_output_dim)
#         state["decoder_hidden"] = final_encoder_output
#         # shape: (batch_size, decoder_output_dim)
#         state["decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self._decoder_output_dim)
#         return state
#
#     def _forward_loop(self,
#                       state: Dict[str, torch.Tensor],
#                       target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
#         # shape: (batch_size, max_input_sequence_length)
#         source_mask = state["source_mask"]
#
#         batch_size = source_mask.size()[0]
#
#         if target_tokens:
#             # shape: (batch_size, max_target_sequence_length)
#             targets = target_tokens["tokens"]
#
#             _, target_sequence_length = targets.size()
#
#             # The last input from the target is either padding or the end symbol.
#             # Either way, we don't have to process it.
#             num_decoding_steps = target_sequence_length - 1
#         else:
#             num_decoding_steps = self._max_decoding_steps
#
#         # Initialize target predictions with the start index.
#         # shape: (batch_size,)
#         last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)
#
#         step_logits: List[torch.Tensor] = []
#         step_predictions: List[torch.Tensor] = []
#         for timestep in range(num_decoding_steps):
#             if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
#                 # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
#                 # during training.
#                 # shape: (batch_size,)
#                 input_choices = last_predictions
#             elif not target_tokens:
#                 # shape: (batch_size,)
#                 input_choices = last_predictions
#             else:
#                 # shape: (batch_size,)
#                 input_choices = targets[:, timestep]
#
#             # shape: (batch_size, num_classes)
#             output_projections, state = self._prepare_output_projections(input_choices, state)
#
#             # list of tensors, shape: (batch_size, 1, num_classes)
#             step_logits.append(output_projections.unsqueeze(1))
#
#             # shape: (batch_size, num_classes)
#             class_probabilities = F.softmax(output_projections, dim=-1)
#
#             # shape (predicted_classes): (batch_size,)
#             _, predicted_classes = torch.max(class_probabilities, 1)
#
#             # shape (predicted_classes): (batch_size,)
#             last_predictions = predicted_classes
#
#             step_predictions.append(last_predictions.unsqueeze(1))
#
#         # shape: (batch_size, num_decoding_steps)
#         predictions = torch.cat(step_predictions, 1)
#
#         output_dict = {"predictions": predictions}
#
#         if target_tokens:
#             # shape: (batch_size, num_decoding_steps, num_classes)
#             logits = torch.cat(step_logits, 1)
#
#             # Compute loss.
#             target_mask = util.get_text_field_mask(target_tokens)
#             loss = self._get_loss(logits, targets, target_mask)
#             output_dict["loss"] = loss
#
#         return output_dict
#
#     def beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         batch_size = state["source_mask"].size()[0]
#         start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)
#
#         # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
#         # shape (log_probabilities): (batch_size, beam_size)
#
#         all_top_k_predictions, log_probabilities = self._beam_search.search(
#                 start_predictions, state, self.take_step)
#
#         output_dict = {
#                 "class_log_probabilities": log_probabilities,
#                 "predictions": all_top_k_predictions,
#         }
#         return output_dict
#
#     def _prepare_output_projections(self,
#                                     last_predictions: torch.Tensor,
#                                     state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         # shape: (group_size, max_input_sequence_length, encoder_output_dim)
#         encoder_outputs = state["encoder_outputs"]
#
#         # shape: (group_size, max_input_sequence_length)
#         source_mask = state["source_mask"]
#
#         # shape: (group_size, decoder_output_dim)
#         decoder_hidden = state["decoder_hidden"]
#
#         # shape: (group_size, decoder_output_dim)
#         decoder_context = state["decoder_context"]
#
#         # shape: (group_size, target_embedding_dim)
#         embedded_input = self._embedder({self._target_namespace: last_predictions})
#
#         if self._attention:
#             # shape: (group_size, encoder_output_dim)
#             attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)
#
#             # shape: (group_size, decoder_output_dim + target_embedding_dim)
#             decoder_input = torch.cat((attended_input, embedded_input), -1)
#         else:
#             # shape: (group_size, target_embedding_dim)
#             decoder_input = embedded_input
#
#         if self._mask_attention and self._mask_embedder:
#             mask_encoder_outputs = state["mask_encoder_outputs"]
#             mask_source_mask = state["mask_source_mask"]
#             mask_attended_input = self._prepare_mask_attended_input(
#                 decoder_hidden,
#                 mask_encoder_outputs,
#                 mask_source_mask
#             )
#             decoder_input = torch.cat((decoder_input, mask_attended_input), -1)
#
#         # shape (decoder_hidden): (batch_size, decoder_output_dim)
#         # shape (decoder_context): (batch_size, decoder_output_dim)
#         decoder_hidden, decoder_context = self._decoder_cell(
#                 decoder_input,
#                 (decoder_hidden, decoder_context))
#
#         state["decoder_hidden"] = decoder_hidden
#         state["decoder_context"] = decoder_context
#
#         # shape: (group_size, num_classes)
#         output_projections = self._output_projection_layer(decoder_hidden)
#
#         return output_projections, state
#
#     def _prepare_attended_input(self,
#                                 decoder_hidden_state: torch.LongTensor = None,
#                                 encoder_outputs: torch.LongTensor = None,
#                                 encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
#         encoder_outputs_mask = encoder_outputs_mask.float()
#         input_weights = self._attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)
#         attended_input = util.weighted_sum(encoder_outputs, input_weights)
#         return attended_input
#
#     def _prepare_mask_attended_input(self,
#                                      decoder_hidden_state: torch.LongTensor = None,
#                                      mask_encoder_outputs: torch.LongTensor = None,
#                                      mask_encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
#         encoder_outputs_mask = mask_encoder_outputs_mask.float()
#         input_weights = self._mask_attention(decoder_hidden_state, mask_encoder_outputs, encoder_outputs_mask)
#         attended_input = util.weighted_sum(mask_encoder_outputs, input_weights)
#         return attended_input
#
#     @staticmethod
#     def _get_loss(logits: torch.LongTensor,
#                   targets: torch.LongTensor,
#                   target_mask: torch.LongTensor) -> torch.Tensor:
#         # shape: (batch_size, num_decoding_steps)
#         relevant_targets = targets[:, 1:].contiguous()
#
#         # shape: (batch_size, num_decoding_steps)
#         relevant_mask = target_mask[:, 1:].contiguous()
#
#         return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
#
#     @overrides
#     def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#         all_metrics: Dict[str, float] = {}
#         if self._bleu and not self.training:
#             all_metrics.update(self._bleu.get_metric(reset=reset))
#         return all_metrics
#
