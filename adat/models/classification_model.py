# from typing import Dict
#
# import torch
# from allennlp.data import Vocabulary
# from allennlp.models import BasicClassifier
# from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
# from allennlp.training.metrics import Auc, F1Measure
# from allennlp.nn.util import get_text_field_mask
#
#
# @BasicClassifier.register(name="classifier_with_metrics")
# class Classifier(BasicClassifier):
#     def __init__(self,
#                  vocab: Vocabulary,
#                  text_field_embedder: TextFieldEmbedder,
#                  seq2vec_encoder: Seq2VecEncoder,
#                  seq2seq_encoder: Seq2SeqEncoder = None,
#                  num_labels: int = None) -> None:
#
#         super().__init__(vocab, text_field_embedder, seq2vec_encoder, seq2seq_encoder, num_labels=num_labels)
#         if self._num_labels == 2:
#             self._auc = Auc()
#             self._f1 = F1Measure(1)
#
#     def forward(self,
#                 tokens: Dict[str, torch.LongTensor],
#                 label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
#         embedded_text = self._text_field_embedder(tokens)
#         mask = get_text_field_mask(tokens).float()
#         if self._seq2seq_encoder:
#             embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)
#
#         embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)
#
#         if self._dropout:
#             embedded_text = self._dropout(embedded_text)
#
#         logits = self._classification_layer(embedded_text)
#         probs = torch.nn.functional.softmax(logits, dim=-1)
#
#         output_dict = {"logits": logits, "probs": probs}
#
#         if label is not None:
#             loss = self._loss(logits, label.long().view(-1))
#             output_dict["loss"] = loss
#             self._accuracy(logits, label)
#
#         if label is not None and self._num_labels == 2:
#             self._auc(output_dict['probs'][:, 1], label.long().view(-1))
#             self._f1(output_dict['probs'], label.long().view(-1))
#
#         return output_dict
#
#     def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#         metrics = super().get_metrics(reset)
#         if self._num_labels == 2:
#             metrics.update({'auc': self._auc.get_metric(reset), 'f1': self._f1.get_metric(reset)[2]})
#         return metrics
