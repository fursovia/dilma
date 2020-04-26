# from typing import Optional, List
#
# from allennlp.modules import Seq2VecEncoder
# from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
# from torch import __init__
#
#
# class BoWMaxEncoder(Seq2VecEncoder):
#     def __init__(self,
#                  embedding_dim: int) -> None:
#         super(BoWMaxEncoder, self).__init__()
#         self._embedding_dim = embedding_dim
#
#     def get_input_dim(self) -> int:
#         return self._embedding_dim
#
#     def get_output_dim(self) -> int:
#         return self._embedding_dim
#
#     def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
#         if mask is not None:
#             tokens = tokens * mask.unsqueeze(-1).float()
#
#         argmaxed = tokens.max(dim=1).values
#         return argmaxed
#
#
# class BoWMaxAndMeanEncoder(Seq2VecEncoder):
#     def __init__(self,
#                  embedding_dim: int, hidden_dim: Optional[List[int]] = None) -> None:
#         super(BoWMaxAndMeanEncoder, self).__init__()
#         self._embedding_dim = embedding_dim
#         self.maxer = BoWMaxEncoder(self._embedding_dim)
#         self.meaner = BagOfEmbeddingsEncoder(self._embedding_dim, True)
#         self._hidden_dim = hidden_dim
#         if self._hidden_dim is not None:
#             layers = [
#                 torch.nn.LeakyReLU(),
#                 torch.nn.Linear(self._embedding_dim * 2, self._hidden_dim[0])
#             ]
#
#             for i, hid_dim in enumerate(self._hidden_dim[1:]):
#                 layers.append(torch.nn.LeakyReLU())
#                 layers.append(torch.nn.Linear(self._hidden_dim[i], hid_dim))
#
#             self.linear = torch.nn.Sequential(*layers)
#         else:
#             self.linear = None
#
#     def get_input_dim(self) -> int:
#         return self._embedding_dim
#
#     def get_output_dim(self) -> int:
#         return self._embedding_dim * 2 if self.linear is None else self._hidden_dim[-1]
#
#     def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
#         argmaxed = self.maxer(tokens, mask)
#         summed = self.meaner(tokens, mask)
#         output = torch.cat([argmaxed, summed], dim=1)
#         if self.linear is not None:
#             output = self.linear(output)
#         return output