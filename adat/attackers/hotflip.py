from typing import List

from allennlp.predictors.predictor import Predictor
from allennlp.interpret.attackers import Hotflip

from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN

from adat.tokens_masker import MASK_TOKEN

TO_DROP_TOKENS = [DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN, MASK_TOKEN, "<START>", "<END>"]


class HotFlipFixed(Hotflip):
    def __init__(self,
                 predictor: Predictor,
                 vocab_namespace: str = "tokens",
                 max_tokens: int = 20000) -> None:
        super().__init__(predictor, vocab_namespace, max_tokens)
        self.invalid_replacement_indices: List[int] = []
        for i in self.vocab._index_to_token[self.namespace]:
            if self.vocab._index_to_token[self.namespace][i] in TO_DROP_TOKENS:
                self.invalid_replacement_indices.append(i)
