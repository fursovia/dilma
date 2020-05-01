from typing import Tuple

import torch
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.common.registrable import Registrable
from allennlp.data import TextFieldTensors


MASK_TOKEN = "@@MASK@@"


# TODO: should take start/end/unknown tokens into account
class TokensMasker(Registrable):
    # https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L111
    def __init__(
            self, vocab: Vocabulary,
            mask_probability: float = 0.3,
            replace_probability: float = 0.1,
            namespace: str = "tokens"
    ) -> None:
        self.vocab = vocab
        self.mask_probability = mask_probability
        self.replace_probability = replace_probability
        self.mask_idx = self.vocab.get_token_index(MASK_TOKEN, namespace)
        ovv_idx = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, namespace)
        assert ovv_idx != self.mask_idx, f"Add `{MASK_TOKEN}` to your vocab"
        self.vocab_size = self.vocab.get_vocab_size(namespace)

    def mask_tokens(self, inputs: TextFieldTensors) -> Tuple[TextFieldTensors, TextFieldTensors]:

        masked_inputs = dict()
        masked_targets = dict()
        for text_field_name, text_field in inputs.items():
            masked_inputs[text_field_name] = dict()
            masked_targets[text_field_name] = dict()
            for key, tokens in text_field.items():
                labels = tokens.clone()

                indices_masked = torch.bernoulli(
                    torch.full(labels.shape, self.mask_probability, device=tokens.device)
                ).bool()
                tokens[indices_masked] = self.mask_idx

                indices_random = torch.bernoulli(
                    torch.full(labels.shape, self.replace_probability, device=tokens.device)
                ).bool() & ~indices_masked
                random_tokens = torch.randint(
                    low=1,
                    high=self.vocab_size,
                    size=labels.shape,
                    dtype=torch.long,
                    device=tokens.device
                )
                tokens[indices_random] = random_tokens[indices_random]

                masked_inputs[text_field_name][key] = tokens
                masked_targets[text_field_name][key] = labels
        return masked_inputs, masked_targets


# We can't decorate `TokensMasker` with `TokensMasker.register()`, because `Model` hasn't been defined yet.  So we
# put this down here.
TokensMasker.register("tokens_masker")(TokensMasker)
