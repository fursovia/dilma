from typing import Tuple

import torch
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.registrable import Registrable
from allennlp.data import TextFieldTensors


MASK_TOKEN = "@@MASK@@"


class TokensMasker(Registrable):
    # https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L111
    skip_index = -100

    def __init__(self, vocab: Vocabulary, mlm_probability: float = 0.15, namespace: str = "tokens") -> None:
        self.vocab = vocab
        self.mlm_probability = mlm_probability
        self.pad_idx = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, namespace)
        self.mask_idx = self.vocab.get_token_index(MASK_TOKEN, namespace)
        self.ovv_idx = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, namespace)
        assert self.ovv_idx != self.mask_idx, f"Add `{MASK_TOKEN}` to your vocab"
        self.vocab_size = self.vocab.get_vocab_size(namespace)

    def mask_tokens(self, inputs: TextFieldTensors) -> Tuple[TextFieldTensors, TextFieldTensors]:

        masked_inputs = dict()
        masked_targets = dict()
        for text_field_name, text_field in inputs.items():
            masked_inputs[text_field_name] = dict()
            masked_targets[text_field_name] = dict()
            for key, tokens in text_field.items():
                labels = tokens.clone()
                probability_matrix = torch.full(labels.shape, self.mlm_probability)

                padding_mask = labels.eq(self.pad_idx)
                probability_matrix.masked_fill_(padding_mask, value=0.0)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                labels[~masked_indices] = self.skip_index  # We only compute loss on masked tokens

                # 80% of the time, we replace masked input tokens with mask_idx
                indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
                tokens[indices_replaced] = self.mask_idx

                # 10% of the time, we replace masked input tokens with random word
                indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
                tokens[indices_random] = random_words[indices_random]

                # The rest of the time (10% of the time) we keep the masked input tokens unchanged
                masked_inputs[text_field_name][key] = tokens
                masked_targets[text_field_name][key] = labels
        return masked_inputs, masked_targets

# We can't decorate `TokensMasker` with `TokensMasker.register()`, because `Model` hasn't been defined yet.  So we
# put this down here.
TokensMasker.register("tokens_masker")(TokensMasker)
