from copy import deepcopy
from typing import List

import numpy
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.interpret.attackers import utils
from allennlp.nn import util
from allennlp.predictors.predictor import Predictor
from allennlp.interpret.attackers import Hotflip

from adat.tokens_masker import MASK_TOKEN

DEFAULT_IGNORE_TOKENS = ["@@NULL@@", ".", ",", ";", "!", "?", "[MASK]",
                         "[SEP]", "[CLS]", MASK_TOKEN, "<START>", "<END>"]
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

    def attack_from_json(
        self,
        inputs: JsonDict,
        input_field_to_attack: str = "tokens",
        grad_input_field: str = "grad_input_1",
        ignore_tokens: List[str] = None,
        target: JsonDict = None,
    ) -> JsonDict:
        if self.embedding_matrix is None:
            self.initialize()
        ignore_tokens = DEFAULT_IGNORE_TOKENS if ignore_tokens is None else ignore_tokens

        sign = -1 if target is None else 1
        instance = self.predictor._json_to_instance(inputs)
        if target is None:
            output_dict = self.predictor._model.forward_on_instance(instance)
        else:
            output_dict = target

        original_instances = self.predictor.predictions_to_labeled_instances(instance, output_dict)

        original_text_field: TextField = original_instances[0][  # type: ignore
            input_field_to_attack
        ]
        original_tokens = deepcopy(original_text_field.tokens)

        final_tokens = []
        for instance in original_instances:
            fields_to_compare = utils.get_fields_to_compare(inputs, instance, input_field_to_attack)

            text_field: TextField = instance[input_field_to_attack]  # type: ignore
            grads, outputs = self.predictor.get_gradients([instance])

            flipped: List[int] = [0, len(text_field.tokens) + 1]
            for index, token in enumerate(text_field.tokens):
                if token.text in ignore_tokens:
                    flipped.append(index + 1)

            while True:
                grad = grads[grad_input_field][0]
                grads_magnitude = [g.dot(g) for g in grad]

                for index in flipped:
                    grads_magnitude[index] = -1

                index_of_token_to_flip = numpy.argmax(grads_magnitude)
                index_of_token_to_flip -= 1
                if grads_magnitude[index_of_token_to_flip] == -1:
                    # If we've already flipped all of the tokens, we give up.
                    break
                flipped.append(index_of_token_to_flip)

                text_field_tensors = text_field.as_tensor(text_field.get_padding_lengths())
                input_tokens = util.get_token_ids_from_text_field_tensors(text_field_tensors)
                original_id_of_token_to_flip = input_tokens[index_of_token_to_flip]

                # Get new token using taylor approximation.
                new_id = self._first_order_taylor(
                    grad[index_of_token_to_flip], original_id_of_token_to_flip, sign
                )

                new_token = Token(
                    self.vocab._index_to_token[self.namespace][new_id]
                )  # type: ignore
                text_field.tokens[index_of_token_to_flip] = new_token
                instance.indexed = False

                grads, outputs = self.predictor.get_gradients([instance])  # predictions
                for key, output in outputs.items():
                    if isinstance(output, torch.Tensor):
                        outputs[key] = output.detach().cpu().numpy().squeeze()
                    elif isinstance(output, list):
                        outputs[key] = output[0]

                labeled_instance = self.predictor.predictions_to_labeled_instances(
                    instance, outputs
                )[0]

                has_changed = utils.instance_has_changed(labeled_instance, fields_to_compare)
                if target is None and has_changed:
                    break
                if target is not None and not has_changed:
                    break

            tokens_to_add = []
            for token in text_field.tokens:
                if token.text not in ["<START>", "<END>"]:
                    tokens_to_add.append(token)

            final_tokens.append(tokens_to_add)

        return sanitize({"final": final_tokens, "original": original_tokens, "outputs": outputs})
