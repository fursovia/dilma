from enum import Enum
from typing import Optional, Union

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model

from .masked_copynet import (
    MaskedCopyNet,
    get_masked_copynet_with_attention,
    get_nonmasked_copynet_with_attention,
    get_nonmasked_copynet_without_attention
)
from .classification_model import (
    Classifier,
    get_classification_model,
    get_classification_model_copynet
)
from .deep_levenshtein import (
    DeepLevenshtein,
    get_deep_levenshtein,
    get_deep_levenshtein_attention,
    get_deep_levenshtein_copynet
)


class Task(str, Enum):
    NONMASKED_COPYNET = 'nonmasked_copynet'
    NONMASKED_COPYNET_WITH_ATTENTION = 'nonmasked_copynet_with_attention'
    MASKED_COPYNET_WITH_ATTNETION = 'masked_copynet_with_attention'
    CLASSIFICATION = 'classification'
    CLASSIFICATION_COPYNET = 'classification_copynet'
    DEEP_LEVENSHTEIN = 'deep_levenshtein'
    DEEP_LEVENSHTEIN_WITH_ATTENTION = 'deep_levenshtein_with_attention'
    DEEP_LEVENSHTEIN_COPYNET = 'deep_levenshtein_copynet'


def get_copynet_by_name(
        task: Task,
        vocab: Vocabulary,
        beam_size: int,
        max_decoding_steps: int
) -> MaskedCopyNet:
    if task == Task.NONMASKED_COPYNET:
        return get_nonmasked_copynet_without_attention(
            vocab,
            max_decoding_steps=max_decoding_steps,
            beam_size=beam_size
        )

    elif task == Task.NONMASKED_COPYNET_WITH_ATTENTION:
        return get_nonmasked_copynet_with_attention(
            vocab,
            max_decoding_steps=max_decoding_steps,
            beam_size=beam_size
        )

    elif task == Task.MASKED_COPYNET_WITH_ATTNETION:
        return get_masked_copynet_with_attention(
            vocab,
            max_decoding_steps=max_decoding_steps,
            beam_size=beam_size
        )
    else:
        raise NotImplementedError


def get_classifier_by_name(
        task: Task,
        num_classes: int,
        vocab: Optional[Vocabulary] = None,
        copynet: Optional[MaskedCopyNet] = None
) -> MaskedCopyNet:
    assert vocab is not None or copynet is not None
    if task == Task.CLASSIFICATION_COPYNET:
        return get_classification_model_copynet(copynet, num_classes)
    elif task == Task.CLASSIFICATION:
        return get_classification_model(vocab, num_classes)
    else:
        raise NotImplementedError


def get_deep_levenshtein_by_name(
        task: Task,
        vocab: Optional[Vocabulary] = None,
        copynet: Optional[MaskedCopyNet] = None
) -> DeepLevenshtein:
    assert vocab is not None or copynet is not None
    if task == Task.DEEP_LEVENSHTEIN:
        return get_deep_levenshtein(vocab)
    elif task == Task.DEEP_LEVENSHTEIN_WITH_ATTENTION:
        return get_deep_levenshtein_attention(vocab)
    elif task == Task.DEEP_LEVENSHTEIN_COPYNET:
        return get_deep_levenshtein_copynet(copynet)
    else:
        raise NotImplementedError


def get_model_by_name(
        task: Union[Task, str],
        vocab: Optional[Vocabulary] = None,
        num_classes: Optional[int] = None,
        beam_size: Optional[int] = None,
        max_decoding_steps: Optional[int] = None,
        copynet: Optional[MaskedCopyNet] = None
) -> Model:
    if 'classification' in task:
        model = get_classifier_by_name(task, num_classes=num_classes, vocab=vocab, copynet=copynet)
    elif 'levenshtein' in task:
        model = get_deep_levenshtein_by_name(task, vocab=vocab, copynet=copynet)
    elif 'masked' in task:
        model = get_copynet_by_name(task, vocab=vocab, beam_size=beam_size, max_decoding_steps=max_decoding_steps)
    else:
        raise NotImplementedError
    return model
