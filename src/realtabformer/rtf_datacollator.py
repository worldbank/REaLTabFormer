from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class RelationalDataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Adopted from the DataCollatorForSeq2Seq:
     https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/data/data_collator.py#L510

    Args:
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(label) for label in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = feature["labels"] + remainder
                else:
                    # Pad always at the right.
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)

        labels = [feature["labels"] for feature in features]
        input_ids = [feature["input_ids"] for feature in features]

        if return_tensors == "np":
            labels = np.vstack(labels)
            input_ids = np.vstack(input_ids)
        elif return_tensors == "pt":
            labels = torch.vstack([torch.tensor(label) for label in labels])
            input_ids = torch.vstack([torch.tensor(ii) for ii in input_ids])
        elif return_tensors == "tf":
            raise ValueError("Tensorflow tensor is not supported yet.")

        return dict(
            labels=labels,
            input_ids=input_ids,
        )
