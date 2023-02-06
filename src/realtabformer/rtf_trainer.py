import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from torch import nn
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    logging,
)
from transformers.integrations import is_fairscale_available
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption
from transformers.utils import is_sagemaker_mp_enabled

logger = logging.get_logger(__name__)


class SaveEpochEndCallback(TrainerCallback):
    """This callback forces a checkpoint save at each epoch end."""

    def __init__(self, save_epochs: int = None) -> None:
        super().__init__()

        self.save_epochs = save_epochs

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.save_epochs is not None:
            control.should_save = math.ceil(state.epoch) % self.save_epochs == 0
        else:
            control.should_save = True

        return control


class ResumableTrainer(Trainer):
    """This trainer makes the scheduler consistent over pauses
    in the training. The scheduler should return values similar
    to when a training is done either intermittently or continuously
    over the `target_epochs`.
    """

    def __init__(
        self,
        target_epochs: int = None,
        save_epochs: int = None,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = None,
    ):
        # Declare here for typing
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None

        if callbacks is None:
            callbacks = []

        callbacks.append(SaveEpochEndCallback(save_epochs=save_epochs))

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.target_epochs = target_epochs

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """

        if self.lr_scheduler is None:
            if self.target_epochs is not None:
                # Compute the max_steps based from the
                # `target_epochs`.
                train_dataloader = self.get_train_dataloader()
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = (
                    len_dataloader // self.args.gradient_accumulation_steps
                )
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

                max_steps = math.ceil(self.target_epochs * num_update_steps_per_epoch)
                num_training_steps = max_steps

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

        return self.lr_scheduler


class FrozenSeq2SeqTrainer(Seq2SeqTrainer):
    """This trainer excludes all parameters that have
    `.requires_grad=False` set.
    """

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    # Add here the `p.requires_grad` condition
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    # Add here the `p.requires_grad` condition
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                # Do the import here...
                if is_fairscale_available():
                    from fairscale.optim import OSS

                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs
                )
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum(
                                {
                                    p.data_ptr(): p.numel() for p in module.parameters()
                                }.values()
                            )
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(
                                module, "weight", {"optim_bits": 32}
                            )
                            logger.debug(
                                f"bitsandbytes: will optimize {module} in fp32"
                            )
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            # Do the import here...
            import smdistributed.modelparallel.torch as smp

            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
