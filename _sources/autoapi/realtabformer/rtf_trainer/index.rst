:py:mod:`realtabformer.rtf_trainer`
===================================

.. py:module:: realtabformer.rtf_trainer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   realtabformer.rtf_trainer.SaveEpochEndCallback
   realtabformer.rtf_trainer.ResumableTrainer
   realtabformer.rtf_trainer.FrozenSeq2SeqTrainer




Attributes
~~~~~~~~~~

.. autoapisummary::

   realtabformer.rtf_trainer.logger


.. py:data:: logger

   

.. py:class:: SaveEpochEndCallback(save_epochs: int = None)

   Bases: :py:obj:`transformers.TrainerCallback`

   This callback forces a checkpoint save at each epoch end.

   .. py:method:: on_epoch_end(args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs)



.. py:class:: ResumableTrainer(target_epochs: int = None, save_epochs: int = None, model: Union[transformers.PreTrainedModel, torch.nn.Module] = None, args: transformers.TrainingArguments = None, data_collator: Optional[transformers.DataCollator] = None, train_dataset: Optional[datasets.Dataset] = None, eval_dataset: Optional[datasets.Dataset] = None, tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None, model_init: Callable[[], transformers.PreTrainedModel] = None, compute_metrics: Optional[Callable[[transformers.EvalPrediction], Dict]] = None, callbacks: Optional[List[transformers.TrainerCallback]] = None, optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None), preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None)

   Bases: :py:obj:`transformers.Trainer`

   This trainer makes the scheduler consistent over pauses
   in the training. The scheduler should return values similar
   to when a training is done either intermittently or continuously
   over the `target_epochs`.

   .. py:method:: create_scheduler(num_training_steps: int, optimizer: torch.optim.Optimizer = None) -> torch.optim.lr_scheduler.LambdaLR

      Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
      passed as an argument.
      :param num_training_steps: The number of training steps to do.
      :type num_training_steps: int



.. py:class:: FrozenSeq2SeqTrainer

   Bases: :py:obj:`transformers.Seq2SeqTrainer`

   This trainer excludes all parameters that have
   `.requires_grad=False` set.

   .. py:method:: create_optimizer()

      Setup the optimizer.
      We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
      Trainer's init through `optimizers`, or subclass and override this method in a subclass.



