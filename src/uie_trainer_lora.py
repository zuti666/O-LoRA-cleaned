import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX

from optimer_sam import SAM, enable_running_stats, disable_running_stats

import h5py
import copy
from datetime import datetime 
from torch.nn import functional as F

def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    # logger.debug(f'model.config._name_or_path: {model.config._name_or_path}')
    
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:
            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
                # final_predictions.append(pred.strip())
    else:
        final_predictions = predictions

    return final_predictions


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control


class UIETrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        if getattr(self.args, "use_probe", False):
            # 尝试寻找 probe_head
            self.probe_head = getattr(self.model, "probe_head", None)
            if self.probe_head is None and hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
                self.probe_head = getattr(self.model.base_model.model, "probe_head", None)
            if self.probe_head is None:
                raise ValueError("Model does not have `probe_head` defined.")

            logger.info(f"Found probe head: {self.probe_head}")

        # if getattr(self.args, "use_probe", False):
        #     logger.info(f"Adding probe head in Trainer init")
        #     num_probe_classes = self.args.probe_num_classes
        #     hidden_size = self.model.config.hidden_size
        #     self.model.probe_head = nn.Linear(hidden_size, num_probe_classes)
        #     self.model.probe_head.to(self.args.device)  # 🔥 关键修复！把分类头移动到GPU

        #     # 冻结其他参数
        #     for name, param in self.model.named_parameters():
        #         if not name.startswith("probe_head"):
        #             param.requires_grad = False

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        logger.info(f"调用自定义 looping 函数进行训练，batch_size: {batch_size}")
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:

            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            # 然后交给 deepspeed，但不让它覆盖 optimizer
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )

            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine



            # ---- 原来的设置
            # deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
            #     self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            # )
            # self.model = deepspeed_engine.module
            # self.model_wrapped = deepspeed_engine
            # self.deepspeed = deepspeed_engine
            # self.optimizer = optimizer
            # self.lr_scheduler = lr_scheduler
            #$ ---- 原来的设置

        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    (total_batched_samples % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        # if self.args.use_probe:
                        if self.args.train_method == 'use_probe':
                            tr_loss_step = self.training_step(model, inputs)
                        elif self.args.train_method == "lora":
                        # 修改进行梯度更新的地方，替换掉原来的函数调用
                        # -------------------
                            model.train()
                            enable_running_stats(model)
                            inputs = self._prepare_inputs(inputs)

                            if is_sagemaker_mp_enabled():
                                loss_mb = smp_forward_backward(
                                    model, inputs, self.args.gradient_accumulation_steps)
                                return loss_mb.reduce_mean().detach().to(self.args.device)
                            
                            # self.print_trainable_parameters(model)
                            with self.compute_loss_context_manager():
                                loss = self.compute_loss(model, inputs)

                            logger.info(f'training_step compute_loss: {loss.item():.4f}')
                            

                            if self.args.n_gpu > 1:
                                loss = loss.mean()

                            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                                loss = loss / self.args.gradient_accumulation_steps

                            ########################## Regularization ##########################
                            if self.args.lora_strategy.lower() == "olora":
                                orthogonal_loss = 0.
                                for name, param in model.named_parameters():
                                    if "lora_A" in name:
                                        for name_, param_ in model.named_parameters():
                                            if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                                                orthogonal_loss += torch.abs(
                                                    torch.mm(param, param_.T)).sum()
                                                break

                                l2_loss = 0.
                                for name, param in model.named_parameters():
                                    if "loranew_" in name:
                                        l2_loss += torch.norm(param, p=2)

                                lamda_1 = self.args.lamda_1
                                lamda_2 = self.args.lamda_2

                                logger.info(f"orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}")

                                loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
                            elif self.args.lora_strategy.lower() == "nlora":
                                l1_loss = 0.
                                loranew_A_params = {}
                                loranew_B_params = {}
                                for name, param in self.model.named_parameters():
                                    if "loranew_A" in name:
                                        loranew_A_params[name.split("loranew_A")[0]] = param
                                    elif "loranew_B" in name:
                                        loranew_B_params[name.split("loranew_B")[0]] = param

                                for key in loranew_A_params:
                                    if key in loranew_B_params:
                                        l1_loss += torch.norm(
                                            torch.mm(loranew_A_params[key], loranew_B_params[key]), p=1)

                                lamda_1 = self.args.lamda_1

                                logger.info(f"Nlora_loss: {l1_loss.item():.4f};   accuracy_loss: {loss.item():.4f}; λ1: {lamda_1};")

                                loss = loss + l1_loss * lamda_1
                            elif self.args.lora_strategy.lower() == "inclora":
                                logger.info(f"inclora accuracy_loss: {loss.item():.4f}")
                            elif self.args.lora_strategy.lower() == "lora_l2":
                                l2_loss = 0.
                                for name, param in model.named_parameters():
                                    if "loranew_" in name:
                                        l2_loss += torch.norm(param, p=2)

                                logger.info(f" l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f};  λ2: {lamda_2}")
                                lamda_2 = self.args.lamda_2
                                loss = loss + l2_loss * lamda_2
                            ######################################################################
                            logger.debug(f"sum_loss: {loss.item():.4f}")

                            if self.do_grad_scaling:
                                self.scaler.scale(loss).backward()
                            elif self.use_apex:
                                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            elif self.deepspeed:
                                # loss gets scaled under gradient_accumulation_steps in deepspeed
                                loss = self.deepspeed.backward(loss)
                            else:
                                loss.backward()

                            tr_loss_step = loss.detach()
                        #------------------------------

                else:
                    # if self.args.use_probe:
                    if self.args.train_method == 'use_probe':
                        tr_loss_step = self.training_step(model, inputs)
                    elif self.args.train_method == "lora":
                        # 修改进行梯度更新的地方，替换掉原来的函数调用
                        # -------------------
                        model.train()
                        enable_running_stats(model)
                        inputs = self._prepare_inputs(inputs)

                        if is_sagemaker_mp_enabled():
                            loss_mb = smp_forward_backward(
                                model, inputs, self.args.gradient_accumulation_steps)
                            return loss_mb.reduce_mean().detach().to(self.args.device)
                        
                        # self.print_trainable_parameters(model)
                        with self.compute_loss_context_manager():
                            loss = self.compute_loss(model, inputs)

                        logger.info(f'training_step compute_loss: {loss.item():.4f}')
                        

                        if self.args.n_gpu > 1:
                            loss = loss.mean()

                        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                            loss = loss / self.args.gradient_accumulation_steps

                        ########################### Regularization ##########################
                        if self.args.lora_strategy.lower() == "olora":
                            orthogonal_loss = 0.
                            for name, param in model.named_parameters():
                                if "lora_A" in name:
                                    for name_, param_ in model.named_parameters():
                                        if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                                            orthogonal_loss += torch.abs(
                                                torch.mm(param, param_.T)).sum()
                                            break

                            l2_loss = 0.
                            for name, param in model.named_parameters():
                                if "loranew_" in name:
                                    l2_loss += torch.norm(param, p=2)

                            lamda_1 = self.args.lamda_1
                            lamda_2 = self.args.lamda_2

                            logger.info(
                                f"orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}")

                            loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
                        elif self.args.lora_strategy.lower() == "nlora":
                            l1_loss = 0.
                            loranew_A_params = {}
                            loranew_B_params = {}
                            for name, param in self.model.named_parameters():
                                if "loranew_A" in name:
                                    loranew_A_params[name.split("loranew_A")[0]] = param
                                elif "loranew_B" in name:
                                    loranew_B_params[name.split("loranew_B")[0]] = param

                            for key in loranew_A_params:
                                if key in loranew_B_params:
                                    l1_loss += torch.norm(
                                        torch.mm(loranew_A_params[key], loranew_B_params[key]), p=1)

                            lamda_1 = self.args.lamda_1

                            logger.info(f"Nlora_loss: {l1_loss.item():.4f};   accuracy_loss: {loss.item():.4f}; λ1: {lamda_1};")

                            loss = loss + l1_loss * lamda_1
                        elif self.args.lora_strategy.lower() == "inclora":
                            logger.info(f"inclora accuracy_loss: {loss.item():.4f}")
                        elif self.args.lora_strategy.lower() == "lora_l2":
                            l2_loss = 0.
                            for name, param in model.named_parameters():
                                if "loranew_" in name:
                                    l2_loss += torch.norm(param, p=2)

                            logger.info(f"lora_l2 l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f};  λ2: {lamda_2}")
                            lamda_2 = self.args.lamda_2
                            loss = loss + l2_loss * lamda_2
                        ######################################################################
                        logger.debug(f"sum_loss: {loss.item():.4f}")
                        if self.do_grad_scaling:
                            self.scaler.scale(loss).backward()
                        elif self.use_apex:
                            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        elif self.deepspeed:
                            # loss gets scaled under gradient_accumulation_steps in deepspeed
                            loss = self.deepspeed.backward(loss)
                        else:
                            loss.backward()

                        tr_loss_step = loss.detach()
                    # -------------------




                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()


                # 真正进行梯度更新的地方
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    # 进行梯度裁剪
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:

                        logger.debug(f'args.max_grad_norm: {args.max_grad_norm}')
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)
                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    # elif is_torch_tpu_available():
                    #     if self.do_grad_scaling:
                    #         self.scaler.step(self.optimizer)
                    #         self.scaler.update()
                    #     else:
                    #         xm.optimizer_step(self.optimizer)
                    # elif self.do_grad_scaling:
                    #     scale_before = self.scaler.get_scale()
                    #     self.scaler.step(self.optimizer)
                    #     self.scaler.update()
                    #     scale_after = self.scaler.get_scale()
                    #     optimizer_was_run = scale_before <= scale_after
                    # else:
                    #     self.optimizer.step()
                    elif hasattr(self.optimizer, "first_step") and hasattr(self.optimizer, "second_step"):
                        # === For SAM Optimizer: first_step-second_step two phases ===
                  
                        
                        # first forward-backward
                        self.optimizer.first_step(zero_grad=True)

                        # second forward-backward
                        disable_running_stats(model)
                        with self.compute_loss_context_manager():
                            second_loss = self.compute_loss(model, inputs)

                        if self.args.n_gpu > 1:
                            second_loss = second_loss.mean()
                        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                            second_loss = second_loss / self.args.gradient_accumulation_steps

                        ########################### Regularization ##########################
                        if self.args.lora_strategy.lower() == "olora":
                            orthogonal_loss = 0.
                            for name, param in model.named_parameters():
                                if "lora_A" in name:
                                    for name_, param_ in model.named_parameters():
                                        if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                                            orthogonal_loss += torch.abs(
                                                torch.mm(param, param_.T)).sum()
                                            break

                            l2_loss = 0.
                            for name, param in model.named_parameters():
                                if "loranew_" in name:
                                    l2_loss += torch.norm(param, p=2)

                            lamda_1 = self.args.lamda_1
                            lamda_2 = self.args.lamda_2

                            logger.info(
                                f"orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; accuracy_loss: {second_loss.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}")

                            second_loss = second_loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
                        elif self.args.lora_strategy.lower() == "nlora":
                            l1_loss = 0.
                            loranew_A_params = {}
                            loranew_B_params = {}
                            for name, param in self.model.named_parameters():
                                if "loranew_A" in name:
                                    loranew_A_params[name.split("loranew_A")[0]] = param
                                elif "loranew_B" in name:
                                    loranew_B_params[name.split("loranew_B")[0]] = param

                            for key in loranew_A_params:
                                if key in loranew_B_params:
                                    l1_loss += torch.norm(
                                        torch.mm(loranew_A_params[key], loranew_B_params[key]), p=1)

                            lamda_1 = self.args.lamda_1

                            logger.info(f"Nlora_loss: {l1_loss.item():.4f};   accuracy_loss: {second_loss.item():.4f}; λ1: {lamda_1};")

                            second_loss = second_loss + l1_loss * lamda_1
                        elif self.args.lora_strategy.lower() == "inclora":
                            logger.info(f"inclora accuracy_loss: {second_loss.item():.4f}")
                        elif self.args.lora_strategy.lower() == "lora_l2":
                            l2_loss = 0.
                            for name, param in model.named_parameters():
                                if "loranew_" in name:
                                    l2_loss += torch.norm(param, p=2)

                            logger.info(f" l2_loss: {l2_loss.item():.4f}; accuracy_loss: {second_loss.item():.4f};  λ2: {lamda_2}")
                            lamda_2 = self.args.lamda_2
                            second_loss = second_loss + l2_loss * lamda_2
                        ######################################################################
                        logger.info(f'SAM training_step second_loss_sum: {second_loss.item():.4f}')
                        ######################################################################

                        # 第二次反向传播
                        if self.do_grad_scaling:
                            self.scaler.scale(second_loss).backward()
                            self.scaler.unscale_(self.optimizer)
                        elif self.use_apex:
                            with amp.scale_loss(second_loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        elif self.deepspeed:
                            # loss gets scaled under gradient_accumulation_steps in deepspeed
                            second_loss = self.deepspeed.backward(second_loss)
                        else:
                            second_loss.backward()

                        self.optimizer.second_step(zero_grad=True)


                    else:
                    # === Standard Optimizer ===
                        if is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()


                    

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        自定义优化器与学习率调度器的创建，兼容SAM。
        """
        if self.optimizer is None:

            # self.scaler = torch.cuda.amp.GradScaler()
            # self.do_grad_scaling = True  # 控制标志
            # optim_name = self.args.optim.lower()  # <<<<<< 改这里，统一从optim读取
            if self.args.optimizer_type:
                optim_name = self.args.optimizer_type.lower()
            logger.debug(
                f'Class ContinualTrainer def create_optimizer_and_scheduler() optim_name: {optim_name}')
            if optim_name == "sgd":
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.args.learning_rate)
            elif optim_name == "sam-sgd":
                self.optimizer = SAM(
                    self.model.parameters(),
                    base_optimizer=torch.optim.SGD,
                    rho=self.args.rho,
                    lr=self.args.learning_rate,
                )
            elif optim_name == "adamw_hf":
                from transformers import AdamW
                self.optimizer = AdamW(
                    self.model.parameters(), lr=self.args.learning_rate)
            elif optim_name == "sam-adamw_hf":
                from transformers import AdamW
                self.optimizer = SAM(
                    self.model.parameters(),
                    base_optimizer=AdamW,
                    rho=self.args.rho,
                    lr=self.args.learning_rate,
                )
            elif optim_name == "adam":
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.args.learning_rate)
            elif optim_name == "sam-adam":
                self.optimizer = SAM(
                    self.model.parameters(),
                    base_optimizer=torch.optim.Adam,
                    rho=self.args.rho,
                    lr=self.args.learning_rate,
                )

            else:
                raise ValueError(f"Unsupported optimizer type: {optim_name}")

        # if self.lr_scheduler is None:
        if 'sam' in optim_name:
            from transformers.optimization import get_constant_schedule
            self.lr_scheduler = get_constant_schedule(
                self.optimizer.base_optimizer)
            # self.lr_scheduler = LambdaLR(self.optimizer.base_optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            from transformers.optimization import get_constant_schedule
            self.lr_scheduler = get_constant_schedule(self.optimizer)

            # self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
    def print_trainable_parameters(self,model):
        total_params = 0
        trainable_params = 0
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"✅ {name}: {param.shape}")
            # else:
            #     print(f"❌ {name}: frozen")

        print(f"\nTotal parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%\n")


    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     自定义训练步骤，兼容SAM优化器和LoRA正则。
    #     """
    #     model.train()
    #     enable_running_stats(model)
    #     inputs = self._prepare_inputs(inputs)

    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(
    #             model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)
        
    #     # self.print_trainable_parameters(model)
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)

    #     logger.info(f'training_step compute_loss: {loss.item():.4f}')
        

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()

    #     if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #         loss = loss / self.args.gradient_accumulation_steps

    #     ########################### Regularization ##########################
    #     orthogonal_loss = 0.
    #     for name, param in model.named_parameters():
    #         if "lora_A" in name:
    #             for name_, param_ in model.named_parameters():
    #                 if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
    #                     orthogonal_loss += torch.abs(
    #                         torch.mm(param, param_.T)).sum()
    #                     break

    #     l2_loss = 0.
    #     for name, param in model.named_parameters():
    #         if "loranew_" in name:
    #             l2_loss += torch.norm(param, p=2)

    #     lamda_1 = self.args.lamda_1
    #     lamda_2 = self.args.lamda_2

    #     logger.info(
    #         f"orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; accuracy_loss: {loss.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}")

    #     loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2



        

    #     # 先分支处理 backward

    #     logger.debug(f'Class ContinualTrainer def training_step_sam2step()  backward')
    #     if self.do_grad_scaling:
    #         logger.debug(
    #             f'Class ContinualTrainer def training_step_()  update optim  self.do_grad_scaling{self.do_grad_scaling}')
    #         self.scaler.scale(loss).backward()
            
    #         #然后进行梯度裁剪 和SAM 优化器更新
    #         if is_torch_tpu_available():
    #             gradients = xm._fetch_gradients(self.optimizer)
    #             xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
    #         self.scaler.unscale_(self.optimizer)

           



    #     elif self.use_apex: 
    #         # 这里的 self.use_apex 是指是否使用 apex 进行混合精度训练
    #         logger.debug(
    #             f'Class ContinualTrainer def training_step_()  update optim  self.use_apex{self.use_apex}')
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     elif self.deepspeed:
    #         logger.debug(
    #             f'Class ContinualTrainer def training_step_sam2step()  update optim self.deepspeed')
    #         loss = self.deepspeed.backward(loss)
    #     else:

    #         logger.debug(
    #             f'Class ContinualTrainer def training_step  update optim else:  {self.args.optimizer_type.lower()}')
    #         loss.backward()

           

    #         if 'sam' in self.args.optimizer_type.lower():
    #             logger.debug( f'Class ContinualTrainer def training_step_sam2step()  update optim else:  rho :{self.args.rho}')
    #             # ==SAM== 包装一步更新

    #             # def closure():
    #             #     # 注意：此处 zero_grad 是必要的，因为 optimizer.step(closure) 预期 closure 内部自行清理梯度
    #             #     self.optimizer.zero_grad()

    #             #     outputs = model(**inputs)
    #             #     loss = outputs.loss

    #             #     # 多卡时取平均
    #             #     if self.args.n_gpu > 1:
    #             #         loss = loss.mean()

    #             #     # 如果有梯度累积，除以累积步数
    #             #     if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #             #         loss = loss / self.args.gradient_accumulation_steps

    #             #     # === 追加正则项：orthogonal loss + L2正则 ===
    #             #     orthogonal_loss = 0.0
    #             #     for name, param in model.named_parameters():
    #             #         if "lora_A" in name:
    #             #             for name_, param_ in model.named_parameters():
    #             #                 if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
    #             #                     orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum()
    #             #                     break

    #             #     l2_loss = 0.0
    #             #     for name, param in model.named_parameters():
    #             #         if "loranew_" in name:
    #             #             l2_loss += torch.norm(param, p=2)

    #             #     lamda_1 = getattr(self.args, "lamda_1", 0.5)
    #             #     lamda_2 = getattr(self.args, "lamda_2", 0.0)

    #             #     # 加到总loss中
    #             #     loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2

    #             #     # 标准 backward
    #             #     loss.backward()

    #             #     return loss

    #             # # === 优化器执行 closure ===
    #             # # ⚡启用 BN层 running_stats（SAM要求）
    #             # enable_running_stats(model)

    #             # # ⚡先手动执行一次 closure，保证第一次 forward/backward，累积梯度
    #             # loss = closure()

    #             # # ⚡由 optimizer 完成 first_step -> second closure -> second_step
    #             # self.optimizer.step(closure=closure)

    #             # # ⚡清空梯度
    #             # self.optimizer.zero_grad()

    #             # === SAM两步更新 ===
    #             if self.args.gradient_accumulation_steps > 1 and hasattr(model, "no_sync"):
    #                 with model.no_sync():
    #                     loss.backward()
    #             else:
    #                 loss.backward()

    #             self.optimizer.first_step(zero_grad=True)

    #             # second forward-backward
    #             disable_running_stats(model)
    #             with self.compute_loss_context_manager():
    #                 second_loss = self.compute_loss(model, inputs)

    #             if self.args.n_gpu > 1:
    #                 second_loss = second_loss.mean()
    #             if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #                 second_loss = second_loss / self.args.gradient_accumulation_steps

    #             orthogonal_loss2 = 0.
    #             for name, param in model.named_parameters():
    #                 if "lora_A" in name: 
    #                     for name_, param_ in model.named_parameters(): 
    #                         if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]: 
    #                             orthogonal_loss2 += torch.abs(
    #                                 torch.mm(param, param_.T)).sum()
    #                             break

    #             l2_loss2 = 0. 
    #             for name, param in model.named_parameters(): 
    #                 if "loranew_" in name: 
    #                     l2_loss2 += torch.norm(param, p=2) 

    #             logger.info( f"SAM second loss orthogonal_loss: {orthogonal_loss2.item():.4f}; l2_loss: {l2_loss2.item():.4f}; accuracy_loss: {second_loss.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}") 

    #             second_loss = second_loss + orthogonal_loss2 * lamda_1 + l2_loss2 * lamda_2 

    #             logger.info(f'SAM training_step second_loss_sum: {second_loss.item():.4f}')

    #             if self.args.gradient_accumulation_steps > 1 and hasattr(model, "no_sync"): 
    #                 with model.no_sync(): 
    #                     second_loss.backward() 
    #             else: 
    #                 second_loss.backward() 
    #             self.optimizer.second_step(zero_grad=True) 
 
    #         else: 
    #             loss.backward() 
    #             self.optimizer.step() 
    #             self.optimizer.zero_grad() 
        
    #     return loss.detach()
                
    

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        兼容 HuggingFace Trainer 默认损失逻辑 + Linear Probe 模式下特征提取 & 分类头损失。

        Linear probe 模式通过参数 `args.use_probe` 控制：
        - False：默认走原始 T5 损失（seq2seq / causal loss）
        - True：替换为 probe_head 分类器损失（线性分类头）

        还支持多种特征抽取方式（cls/eos/mean）通过 `args.probe_feature_mode` 控制。
        """
        # --- 默认 HuggingFace 损失逻辑（非 probe 模式） ---
        if not getattr(self.args, "use_probe", False):
            # 🔁 调用父类默认实现
            # 默认路径，使用 HF 的 loss 机制（适配 label_smoother、AMP、多卡等）
            logger.debug(f" run father loss")
            return super().compute_loss(model, inputs, return_outputs)
        
        else:

            # --- Linear Probe 模式 ---
            # real_model = unwrap_model(model)

            # 🔥 注意！！现在不需要自己取 hidden_state 手动处理了
            # 只需要 forward 返回 logits（在 base_model 的 forward 里已经加了 probe_head）
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                decoder_input_ids=inputs.get("decoder_input_ids", None),
                decoder_attention_mask=inputs.get("decoder_attention_mask", None),
                output_hidden_states=True,  # <<< ✨✨这里必须加✨✨
                return_dict=True,
            )

            logits = outputs.logits  # 🔥 直接拿 logits，不需要手动 probe_head(features)
            class_labels = inputs["class_labels"]  # 🔥 直接用 class_labels

            # CrossEntropy Loss
            loss = F.cross_entropy(logits, class_labels.view(-1))

            return (loss, {"logits": logits, "labels": class_labels}) if return_outputs else loss
                # # 🔥 核心：Linear Probe 使用 features + probe_head 分类
            # real_model =  unwrap_model(model)  # ✅ 解包，拿原始 model

            # # logger.debug(f" run Linear Probe loss")
            # # # === Linear Probe 模式 ===
            # # labels = inputs["labels"]
            # # decoder_input_ids = inputs.get("decoder_input_ids", None)
            # # --- 取输入 ---
            # class_labels = inputs["class_labels"]  # ✅ 使用整数编号
            # decoder_input_ids = inputs.get("decoder_input_ids", None)

            # # Forward 获取 encoder/decoder 表征
            # outputs = model(
            #     input_ids=inputs["input_ids"],
            #     attention_mask=inputs.get("attention_mask", None),
            #     decoder_input_ids=decoder_input_ids,
            #     output_hidden_states=True,
            #     return_dict=True
            # )

            # decoder_hidden_states = outputs.decoder_hidden_states[-1]  # (B, T, H)
            # decoder_attention_mask = inputs.get("decoder_attention_mask", None)

            # # --- 特征提取策略 ---
            # mode = getattr(self.args, "probe_feature_mode", "cls")

            # if mode == "cls":
            #     # 使用 decoder 的第一个 token 的 embedding（常用于 T5 等）
            #     features = decoder_hidden_states[:, 0]  # (B, H)

            # elif mode == "eos":
            #     # 使用 decoder 中 EOS token 的 embedding
            #     eos_token_id = getattr(model.config, "eos_token_id", 1)
            #     eos_mask = (decoder_input_ids == eos_token_id)  # (B, T)
            #     if eos_mask.sum(dim=1).min() == 0:
            #         raise ValueError("Some sequences do not contain an EOS token.")
            #     eos_idx = eos_mask.float().cumsum(dim=1).argmax(dim=1)  # (B,)
            #     features = decoder_hidden_states[torch.arange(decoder_hidden_states.size(0)), eos_idx]  # (B, H)

            # elif mode == "mean":
            #     # 使用 decoder 的非 padding token 平均
            #     if decoder_attention_mask is None:
            #         raise ValueError("decoder_attention_mask is required for mean pooling.")
            #     mask = decoder_attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            #     summed = (decoder_hidden_states * mask).sum(dim=1)  # (B, H)
            #     count = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
            #     features = summed / count  # (B, H)

            # else:
            #     raise ValueError(f"Unsupported probe_feature_mode: {mode}")

            # # --- Linear Probe 分类头 ---
            # # probe_head = getattr(model, "probe_head", None)
            # # if probe_head is None:
            # #     probe_head = getattr(model.base_model.model, "probe_head", None)
            # # if probe_head is None:
            # #     raise ValueError("Model does not have `probe_head` defined.")

            # # logits = model.probe_head(features)  # (B, num_classes)
            # # 🔥 核心改动：用 unwrapped model 的 probe_head
            # # logits = real_model.probe_head(features)
            # # loss = F.cross_entropy(logits, labels)
            # # loss = F.cross_entropy(logits, labels.view(-1))  # ✅ 处理 labels shape

            
            # # # ✅ 新增：处理 labels 是文本字符串的情况
            # # if isinstance(labels[0], str):
            # #     unique_labels = sorted(list(set(labels)))
            # #     label2id = {label: idx for idx, label in enumerate(unique_labels)}
            # #     labels = torch.tensor([label2id[label] for label in labels], dtype=torch.long, device=logits.device)

            # # elif isinstance(labels, list):
            # #     # 如果已经是 list of int
            # #     labels = torch.tensor(labels, dtype=torch.long, device=logits.device)

            # # elif isinstance(labels, torch.Tensor):
            # #     labels = labels.to(device=logits.device, dtype=torch.long)
            # #     if labels.dim() == 2:
            # #         labels = labels[:, 0]  # 取每条 label 的第一个 token

            # # else:
            # #     raise ValueError(f"Unsupported labels type: {type(labels)}")
            
            # # # 🧹 保证 labels shape 正确，cross_entropy 要求 (B,) 而不是 (B, 1)
            # # labels = labels.view(-1)
            # # loss = F.cross_entropy(logits, labels)
            # # --- CrossEntropy Loss ---

            #  # --- Probe Head 分类 ---
            # logits = real_model.probe_head(features)  # (B, num_classes)

            # # 🚀 计算 CrossEntropy Loss
            # loss = F.cross_entropy(logits, class_labels.view(-1))  # ✅ 注意 shape


            # return (loss, {"logits": logits, "labels": class_labels}) if return_outputs else loss


    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.

    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)

    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)

    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #     if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #         # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
    #         loss = loss / self.args.gradient_accumulation_steps

    #     ########################### Regularization ##########################
    #     orthogonal_loss = 0.
    #     for name, param in self.model.named_parameters():
    #         if "lora_A" in name:
    #             for name_, param_ in self.model.named_parameters():
    #                 if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
    #                     orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
    #                     break # target modules have been matched

    #     # l2-normalization for loranew_A/B
    #     l2_loss = 0.
    #     for name, param in self.model.named_parameters():
    #         if "loranew_" in name:
    #             l2_loss += torch.norm(param, p=2)

    #     lamda_1 = self.args.lamda_1
    #     lamda_2 = self.args.lamda_2

    #     logger.info(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {lamda_1}; λ2: {lamda_2}")
    #     loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
    #     ######################################################################

    #     if self.do_grad_scaling:
    #         self.scaler.scale(loss).backward()
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     elif self.deepspeed:
    #         # loss gets scaled under gradient_accumulation_steps in deepspeed
    #         loss = self.deepspeed.backward(loss)
    #     else:
    #         loss.backward()

    #     return loss.detach()

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        logger.debug(
            f"调用自定义函数 Class ContinualTrainer def evaluation_loop self.args.predict_with_generate: {self.args.predict_with_generate},  prediction_loss_only: { prediction_loss_only},ignore_keys: {ignore_keys}"
        )


        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None,  # inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(
                dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        # logger.debug(
        #     f"调用自定义函数 Class ContinualTrainer def prediction_step() self.args.predict_with_generate: {self.args.predict_with_generate},  prediction_loss_only: { prediction_loss_only},ignore_keys: {ignore_keys}"
        # )

        # 🔵 处理 Linear Probe 模式
        if getattr(self.args, "use_probe", False) :
           inputs = self._prepare_inputs(inputs)

           with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    decoder_input_ids=inputs.get("decoder_input_ids", None),
                    decoder_attention_mask=inputs.get("decoder_attention_mask", None),
                    output_hidden_states=True,
                    return_dict=True,
                )

                logits = outputs.logits  # 🔥 直接拿 logits
                class_labels = inputs["class_labels"]  # 🔥 使用 class_labels

                loss = F.cross_entropy(logits, class_labels.view(-1))  # 🔥 注意 view(-1)

                if prediction_loss_only:
                    return (loss, None, None)
                else:
                    return (loss, logits, class_labels)

       
        # 若未启用生成式预测（如分类任务），或者只需要 loss（如只计算验证损失），就用原始 Trainer 的 prediction_step
        if not self.args.predict_with_generate or prediction_loss_only:
            logger.debug(f"调用父类 prediction_step()")
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs
        gen_kwargs["synced_gpus"] = True if is_deepspeed_zero3_enabled() else False

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        generation_config = GenerationConfig(**gen_kwargs)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        # 某些模型（如 encoder-decoder）encoder.main_input_name 不一定是 'input_ids'，因此进行兼容性处理。
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:

            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        
        # 生成式预测时，labels 作为 decoder 的输入 
        # 使用 .generate() 来生成预测文本序列
        generated_tokens = self.model.generate(
            input_ids=generation_inputs,
            generation_config=generation_config
        )

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, max_length)

        with torch.no_grad():
            # 如果有标签，则正向推理一遍算 loss
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        # 若只需要 loss，就不返回预测结果。
        if self.args.prediction_loss_only:
            return (loss, None, None)
        # 否则返回三元组：
        # loss：用于评估损失
        # generated_tokens：用于 decode 出预测文本
        # labels：ground truth 标签
        

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    # def _inner_training_loop(
    #     self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     logger.debug(
    #         f"调用自定义函数 Class ContinualTrainer def _inner_training_loop() batch_size: {batch_size}, args: {args}, resume_from_checkpoint: {resume_from_checkpoint}, trial: {trial}, ignore_keys_for_eval: {ignore_keys_for_eval}"
    #     )
    #     self._train_batch_size = batch_size
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = args.train_batch_size * \
    #         args.gradient_accumulation_steps * args.world_size

    #     len_dataloader = None
    #     if has_length(train_dataloader):
    #         len_dataloader = len(train_dataloader)
    #         num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         num_examples = self.num_examples(train_dataloader)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #         else:
    #             max_steps = math.ceil(
    #                 args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = self.num_examples(
    #                 train_dataloader) * args.num_train_epochs
    #     elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_examples = total_train_batch_size * args.max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #     else:
    #         raise ValueError(
    #             "args.max_steps must be set to a positive value if dataloader does not have a length, was"
    #             f" {args.max_steps}"
    #         )

    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torch.distributed.launch)."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    #     delay_optimizer_creation = (
    #         self.sharded_ddp is not None
    #         and self.sharded_ddp != ShardedDDPOption.SIMPLE
    #         or is_sagemaker_mp_enabled()
    #         or self.fsdp is not None
    #     )
    #     if args.deepspeed:

    #         # self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    #         # # 然后交给 deepspeed，但不让它覆盖 optimizer
    #         # deepspeed_engine, _, _ = deepspeed_init(
    #         #     self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
    #         # )

    #         # self.model = deepspeed_engine.module
    #         # self.model_wrapped = deepspeed_engine
    #         # self.deepspeed = deepspeed_engine

    #         # ---- 原来的设置
    #         deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
    #             self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine
    #         self.optimizer = optimizer
    #         self.lr_scheduler = lr_scheduler
    #         # $ ---- 原来的设置

    #     elif not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None

    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable()

    #     model = self._wrap_model(self.model_wrapped)

    #     if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
    #         self._load_from_checkpoint(resume_from_checkpoint, model)

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     if delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)

    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples:,}")
    #     logger.info(f"  Num Epochs = {num_train_epochs:,}")
    #     logger.info(
    #         f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
    #     logger.info(
    #         f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    #     logger.info(
    #         f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps:,}")
    #     logger.info(
    #         f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None

    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(
    #             os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (
    #                 num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info(
    #             "  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(
    #             f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             if skip_first_batches is None:
    #                 logger.info(
    #                     f"  Will skip the first {epochs_trained} epochs then the first"
    #                     f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
    #                     " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
    #                     " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
    #                     " training on data already seen by your model."
    #                 )
    #             else:
    #                 logger.info(
    #                     f"  Will skip the first {epochs_trained} epochs then the first"
    #                     f" {steps_trained_in_current_epoch} batches in the first epoch."
    #                 )
    #             if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
    #                 steps_trained_progress_bar = tqdm(
    #                     total=steps_trained_in_current_epoch)
    #                 steps_trained_progress_bar.set_description(
    #                     "Skipping the first batches")

    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     if self.hp_name is not None and self._trial is not None:
    #         # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
    #         # parameter to Train when using DDP.
    #         self.state.trial_name = self.hp_name(self._trial)
    #     if trial is not None:
    #         assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()

    #     self.control = self.callback_handler.on_train_begin(
    #         args, self.state, self.control)

    #     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    #     if not args.ignore_data_skip:
    #         for epoch in range(epochs_trained):
    #             is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
    #                 train_dataloader.sampler, RandomSampler
    #             )
    #             if is_torch_less_than_1_11 or not is_random_sampler:
    #                 # We just need to begin an iteration to create the randomization of the sampler.
    #                 # That was before PyTorch 1.11 however...
    #                 for _ in train_dataloader:
    #                     break
    #             else:
    #                 # Otherwise we need to call the whooooole sampler cause there is some random operation added
    #                 # AT THE VERY END!
    #                 _ = list(train_dataloader.sampler)

    #     total_batched_samples = 0
    #     for epoch in range(epochs_trained, num_train_epochs):
    #         if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
    #             train_dataloader.sampler.set_epoch(epoch)
    #         elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
    #             train_dataloader.dataset.set_epoch(epoch)

    #         if is_torch_tpu_available():
    #             parallel_loader = pl.ParallelLoader(
    #                 train_dataloader, [args.device]).per_device_loader(args.device)
    #             epoch_iterator = parallel_loader
    #         else:
    #             epoch_iterator = train_dataloader

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = (
    #             len(epoch_iterator)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(
    #             args, self.state, self.control)

    #         if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
    #             self._load_rng_state(resume_from_checkpoint)

    #         rng_to_sync = False
    #         steps_skipped = 0
    #         if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
    #             epoch_iterator = skip_first_batches(
    #                 epoch_iterator, steps_trained_in_current_epoch)
    #             steps_skipped = steps_trained_in_current_epoch
    #             steps_trained_in_current_epoch = 0
    #             rng_to_sync = True

    #         step = -1
    #         for step, inputs in enumerate(epoch_iterator):
    #             if step == 0:
    #                 logger.info("===== Sample input batch =====")
    #                 for k, v in inputs.items():
    #                     if isinstance(v, torch.Tensor):
    #                         logger.info(f"{k}: shape={v.shape}, dtype={v.dtype}")
    #                         logger.info(f"{k} sample values: {v[0][:10]}")
    #                     else:
    #                         logger.info(f"{k}: type={type(v)}")



    #             total_batched_samples += 1
    #             if rng_to_sync:
    #                 self._load_rng_state(resume_from_checkpoint)
    #                 rng_to_sync = False

    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 if steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.update(1)
    #                 if steps_trained_in_current_epoch == 0:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                 continue
    #             elif steps_trained_progress_bar is not None:
    #                 steps_trained_progress_bar.close()
    #                 steps_trained_progress_bar = None

    #             if step % args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(
    #                     args, self.state, self.control)

    #             if (
    #                 (total_batched_samples %
    #                  args.gradient_accumulation_steps != 0)
    #                 and args.local_rank != -1
    #                 and args._no_sync_in_gradient_accumulation
    #             ):
    #                 # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
    #                 with model.no_sync():
    #                     tr_loss_step = self.training_step(model, inputs)
    #             else:
    #                 tr_loss_step = self.training_step(model, inputs)

    #             if (
    #                 args.logging_nan_inf_filter
    #                 and not is_torch_tpu_available()
    #                 and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #             ):
    #                 # if loss is nan or inf simply add the average of previous logged losses
    #                 tr_loss += tr_loss / \
    #                     (1 + self.state.global_step - self._globalstep_last_logged)
    #             else:
    #                 tr_loss += tr_loss_step

    #             self.current_flos += float(self.floating_point_ops(inputs))

    #             # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
    #             if self.deepspeed:
    #                 self.deepspeed.step()

    #             if total_batched_samples % args.gradient_accumulation_steps == 0 or (
    #                 # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                 steps_in_epoch <= args.gradient_accumulation_steps
    #                 and (step + 1) == steps_in_epoch
    #             ):
    #                 # Gradient clipping
    #                 if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
    #                     # deepspeed does its own clipping

    #                     if self.do_grad_scaling:
    #                         # Reduce gradients first for XLA
    #                         if is_torch_tpu_available():
    #                             gradients = xm._fetch_gradients(self.optimizer)
    #                             xm.all_reduce("sum", gradients,
    #                                           scale=1.0 / xm.xrt_world_size())
    #                         # AMP: gradients need unscaling
    #                         self.scaler.unscale_(self.optimizer)

    #                     if is_sagemaker_mp_enabled() and args.fp16:
    #                         self.optimizer.clip_master_grads(
    #                             args.max_grad_norm)
    #                     elif hasattr(self.optimizer, "clip_grad_norm"):
    #                         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
    #                         self.optimizer.clip_grad_norm(args.max_grad_norm)
    #                     elif hasattr(model, "clip_grad_norm_"):
    #                         # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
    #                         model.clip_grad_norm_(args.max_grad_norm)
    #                     else:
    #                         # Revert to normal clipping otherwise, handling Apex or full precision
    #                         nn.utils.clip_grad_norm_(
    #                             amp.master_params(
    #                                 self.optimizer) if self.use_apex else model.parameters(),
    #                             args.max_grad_norm,
    #                         )

    #                 # Optimizer step
    #                 optimizer_was_run = True
    #                 # if self.deepspeed:
    #                 #     pass  # called outside the loop
    #                 # elif is_torch_tpu_available():
    #                 #     if self.do_grad_scaling:
    #                 #         self.scaler.step(self.optimizer)
    #                 #         self.scaler.update()
    #                 #     else:
    #                 #         xm.optimizer_step(self.optimizer)
    #                 # elif self.do_grad_scaling:
    #                 #     scale_before = self.scaler.get_scale()
    #                 #     self.scaler.step(self.optimizer)
    #                 #     self.scaler.update()
    #                 #     scale_after = self.scaler.get_scale()
    #                 #     optimizer_was_run = scale_before <= scale_after
    #                 # else:
    #                 #     self.optimizer.step()

    #                 if optimizer_was_run and not self.deepspeed:
    #                     self.lr_scheduler.step()

    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = epoch + \
    #                     (step + 1 + steps_skipped) / steps_in_epoch
    #                 self.control = self.callback_handler.on_step_end(
    #                     args, self.state, self.control)

    #                 self._maybe_log_save_evaluate(
    #                     tr_loss, model, trial, epoch, ignore_keys_for_eval)
    #             else:
    #                 self.control = self.callback_handler.on_substep_end(
    #                     args, self.state, self.control)

    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems to be not a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(
    #             args, self.state, self.control)
    #         self._maybe_log_save_evaluate(
    #             tr_loss, model, trial, epoch, ignore_keys_for_eval)

    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_tpu_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info(
    #         "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sur the model has been saved by process 0.
    #         if is_torch_tpu_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         elif args.local_rank != -1:
    #             dist.barrier()
    #         elif is_sagemaker_mp_enabled():
    #             smp.barrier()

    #         self._load_best_model()

    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     train_loss = self._total_loss_scalar / self.state.global_step

    #     metrics = speed_metrics(
    #         "train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss

    #     self.is_in_train = False

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     self.log(metrics)

    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(
    #         use_mtime=False, output_dir=run_dir)

    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    #         for checkpoint in checkpoints_sorted:
    #             if checkpoint != self.state.best_model_checkpoint:
    #                 logger.info(
    #                     f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #                 shutil.rmtree(checkpoint)

    #     self.control = self.callback_handler.on_train_end(
    #         args, self.state, self.control)

    #     return TrainOutput(self.state.global_step, train_loss, metrics)

    def _compute_stats(self, eigenvalues):
        """改进点13：统一统计计算"""
        if not eigenvalues:
            return {"min": 0., "max": 0., "median": 0., "mean": 0.}

        arr = np.array(eigenvalues)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }

    def compute_loss_landscape(
        self, eval_dataset: Dataset, output_dir, x_range=(-1, 1), y_range=(-1, 1), num_points=10, max_batches=5, sample_batches=False,

    ):
        """
        计算损失景观，并分别计算:
        1️⃣ **完整模型（Base Model + LoRA Adapter）** 的损失景观
        2️⃣ **仅 LoRA 适配器（LoRA Adapter）** 的损失景观
        计算损失景观，并分别计算完整模型 & LoRA Adapter 的损失表面。
        针对大模型使用多 GPU，可以并行分配 (i, j) 坐标网格。
        """
        args = self.args
        device = args.device
        # 最终损失网络的数值
        loss_grid = np.zeros((num_points, num_points))

        # 根据模型判断，这里不妨使用手动
        if args.do_train:
            # 如果进行训练，那么模型的new_lora就是新的当前任务添加的lora
            # Flag_newtaskLoRA = ''  # 如果只干扰训练后的新的任务的lora
            # Flag_onlyLoRA = ''  # 如果只干扰lora 部分
            # Flag_fullModel =  'fullModel'  # 如果干扰模型的所有参数
            distrub_name = 'fullModel'
            surf_file = os.path.join(
                    output_dir, f"{args.lora_strategy}_{distrub_name}_T5large_testData.h5")
        else:
            distrub_name = 'fullModel'
            # 如果不进行训练，只展示加载模型的 lossland ，则对模型的所有参数(不包括new_lora)都进行干扰，
            surf_file = os.path.join(output_dir, f"{distrub_name}_T5large_testData.h5")


        # 确保目标目录存在（若不存在则创建）
        if not os.path.exists(surf_file):
            with open(surf_file, 'w') as f:
                pass  # 成功创建空文件

        # ✅ 记录日志信息
        logger.info(f'***5***--5-2 **1 compute_loss_landscape  ')
        logger.info(
            f"***** Running Loss Landscape Calculati on {distrub_name}*****")
        logger.info(
            f"Output Dir = {output_dir}，Output File :{surf_file} Num points = {num_points}x{num_points} max_batches = {max_batches}")

        # ✅ 兼容 AMP 和分布式训练
        # 复制模型避免污染
        model = copy.deepcopy(self.model)
        model = self._wrap_model(model, training=False)

        # ✅ 处理 FP16/BF16 评估模式
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=device)

        model = model.to(device=device)
        model.eval()  # 确保模型在 eval 模式

        # --------------------- 生成参数扰动阶段阶段 ---------------------
        # 确定需要扰动的参数名称
        # 根据不同的微调方法（Nlora或lora）确定需要保存原始值的参数
        original_params_to_perturb = {}
        for name, param in model.named_parameters():
            if args.do_train:
                if distrub_name == 'fullModel':
                    pass
            else:# 不是训练后直接评估，加载的模型不应该包括新添加的 new_lora 部分
                if distrub_name == 'fullModel':
                    if "new_lora" not in name : 
                        original_params_to_perturb[name] = param.data.clone()

        # --------------------- 扰动生成优化 ---------------------
        torch.manual_seed(42)  # 固定随机种子保证可重复性
        perturb_x, perturb_y = {}, {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in original_params_to_perturb:
                    continue

                # 直接生成归一化扰动（优化点2）
                # norm_factor = torch.norm(param) + 1e-8  # 计算神经网络参数的尺度
                # torch.manual_seed(seed_x)
                # d_x_perturb = torch.randn_like(param).to(device)  # 生成随机扰动
                # perturb_x[name] = (d_x_perturb / torch.norm(d_x_perturb)) * norm_factor  # 归一化并调整尺度
                else:
                    d_x = torch.randn_like(param)
                    d_x = (d_x / d_x.norm()) * (param.norm() + 1e-8)  # 直接归一化
                    perturb_x[name] = d_x.to(device)

                    # 生成正交扰动
                    d_y = torch.randn_like(param)
                    d_y = d_y - torch.sum(d_y * d_x) * \
                        d_x / (d_x.norm()**2)  # 施密特正交化
                    d_y = (d_y / d_y.norm()) * (param.norm() + 1e-8)     # 归一化
                    perturb_y[name] = d_y.to(device)
        # 使用FP16存储扰动（优化点5）
        if args.fp16_full_eval or args.bf16_full_eval:
            perturb_x = {k: v.half() for k, v in perturb_x.items()}
            perturb_y = {k: v.half() for k, v in perturb_y.items()}

        # --------------------- 并行化网格计算（优化点3）---------------------
        x_coords = np.linspace(x_range[0], x_range[1], num_points)
        y_coords = np.linspace(y_range[0], y_range[1], num_points)

        # 分布式任务划分
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()  # 获取总进程数（GPU数量）
            rank = torch.distributed.get_rank()  # 获取当前进程的编号（从0开始）
            # 按行划分任务
            chunk = num_points // world_size
            start_idx = rank * chunk
            end_idx = (rank + 1) * chunk if rank != world_size - \
                1 else num_points
            # all_batches = all_batches[rank::world_size]  # 数据分片
            logger.info(
                f'***5***--5-2**4 distribute--yes world_size:{world_size} rank:{rank},start_idx:{start_idx},end_idx:{end_idx} ')
        else:
            # logger.info(f'****5***--5-2**2-1 if**5***--5-2**5 distribute--no ')
            world_size = 1
            rank = 0
            start_idx, end_idx = 0, num_points
            logger.info(
                f'***5***--5-2**4 distribute--no world_size:{world_size} rank:{rank},start_idx:{start_idx},end_idx:{end_idx} ')

        # --------------------- 数据准备阶段 ---------------------
        logger.info(
            f'***5***--5-2**2 begin load data---{torch.distributed.is_initialized()} ')
        dataloader = self.get_eval_dataloader(eval_dataset)
        all_batches = list(dataloader)[:max_batches]
        logger.info(f'***5***--5-2**2 begin load data---{len(all_batches)} ')

        # 合并所有批次为单个大批次（显存允许时）
        if not sample_batches and len(all_batches) > 0:
            try:
                # 尝试合并所有小批次为一个大批次
                big_batch = {
                    # 对每个特征键（如input_ids、labels）进行纵向拼接
                    k: torch.cat([b[k] for b in all_batches], dim=0)
                    for k in all_batches[0].keys()  # 假设所有批次结构相同
                }
                # 用合并后的大批次替换原始批次列表
                all_batches = [big_batch]  # 现在只包含一个合并后的批次
            except RuntimeError:
                # 显存不足时回退到原始小批次
                logger.warning("无法合并批次，保持原有批次数量")

# --------------------- 主计算循环优化 ---------------------

        for i in tqdm(range(start_idx, end_idx), desc=f"Rank {rank} Processing"):
            xv = x_coords[i]
            for j, yv in enumerate(y_coords):
                # 恢复参数时仅操作需要修改的部分（优化点1）
                for name in original_params_to_perturb:
                    model.state_dict()[name].copy_(
                        original_params_to_perturb[name])

                # 应用扰动
                with torch.no_grad():
                    for name in original_params_to_perturb:
                        param = model.state_dict()[name]
                        delta = xv * \
                            perturb_x[name].to(
                                param.dtype) + yv * perturb_y[name].to(param.dtype)
                        param.add_(delta)  # 原位操作减少内存分配

                # 计算损失
                total_loss = 0.0
                for batch in all_batches:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    with torch.cuda.amp.autocast(enabled=args.fp16):  # 支持混合精度
                        outputs = model(**inputs)
                        loss = F.cross_entropy(
                            outputs.logits.view(-1, outputs.logits.size(-1)),
                            inputs["labels"].view(-1)
                        )
                    total_loss += loss.item()

                loss_grid[i, j] = total_loss / len(all_batches)

                # 每5次迭代清理一次缓存（优化点6）
                if j % 5 == 0:
                    torch.cuda.empty_cache()

        # --------------------- 分布式结果收集 ---------------------
        if torch.distributed.is_initialized():
            # 收集所有进程的loss_grid
            # 将 loss_grid 转换为张量
            loss_grid_tensor = torch.tensor(loss_grid, device=device)
            # 创建一个列表，用于接收所有进程的 loss_grid
            all_loss = [torch.zeros_like(loss_grid_tensor)
                        for _ in range(world_size)]
            # 收集所有进程的 loss_grid
            torch.distributed.all_gather(all_loss, loss_grid_tensor)
            # 将收集到的结果拼接成一个完整的 loss_grid
            # loss_grid_tensor = torch.cat(all_loss, dim=0)
            loss_grid_tensor = torch.stack(all_loss, dim=0).mean(dim=0)
            loss_grid = loss_grid_tensor.cpu().numpy()  # 转回 NumPy 数组后再保存
        else:
            all_loss = [loss_grid]

        # 仅rank 0进程保存结果
        if rank == 0:
            with h5py.File(surf_file, 'w') as f:
                f.create_dataset('xcoordinates', data=x_coords)
                f.create_dataset('ycoordinates', data=y_coords)
                f.create_dataset('train_loss', data=loss_grid)

            logger.info(f"计算完成，结果保存至{surf_file}")

        # return True
    


    def compute_fisher_information(self, eval_dataset, output_dir, name="FisherInfo", fisher_samples=1000, batch_size=32):
        """
        计算 Fisher Information 并存储到 HDF5 文件，包括统计信息
        :param flag_Nlora: 是否使用 Nlora 方法
        :param eval_dataset: 评估数据集
        :param output_dir: 结果存储目录
        :param name: 结果文件名称
        :param fisher_samples: 计算 Fisher 信息所用的样本数
        :param batch_size: 计算 Fisher 信息时的 batch 大小
        """
        args = self.args
        device = args.device

        if args.lora_strategy.lower() == 'nlora':
            fisher_file = os.path.join(
                output_dir, f"{name}_Fisher_Nlora_task.h5")
        elif args.lora_strategy.lower() == 'lora':
            fisher_file = os.path.join(output_dir, f"{name}_Fisher_LoRA.h5")

        os.makedirs(output_dir, exist_ok=True)

        # 复制模型避免污染
        model = copy.deepcopy(self.model)
        model = self._wrap_model(model, training=False)
        model.to(device).eval()

        # 随机采样训练样本
        sampler = RandomSampler(
            eval_dataset, replacement=True, num_samples=fisher_samples)
        dataloader = DataLoader(
            eval_dataset, sampler=sampler, batch_size=batch_size)

        fisher_vector = None
        total_samples = 0

        logger.info(
            f"计算 Fisher Information, 样本数: {fisher_samples}, batch_size: {batch_size}")

        for batch in tqdm(dataloader, desc="Computing Fisher Information"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            model.zero_grad()

            # 计算模型输出
            outputs = model(inputs)
            log_probs = F.log_softmax(outputs.logits, dim=-1)

            # 选择 ground-truth 对应的 log 概率
            log_probs_selected = log_probs.gather(
                dim=-1, index=labels.unsqueeze(-1)).squeeze()

            # 计算对数概率的梯度
            gradients = torch.autograd.grad(
                log_probs_selected.sum(), model.parameters(), create_graph=False)

            # 计算 Fisher Information 的对角近似
            grad_vector = torch.cat([g.view(-1)
                                    for g in gradients]).detach() ** 2

            # Fisher 信息累加
            if fisher_vector is None:
                fisher_vector = grad_vector
            else:
                fisher_vector += grad_vector

            total_samples += len(labels)

            # 每 5 个 batch 释放显存
            if total_samples % (5 * batch_size) == 0:
                torch.cuda.empty_cache()

        fisher_vector /= total_samples  # 归一化

        # **计算 Fisher Information 统计信息**
        fisher_stats = self._compute_fisher_stats(fisher_vector.cpu().numpy())

        # **保存 Fisher 信息和统计信息**
        with h5py.File(fisher_file, 'w') as f:
            f.create_dataset('fisher_vector', data=fisher_vector.cpu().numpy())
            for k, v in fisher_stats.items():
                f.attrs[k] = v  # 直接将统计数据写入 HDF5 文件

        logger.info(f"Fisher Information 计算完成，结果保存至 {fisher_file}")
        logger.info(f"Fisher 统计信息: {fisher_stats}")

    def _compute_fisher_stats(self, fisher_values):
        """计算 Fisher Information 的统计信息"""
        if fisher_values is None or len(fisher_values) == 0:
            return {"min": 0., "max": 0., "median": 0., "mean": 0., "std": 0.}

        arr = np.array(fisher_values)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }

    def compute_hessian_version1(
        self,
        flag_Nlora,
        eval_dataset,
        output_dir,
        name="hessian",
        max_batches=10,
        sample_batches=False,
        use_gpu=True,
        flag_Nlora_full=True,

    ):
        """
        改进版Hessian矩阵计算函数，主要优化：
        1. 完全消除参数污染风险
        2. 支持分布式训练环境
        3. 增强数值稳定性
        4. 内存效率优化
        5. 增加特征向量分析

        参数说明：
        - max_batches: 最大计算batch数（用于大数据集采样）
        - sample_batches: 是否随机采样batch（True=随机，False=顺序取前N个）

        """
        # --------------------- 初始化阶段 ---------------------
        # 根据模型判断，这里不妨使用手动

        if flag_Nlora and (not flag_Nlora_full):
            # 如果是 Nlora 方法，只干扰 task 相关的代码
            Flag_Nlora_newtask = True
            Flag_Nlora_full = False
            Flag_lora = False
        elif flag_Nlora and flag_Nlora_full:
            # 如果是 Nlora 方法，干扰 task_lora 和之前的lora 相关的代码
            Flag_Nlora_newtask = True  # 只要是使用了Nlora，这里就是True
            Flag_Nlora_full = True  # 这个是用来唯一区别 full 还是 task
            Flag_lora = False
        else:
            Flag_Nlora_newtask = False
            Flag_Nlora_full = False
            Flag_lora = True

        logger.info(f'***5***--5-3**1 begin init   ')
        logger.info(
            f'***5***--5-3**1 use distribute ---- {torch.distributed.is_initialized()}')
        args = self.args
        device = args.device

        hessian_file_Nlora_fulllora = os.path.join(
            output_dir, f"{name}_Nlora_only-predictDataset.h5")
        hessian_file_Nlora_tasklora = os.path.join(
            output_dir, f"{name}_Nlora_only-predictDataset.h5")
        hessian_file_lora = os.path.join(
            output_dir, f"{name}_lora_only-predictDataset_lanczos.h5")

        # 创建独立模型副本（关键改进点1：隔离原始模型）
        model = copy.deepcopy(self.model)
        model = self._wrap_model(model, training=False)

        # 混合精度处理
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=device)

        model = model.to(device=device)
        model.eval()

        # 保存初始参数状态（关键改进点2：消除参数污染）
        # original_state = {
        # k: v.to(device) if isinstance(v, torch.Tensor) else v
        # for k, v in model.state_dict().items()
        # }
        # original_state = copy.deepcopy(model.state_dict())
        logger.info(f'***5***--LORA Hessian**1 finish init ')

        # --------------------- 数据准备阶段 ---------------------
        logger.info(f'***5***--5-3**2 begin load data   ')
        dataloader = self.get_eval_dataloader(eval_dataset)
        all_batches = list(dataloader)

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            all_batches = all_batches[rank::world_size]  # 数据分片
            logger.info(
                f'***5***--LORA Hessian**2 distribute yes, rank:{rank}, total_batches:{len(all_batches)}')

        else:
            logger.info(f'***5***--5-3**4 distribute--no ')
            world_size = 1
            rank = 0

        # 批量采样逻辑（改进点3：内存优化）
        logger.info(
            f'***5***--5-3**2-1 if  {len(all_batches)} , {max_batches} ')
        if len(all_batches) > max_batches:
            if sample_batches:
                indices = np.random.choice(
                    len(all_batches), max_batches, replace=False)
                all_batches = [all_batches[i] for i in indices]
            else:
                all_batches = all_batches[:max_batches]

        logger.info(f'***5***--5-3**2 finish load data ')
        # --------------------- 核心算法定义 ---------------------

        class HessianCalculator:
            """
            Hessian 计算器：
            - 计算 Hessian-Vector Product (HVP)
            - 使用 Lanczos 方法估计 Hessian 的特征值

            参数：
            - model: 计算 Hessian 的神经网络模型
            - device: 计算设备 (CPU/GPU)
            """

            def __init__(self, model, device, max_dim=10000):
                logger.info(f'***5***--5-3**3 class HessianCalculator init   ')
                self.model = model
                self.device = device
                self.criterion = torch.nn.CrossEntropyLoss()
                # self.max_dim = max_dim  # 限制 Hessian 计算的最大维度

            @staticmethod
            def _safe_normalize(v, eps=1e-12):
                """改进点4：安全归一化防止除零错误"""
                norm = torch.norm(v) + eps
                return v / norm

            def compute_hvp(self, batch, param_list=None):
                """计算Hessian-vector乘积函数 (HVP) """
                logger.info(f'***5***--5-3**5 begin compute_hvp() ')
                self.model.zero_grad()  # 清空梯度
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autograd.set_grad_enabled(True):
                    # outputs = checkpoint(self.model, **batch)  # 梯度检查点
                    outputs = self.model(**batch)
                    loss = self.criterion(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        batch["labels"].view(-1)
                    )

                # 根据传入参数的类型适配：字典或列表/元组
                # 选择需要计算 Hessian 的参数
                if isinstance(param_list, dict):
                    params = [p for p in param_list.values()
                              if p.requires_grad]
                elif isinstance(param_list, (list, tuple)):
                    params = [p for p in param_list if p.requires_grad]
                else:
                    params = []  # 若传入为 None 或其它类型，则空处理

                # 计算一阶梯度
                grads = torch.autograd.grad(loss, params, create_graph=True)

                def hvp_func(v):
                    """计算 Hessian 向量积,闭包函数保持计算图"""

                    split_sizes = [p.numel()
                                   for p in params]  # 计算每个 param 对应的大小
                    v_split = torch.split(v, split_sizes)  # 按照每个参数的形状拆分 v
                    v_reshaped = [v_i.view(p.shape) for v_i, p in zip(
                        v_split, params)]  # 重新调整 v_i 形状

                    # logger.info(f"grads shape: {[g.shape for g in grads]}")
                    # logger.info(f"v shape: {v.shape}")
                    # logger.info(f"Total params count: {len(params)}, Total elements in params: {sum(split_sizes)}")
                    # logger.info(f"🔹 First 5 split sizes: {split_sizes[:5]}")
                    # logger.info(f"🔹 First 5 param shapes: {[p.shape for p in params[:5]]}")
                    # logger.info(f"🔹 First 5 v_split shapes: {[v_i.shape for v_i in v_split[:5]]}")

                    # 计算 Hessian 作用
                    Hv = torch.autograd.grad(
                        grads, params, grad_outputs=v_reshaped,
                        retain_graph=True, allow_unused=True
                    )
                    # modified
                    # Hv = torch.autograd.grad(
                    #     grads, params, grad_outputs=v_reshaped,
                    #     retain_graph=False, allow_unused=True
                    # )

                    # 拼接计算结果
                    Hv_flattened = torch.cat([
                        hv.contiguous().flatten() if hv is not None else torch.zeros_like(p).flatten()
                        for hv, p in zip(Hv, params)
                    ]).to(self.device)

                    # logger.info(f"🔹 Hv computed successfully, shape: {Hv_flattened.shape}")
                    torch.cuda.empty_cache()  # 释放显存
                    return Hv_flattened

                return hvp_func

            # def lanczos_algorithm(self, hvp_func, dim, order=5, num_splits=4, random_seed=0):
                """
                使用标准 Lanczos 方法计算 Hessian 特征值，并基于 v_chunks 分块计算
                - hvp_func: Hessian-Vector Product 计算函数
                - dim: 需要计算的参数维度
                - order: Lanczos 迭代阶数
                - num_splits: 将 v 拆分成 num_splits 份，逐步计算 HVP 以减少显存占用
                - random_seed: 随机种子
                """
                torch.manual_seed(random_seed)
                warnings.filterwarnings("ignore", category=UserWarning)

                # 限制 Hessian 计算的最大维度
                dim = min(dim, self.max_dim)

                # 使用 `float16` 降低显存需求
                tridiag = torch.zeros(
                    (order, order), dtype=torch.float16, device=self.device)
                vecs = torch.zeros(
                    (dim, order), dtype=torch.float16, device=self.device)

                # 生成初始随机向量并归一化
                init_vec = torch.randn(
                    dim, 1, dtype=torch.float16, device=self.device)
                init_vec /= torch.norm(init_vec)
                vecs[:, 0:1] = init_vec

                beta = 0
                v_old_chunks = [torch.zeros_like(chunk) for chunk in torch.chunk(
                    init_vec, num_splits, dim=0)]  # 存储上一次的 v_old

                for i in range(order):
                    start_time = time.time()
                    v = vecs[:, i:i+1]

                    # ✅ 分块计算 HVP
                    v_chunks = torch.chunk(v, num_splits, dim=0)
                    w_chunks = []

                    for j, v_chunk in enumerate(v_chunks):
                        w_chunk = hvp_func(v_chunk)  # 计算 HVP
                        w_chunk = w_chunk - beta * \
                            v_old_chunks[j]  # 计算 w - beta * v_old
                        w_chunks.append(w_chunk)

                    # ✅ 计算 alpha（分块）
                    alpha_chunks = [torch.matmul(
                        w_chunk.T, v_chunk) for w_chunk, v_chunk in zip(w_chunks, v_chunks)]
                    alpha = sum(alpha_chunks)  # 逐块累加
                    tridiag[i, i] = alpha

                    # ✅ 计算 w_chunks = w_chunks - alpha * v_chunks
                    for j in range(num_splits):
                        w_chunks[j] = w_chunks[j] - alpha * v_chunks[j]

                    # ✅ 重正交化（分块计算）
                    for j in range(i):
                        tau_chunks = torch.chunk(
                            vecs[:, j:j+1], num_splits, dim=0)
                        coeff_chunks = [torch.matmul(
                            w_chunk.T, tau_chunk) for w_chunk, tau_chunk in zip(w_chunks, tau_chunks)]
                        coeff = sum(coeff_chunks)

                        for k in range(num_splits):
                            w_chunks[k] = w_chunks[k] - coeff * tau_chunks[k]

                    # ✅ 重新计算 beta
                    beta = torch.norm(torch.cat(w_chunks, dim=0))
                    if beta < 1e-6:
                        warnings.warn(
                            f"数值稳定性问题: beta={beta.item()} 在迭代 {i} 时过小。")

                    # ✅ 更新 vecs（直接使用 w_chunks）
                    if i + 1 < order:
                        tridiag[i, i+1] = beta
                        tridiag[i+1, i] = beta
                        vecs[:, i+1:i+2] = torch.cat(
                            [w_chunks[j] / beta for j in range(num_splits)], dim=0)

                    # ✅ 更新 v_old_chunks（用于下轮迭代）
                    v_old_chunks = [
                        w_chunks[j].clone() / beta for j in range(num_splits)]

                    elapsed_time = time.time() - start_time
                    print(
                        f"Iter {i}/{order}: α={alpha.item():.6f}, β={beta.item():.6f}, Time={elapsed_time:.2f}s")

                    torch.cuda.empty_cache()  # 释放显存

                return tridiag

            def block_lanczos(self, hvp_func, dim, k=10, block_size=4):
                """
                改进点7：分块Lanczos算法（内存优化）
                其中超参数 k 是迭代次数，block_size 是块的大小。

                """
                logger.info(f'***5***--5-3**5 begin block_lanczos() ')
                # 初始化分块正交基
                Q = torch.zeros((k+1)*block_size, dim, device=self.device)
                T = torch.zeros(k*block_size, k*block_size, device=self.device)

                # 生成初始分块
                V = torch.randn(dim, block_size, device=self.device)
                V, _ = torch.linalg.qr(V)  # 正交化
                Q[:block_size] = V.T

                for i in range(k):
                    start_idx = i * block_size
                    # 计算Hessian作用
                    HV = torch.stack([hvp_func(Q[start_idx + j])
                                     for j in range(block_size)])

                    # 正交化过程
                    for j in range(start_idx, start_idx + block_size):
                        T[j, :j+1] = Q[:j+1] @ HV[j-start_idx]  # 计算三对角矩阵 T
                        # HV[j-start_idx] -= Q[:j+1] @ T[j, :j+1].T
                        # HV[j-start_idx] -= Q[:j+1] @ T[j, :j+1].unsqueeze(1)  # ✅ 修正形状问题
                        # HV[j-start_idx] -= Q[:j+1].T @ T[j, :j+1].unsqueeze(1)  # ✅ 修正形状问题
                        # 🔹 在计算前检查形状
                        # logger.info(f"🔹 Q[:j+1].shape: {Q[:j+1].shape}")
                        # logger.info(f"🔹 Q[:j+1].T.shape: {Q[:j+1].T.shape}")
                        # logger.info(f"🔹 T[j, :j+1].shape: {T[j, :j+1].shape}")
                        # logger.info(f"🔹 T[j, :j+1].unsqueeze(1).shape: {T[j, :j+1].unsqueeze(1).shape}")
                        # logger.info(f"🔹 HV[j-start_idx].shape: {HV[j-start_idx].shape}")

                        # ✅ 修正形状
                        HV[j-start_idx] -= (Q[:j+1].T @
                                            T[j, :j+1].unsqueeze(1)).squeeze()

                    # QR分解
                    V, R = torch.linalg.qr(HV.T)  # 正交化
                    Q[start_idx+block_size:start_idx+2*block_size] = V.T

                    # ✅ 修正错误：确保索引范围不会为空
                    if start_idx+block_size < T.shape[0]:
                        end_row = min(start_idx+2*block_size, T.shape[0])
                        end_col = min(start_idx+block_size, T.shape[1])

                        # logger.info(f"🔹 Updating T matrix at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}]")

                        T[start_idx+block_size:end_row, start_idx:end_col] = R.T
                    else:
                        logger.info(
                            f"❌ Skipping T update at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}] to prevent empty slice.")

                    # # ✅ 确保 T 不会越界
                    # end_row = min(start_idx+2*block_size, T.shape[0])
                    # end_col = min(start_idx+block_size, T.shape[1])

                    # logger.info(f"🔹 Updating T matrix at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}]")

                    # T[start_idx+block_size:end_row, start_idx:end_col] = R.T
                    # T[start_idx+block_size:start_idx+2*block_size,
                    # start_idx:start_idx+block_size] = R.T

                # 计算特征值
                T_np = T.cpu().numpy()
                eigvals = np.linalg.eigvalsh(T_np)
                return eigvals[-block_size:]  # 返回最大特征值

        # --------------------- 主计算流程 ---------------------
        logger.info(f'***5***--5-3**3 begin main loop   ')

        # 只存储初始状态（不带梯度）,用于恢复模型状态
        original_params_to_calculate_hessian = {}

        if Flag_Nlora_newtask and (not Flag_Nlora_full):
            dom_eigs_Nlora_tasklora = []
            Nlora_params_tasklora = {}
            # Nlora_params_lora = {}
            for name, param in model.named_parameters():
                if name.find("loranew_") != -1:
                    # 当使用Nlora 方法时，需要对lora_ 和 loranew_ 进行区分
                    # 进行扰动只考虑 loranew 的部分，即只更新  与任务有关的那一部分 lora
                    Nlora_params_tasklora[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )

        elif Flag_Nlora_newtask and (Flag_Nlora_full):
            dom_eigs_Nlora_lora = []
            Nlora_params_lora = {}
            for name, param in model.named_parameters():
                if name.find("loranew_") != -1:
                    # 当使用Nlora 方法时，需要对lora_ 和 loranew_ 进行区分
                    Nlora_params_lora[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )
                elif name.find("lora_") != -1:
                    # 当使用lora 方法时，只有一个 lora的部分 进行扰动只考虑 lora 的部分，即只更新
                    Nlora_params_lora[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )

        elif Flag_lora:
            dom_eigs_lora = []
            # 获取参数集合（改进点8：动态参数处理）
            lora_params = {}
            for name, param in model.named_parameters():
                if name.find("lora_") != -1:
                    lora_params[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )

        calculator = HessianCalculator(model, device)

        # 分布式通信初始化（改进点9：分布式支持）

        if torch.distributed.is_initialized():
            logger.info(f'***5***--5-3**4 distribute--yes ')
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            all_batches = all_batches[rank::world_size]  # 数据分片
            logger.info(f'***5***--5-3**4 {len(all_batches)} ')
        else:
            logger.info(f'***5***--5-3**4 distribute--no ')
            world_size = 1
            rank = 0

        try:
            for batch in tqdm(all_batches, desc=f"Rank {rank}: Processing"):
                # 重置模型参数（关键改进点10：消除参数污染）
                # model.load_state_dict(original_state)
                # 恢复参数时仅操作需要修改的部分（优化点1）
                for name in original_params_to_calculate_hessian:
                    model.state_dict()[name].copy_(
                        original_params_to_calculate_hessian[name])

                # 计算全模型Hessian
                # logger.info(f'***5***--5-3**5 begin calculate Hessian ')
                # hvp_full = calculator.compute_hvp(batch)
                # eigvals = calculator.block_lanczos(hvp_full, dim=sum(p.numel() for p in full_params))
                # dom_eigs_full.extend(eigvals.tolist())

                # ------------full model paramter analyase Hessien
                # saved_flags = {}
                # for name, param in calculator.model.named_parameters():
                #     # saved_flags保存之前的神经网络的 requires_grad 状态
                #     saved_flags[name] = param.requires_grad
                #      # 对全模型 Hessian：临时激活所有参数
                #     param.requires_grad = True

                # ---根据saved_flags 恢复之前的状态
                # for name, param in calculator.model.named_parameters():
                #     param.requires_grad = saved_flags.get(name, param.requires_grad)

                # ------------full model paramter analyase Hessien
                # hvp_full = compute_batch_hvp(batch, full_params)
                # eigvals_full = lanczos_iteration(hvp_full, full_params, k=num_iter)
                # dom_eigs_full.append(eigvals_full.max())
                # restore_gradients(model, saved_flags)

                # 计算LoRA Hessian
                if Flag_Nlora_newtask and (not Flag_Nlora_full):

                    logger.info(
                        f'***5***--5-3**Model device: {next(model.parameters()).device},Batch device: {next(iter(batch.values())).device} ')
                    logger.info(
                        f"Type of lora_params: {type(Nlora_params_tasklora)}")
                    # logger.info(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.info(
                        f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.info(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个

                    hvp_Nlora_tasklora = calculator.compute_hvp(
                        batch, Nlora_params_tasklora)
                    eigvals = calculator.block_lanczos(hvp_Nlora_tasklora, dim=sum(
                        p.numel() for p in Nlora_params_tasklora.values()))
                    dom_eigs_Nlora_tasklora.extend(eigvals.tolist())

                    # hvp_Nlora_lora = calculator.compute_hvp(batch,Nlora_params_lora)
                    # eigvals = calculator.block_lanczos(hvp_Nlora_lora, dim=sum(p.numel() for p in Nlora_params_fulllor))
                    # dom_eigs_Nlora_lora.extend(eigvals.tolist())

                # 计算LoRA Hessian 这里计算的是所有lora相关部分
                if Flag_Nlora_newtask and (Flag_Nlora_full):

                    logger.info(
                        f'***5***--5-3**Model device: {next(model.parameters()).device},Batch device: {next(iter(batch.values())).device} ')
                    logger.info(
                        f"Type of lora_params: {type(Nlora_params_tasklora)}")
                    # logger.info(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.info(
                        f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.info(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个

                    # hvp_Nlora_tasklora = calculator.compute_hvp(batch,Nlora_params_tasklora)
                    # eigvals = calculator.block_lanczos(hvp_Nlora_tasklora, dim=sum(p.numel() for p in Nlora_params_tasklora.values()))
                    # dom_eigs_Nlora_tasklora.extend(eigvals.tolist())

                    hvp_Nlora_lora = calculator.compute_hvp(
                        batch, Nlora_params_lora)
                    eigvals = calculator.block_lanczos(hvp_Nlora_lora, dim=sum(
                        p.numel() for p in Nlora_params_lora.values()))
                    dom_eigs_Nlora_lora.extend(eigvals.tolist())

                if Flag_lora:
                    logger.info(
                        f'***5***--5-3**Model device: {next(model.parameters()).device},Batch device: {next(iter(batch.values())).device} ')
                    logger.info(f"Type of lora_params: {type(lora_params)}")
                    # logger.info(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.info(
                        f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.info(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个

                    hvp_lora = calculator.compute_hvp(batch, lora_params)
                    eigvals = calculator.block_lanczos(
                        hvp_lora, dim=sum(p.numel() for p in lora_params.values()))
                    dom_eigs_lora.extend(eigvals.tolist())
                    # tridiag = calculator.lanczos_algorithm(hvp_lora, dim=sum(p.numel() for p in lora_params.values()))
                    # dom_eigs_lora.extend(torch.linalg.eigvalsh(tridiag).tolist())

                # 内存清理
                torch.cuda.empty_cache()

        except RuntimeError as e:
            logger.error(f"Hessian计算失败: {str(e)}")
            if "CUDA out of memory" in str(e):
                logger.warning("尝试启用梯度检查点...")
                # 此处可添加fallback逻辑
            raise

        # --------------------- 结果处理与保存 ---------------------
        logger.info(f'***5***--5-3**6 begin save ')
        # 分布式结果聚合（改进点11）
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if Flag_Nlora_newtask and (not Flag_Nlora_full):
                dom_eigs_Nlora_tasklora_tensor = torch.tensor(
                    dom_eigs_Nlora_tasklora, device=device)
                # dom_eigs_Nlora_fulllora_tensor = torch.tensor(dom_eigs_Nlora_lora, device=device)

                # 创建接收所有进程数据的列表
                all_dom_eigs_Nlora_tasklora = [torch.zeros_like(
                    dom_eigs_Nlora_tasklora_tensor) for _ in range(world_size)]
                # all_dom_eigs_Nlora_fulllora = [torch.zeros_like(dom_eigs_Nlora_fulllora_tensor) for _ in range(world_size)]

                # 收集所有进程的数据
                torch.distributed.all_gather(
                    all_dom_eigs_Nlora_tasklora, dom_eigs_Nlora_tasklora_tensor)
                # torch.distributed.all_gather(all_dom_eigs_Nlora_fulllora, dom_eigs_Nlora_fulllora_tensor)

                # 拼接所有收集到的数据
                dom_eigs_Nlora_tasklora = torch.cat(
                    all_dom_eigs_Nlora_tasklora, dim=0).cpu().numpy().tolist()
                # dom_eigs_Nlora_fulllora = torch.cat(all_dom_eigs_Nlora_fulllora, dim=0).cpu().numpy().tolist()
                # 计算统计指标
                stats_Nlora_tasklora = self._compute_stats(
                    dom_eigs_Nlora_tasklora)
                # stats_Nlora_lora = self._compute_stats(dom_eigs_Nlora_fulllora)

            elif Flag_Nlora_newtask and (Flag_Nlora_full):
                # dom_eigs_Nlora_tasklora_tensor = torch.tensor(dom_eigs_Nlora_tasklora, device=device)
                dom_eigs_Nlora_fulllora_tensor = torch.tensor(
                    dom_eigs_Nlora_lora, device=device)

                # 创建接收所有进程数据的列表
                # all_dom_eigs_Nlora_tasklora = [torch.zeros_like(dom_eigs_Nlora_tasklora_tensor) for _ in range(world_size)]
                all_dom_eigs_Nlora_fulllora = [torch.zeros_like(
                    dom_eigs_Nlora_fulllora_tensor) for _ in range(world_size)]

                # 收集所有进程的数据
                # torch.distributed.all_gather(all_dom_eigs_Nlora_tasklora, dom_eigs_Nlora_tasklora_tensor)
                torch.distributed.all_gather(
                    all_dom_eigs_Nlora_fulllora, dom_eigs_Nlora_fulllora_tensor)

                # 拼接所有收集到的数据
                # dom_eigs_Nlora_tasklora = torch.cat(all_dom_eigs_Nlora_tasklora, dim=0).cpu().numpy().tolist()
                dom_eigs_Nlora_fulllora = torch.cat(
                    all_dom_eigs_Nlora_fulllora, dim=0).cpu().numpy().tolist()
                # 计算统计指标
                # stats_Nlora_tasklora = self._compute_stats(dom_eigs_Nlora_tasklora)
                stats_Nlora_lora = self._compute_stats(dom_eigs_Nlora_fulllora)

            elif Flag_lora:
                # dom_eigs_lora = torch.tensor(dom_eigs_lora, device=device)
                # torch.distributed.all_reduce(dom_eigs_lora)
                # dom_eigs_lora = dom_eigs_lora.cpu().numpy().tolist()
                dom_eigs_lora_tensor = torch.tensor(
                    dom_eigs_lora, device=device)

                # 创建接收所有进程数据的列表
                all_dom_eigs_lora = [torch.zeros_like(
                    dom_eigs_lora_tensor) for _ in range(world_size)]

                # 收集所有进程的数据
                torch.distributed.all_gather(
                    all_dom_eigs_lora, dom_eigs_lora_tensor)

                # 拼接所有收集到的数据
                dom_eigs_lora = torch.cat(
                    all_dom_eigs_lora, dim=0).cpu().numpy().tolist()

                # 计算统计指标
                stats_lora = self._compute_stats(dom_eigs_lora)

        # HDF5保存（改进点12：元数据增强）
        # 仅 rank=0 进程保存结果，避免多个进程同时写入 HDF5
        if rank == 0:
            if Flag_Nlora_newtask and (not Flag_Nlora_full):
                # with h5py.File(hessian_file_Nlora_full, "w") as hf:
                #     hf.attrs["created_at"] = datetime.now().isoformat()
                #     hf.attrs["model_type"] = type(model).__name__
                #     for k, v in stats_Nlora_lora.items():
                #         hf.create_dataset(k, data=v)
                #     hf.create_dataset("dominant_eigs", data=np.array(dom_eigs_Nlora_fulllora))

                with h5py.File(hessian_file_Nlora_tasklora, "w") as hf:
                    hf.attrs["created_at"] = datetime.now().isoformat()
                    hf.attrs["model_type"] = type(model).__name__
                    for k, v in stats_Nlora_tasklora.items():
                        hf.create_dataset(k, data=v)
                    hf.create_dataset("dominant_eigs", data=np.array(
                        dom_eigs_Nlora_tasklora))

                logger.info(f"计算完成，结果保存至 {hessian_file_Nlora_tasklora} 文件")

            elif Flag_Nlora_newtask and (Flag_Nlora_full):
                with h5py.File(hessian_file_Nlora_fulllora, "w") as hf:
                    hf.attrs["created_at"] = datetime.now().isoformat()
                    hf.attrs["model_type"] = type(model).__name__
                    for k, v in stats_Nlora_tasklora.items():
                        hf.create_dataset(k, data=v)
                    hf.create_dataset(
                        "dominant_eigs", data=np.array(dom_eigs_Nlora_lora))

                logger.info(f"计算完成，结果保存至 {hessian_file_Nlora_fulllora} 文件")

            elif Flag_lora:
                with h5py.File(hessian_file_lora, "w") as hf:
                    hf.attrs["created_at"] = datetime.now().isoformat()
                    hf.attrs["model_type"] = type(model).__name__
                    for k, v in stats_lora.items():
                        hf.create_dataset(k, data=v)
                    hf.create_dataset(
                        "dominant_eigs", data=np.array(dom_eigs_lora))

                logger.info(f"计算完成，结果保存至 {hessian_file_lora} 文件")
        # 🚀 添加进程同步 & 关闭分布式进程
        if torch.distributed.is_initialized():
            logger.info("所有进程同步中...")
            torch.distributed.barrier()  # 确保所有进程都完成再继续

            if torch.distributed.get_rank() == 0:
                logger.info("所有进程已完成计算，开始关闭分布式进程...")

            torch.distributed.destroy_process_group()  # 释放 NCCL 资源
            logger.info("分布式进程已正确关闭")
        return True
