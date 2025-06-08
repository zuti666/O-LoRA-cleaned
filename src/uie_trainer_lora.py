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
import json

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

        # logger.debug("UIETrainer args: %r", args)
        # logger.debug("UIETrainer kwargs: %r", kwargs)

        # 当 使用 rwp 来生成扰动
        # if self.args.rwp_type != '':
        rwp_type = getattr(self.args, "rwp_type", "")
        logger.debug(f"rwp_type: {rwp_type}")
        if rwp_type != "":
            rwp_noise_type = getattr(self.args, "rwp_noise_type", "")
            noise_std= getattr(self.args, "noise_std", 1)
            if "fisher" in rwp_noise_type:
                self.fisher_dict = {}
                for name, param in self.model.named_parameters():
                    self.fisher_dict[name] = torch.zeros_like(param)
            logger.debug(f"rwp_noise_type: {rwp_noise_type}; noise_std:{noise_std}")


        use_probe = getattr(self.args, "train_method", "")
        if  use_probe == "use_probe":
            # 尝试寻找 probe_head
            probe_num_classes = getattr(self.args, "probe_num_classes", "")
            probe_feature_mode = getattr(self.args, "probe_feature_model", "")
            logger.info(f"use_probe, probe_num_classes: {probe_num_classes}; probe_feature_mode:{probe_feature_mode}")




    # def _run_rwp_step(self, model, inputs, noise_range="lora", use_ddp=False):
        
    #     # 设置累积次数为2之后，第一次计算的结果就是原来在第一，第二个batch的损失的累积（g_batch1和 g_batch2）

    #     # Step 2: g₁ - 添加扰动，重新计算梯度 这里实际上只给第二个batch添加了扰动 g_batch2,然后进行混合的话

    #     #  lammda_1（g_batch1+ g_batch2） + (1 - lammda_1) * (g_batch2+noise)

    #     # g0 = self._gather_grad_vector(model)

    #     # # Step 2: g₁ - 添加扰动并计算噪声梯度
    #     # # === Step 2: g₁ ===
    #     # disable_running_stats(model)
    #     # noise_list = []

    #     # with torch.no_grad():
    #     #     for name, param in model.named_parameters():
    #     #         # === 根据噪声注入范围决定是否处理当前参数 ===
    #     #         if noise_range == "lora":
    #     #             if param.requires_grad and "loranew_" in name:
    #     #                 noise = torch.randn_like(param) * self.args.sigma
    #     #                 param.data.add_(noise)
    #     #                 noise_list.append((param, noise))

    #     #                 logger.debug(f"[RWP] Injected noise into: {name} | shape: {param.shape}")


    #     #         elif noise_range == "full":
    #     #             noise = torch.randn_like(param) * self.args.sigma
    #     #             param.data.add_(noise)
    #     #             noise_list.append((param, noise))

    #     #             logger.debug(f"[RWP] Injected noise into: {name} | shape: {param.shape}")


    #     #         else:
    #     #             raise ValueError(f"[RWP] Unsupported noise_range setting: {noise_range}")


            
    #     # # forward + backward with noise → 得到 g₁
    #     # with self.compute_loss_context_manager():
    #     #     loss_noisy = self.compute_loss(model, inputs)

        


    #     # if self.args.n_gpu > 1:
    #     #     loss_noisy = loss_noisy.mean()
    #     # if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
    #     #     loss_noisy = loss_noisy / self.args.gradient_accumulation_steps


    #     # self.model.zero_grad()

    #     # # loss 的 backward
    #     # if self.do_grad_scaling:
    #     #     self.scaler.scale(loss).backward()
    #     # elif self.use_apex:
    #     #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #     #         scaled_loss.backward()
    #     # elif self.deepspeed:
    #     #     # loss gets scaled under gradient_accumulation_steps in deepspeed
    #     #     loss = self.deepspeed.backward(loss)
    #     # else:
    #     #     loss.backward()

    #     # g1 = self._gather_grad_vector(model)

    #     # # 恢复参数值
    #     # with torch.no_grad():
    #     #     for param, noise in noise_list:
    #     #         param.sub_(noise)

    #     # # === Gradient mix ===
    #     # # Step 3: 梯度融合
    #     # # g ← λ·g₁ + (1 - λ)·g₀ 
    #     # # 其中 g0 是没有扰动的， g1 是有扰动的 ,其对应的权重系数分别为  rwp_a，  rwp_b
    #     # mixed_grad = self.args.rwp_a * g0 + self.args.rwp_b * g1  
    #     # self._assign_grad_vector(model, mixed_grad)

    #     # DDP 模式下进行梯度同步
    #     if use_ddp and torch.distributed.is_initialized():
    #         torch.distributed.all_reduce(mixed_grad)
    #         mixed_grad /= torch.distributed.get_world_size()

    #     # === Optimizer step ===
    #     if is_torch_tpu_available():
    #         if self.do_grad_scaling:
    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()
    #         else:
    #             xm.optimizer_step(self.optimizer)
    #     elif self.do_grad_scaling:
    #         scale_before = self.scaler.get_scale()
    #         self.scaler.step(self.optimizer)
    #         self.scaler.update()
    #         scale_after = self.scaler.get_scale()
    #         optimizer_was_run = scale_before <= scale_after
    #     else:
    #         self.optimizer.step()

    
    def generate_noise(self,param, mode, std, fisher_param=None,fisher_scaler=0): 
        """
        Generate noise to inject into a parameter tensor using specified mode.

        """

        # 跳过空参数，初始的lora_a，b
        if param.numel() == 0:
            return torch.zeros_like(param)


        if mode == "Gauss_standard":
            # ε_ij ~ N(0, σ²) 
            # 正态分布，每个元素的标准差都是σ，所有噪声元素共享相同标准差,与参数的值无关
            # 标准高斯噪声：所有元素独立采样于 N(0, σ²)
            # randn_like 生成标准正态分布（标准差为 1），再通过 * std 缩放为 N(0, sigma²)
            
            # 下面代码 与 noise = torch.normal(mean=0., std=std, size=param.shape)完全等价
            
            scaler =  std
            noise = torch.randn_like(param) * scaler  # 标准差为 std

            return noise
        elif mode == "Gauss_element":
            # ε_ij ~ N(0, σ² * |W_ij|²)
            # 每个元素都是独立地根据 param的绝对值进行缩放 
            # 元素级噪声：每个元素独立采样于 N(0, σ² * |W_ij|²)
            # noise = torch.normal(
            #     mean=0.0, 
            #     std=std * (param.abs()+ 1e-16)  # 标准差为 sigma * |W_ij|
            # )
            # return noise
            # 经过分析  torch.normal 和  torch.randn_like   的实现是一样的

            # 另一种实现 
            # 生成标准正态分布后 然后再乘以 param 的绝对值
            scaler = std * (param.abs()+ 1e-16)
            noise = torch.randn_like(param) * scaler
            return noise

        elif mode == "Gauss_matrix":
            # # ε_ij ~ N(0, σ² * ||W||_F²)
            # 计算 Frobenius 范数：||W||_F = sqrt(sum(|W_ij|^2))
            fro_norm = torch.norm(param, p='fro')  # 直接计算 Frobenius 范数
            # 生成噪声：ε_ij ~ N(0, σ² * ||W||_F²)
            # 所有噪声元素共享 相同标准差 σ * ||W||_F，因此生成标准正态分布后统一乘以该标准差
            scaler =  std * fro_norm
            noise = torch.randn_like(param)  * scaler

        

        elif mode == "lpf_sgd":
            # LPF-SGD: ε_ij ~ N(0, σ² * ||W_i||^2)
            # ε_ij ~ N(0,σ² * ||W_i,:||^2)

            """# 参考的官方实现 
            # [LPF-SGD/codes/wrn_dataaug/example/lpf_train.py at master · devansh20la/LPF-SGD]
            # (https://github.com/devansh20la/LPF-SGD/blob/master/codes/wrn_dataaug/example/lpf_train.py)
            if len(mp.shape) > 1:
                    sh = mp.shape
                    sh_mul = np.prod(sh[1:])
            
            # 下面这行代码调用了三个代码 
            # mp.view(m, -1).norm(dim=1, keepdim=True)：取每行范数 ||W_i||_2, shape = [m, 1]
            # row_norm.repeat(1, n).view(m, n) ： 将它复制成和 param 同形状 scale 矩阵，shape = [m, n]
            
                    temp = mp.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(mp.shape)
            # torch.normal :从 N(0, (std * scale)^2) 中采样
                    temp = torch.normal(0, std*temp).to(mp.data.device)
            else: # 如果是偏置或 LayerNorm weight 这类1D参数
                    temp = torch.empty_like(mp, device=mp.data.device)
                    temp.normal_(0, std*(mp.view(-1).norm().item() + 1e-16))
            """
            # 按着原代码逻辑与风格的实现
            # if len(param.shape) > 1:
            #     sh = param.shape
            #     #  filter 也就是矩阵的每一行
            #     sh_mul = np.prod(sh[1:])
            #     #  param.view(sh[0], -1).norm(dim=1, keepdim=True):计算每一行的范数  shape = [m, 1]
            #     #  repeat(1, sh_mul): 将这一行的结果都变成这一行的范数
            #     noise_temp = param.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(param.shape)
            #     #  生成一个与 std * param 形状相同的张量，
            #     # 每个元素从均值为 0、标准差为 std * param 对应位置值的正态分布中采样
            #     noise_temp = torch.normal(0, std*noise_temp).to(param.data.device)
            # else:
            #     # 对于 bias 或 LayerNorm 的一维参数
            #     noise_temp = torch.empty_like(param)
            #     noise_temp.normal_(0, std*(param.view(-1).norm().item() + 1e-16))
            # return noise_temp

             

            # 从 N(0, 1)进行缩放，而不是 torch.normal 
            if len(param.shape) > 1:
                sh = param.shape
                sh_mul = np.prod(sh[1:])
                # 每行的L2范数：[m, 1]
                row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)
                # 将其扩展为与param相同形状
                std_matrix = row_norms.repeat(1, sh_mul).view(param.shape)
                # 使用 randn_like，然后缩放为每个元素 std = std * row_norm
                scaler = std * std_matrix
                noise_temp = torch.randn_like(param) * scaler
            else:
                # 对于一维参数，先计算整体L2范数（加上稳定项）
                scale = std * (param.view(-1).norm().item() + 1e-16)
                noise_temp = torch.randn_like(param) * scale

            return noise_temp

        elif mode == "mARWP_fisher":

            """
            https://github.com/nblt/mARWP/blob/main/train_marwp.py

            官方实现： 
            with torch.no_grad():
                noise = []
                for ii, mp in enumerate(model.parameters()):
                    sh = mp.data.shape   # 当前参数张量的形状
                    sh_mul = int(np.prod(sh[1:])) # 每一行（如 conv/filter）中包含的元素个数，用于广播
                    # 如果提供了 Fisher 信息（任务相关的方向不确定性指标）
                    if fisher_arr != []:
                        # 将 Fisher 信息张量调整形状为 [行数, -1]，在每一行上求和（即累计该行的 Fisher 信息）
                        # 将 [行数, 1] 的 Fisher 信息扩展（广播）为与参数形状相同的张量
                        fisher = fisher_arr[ii].view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
                        
                    if len(mp.shape) > 1:
                        # 对于多维参数（如 weight matrix），先计算每一行的 L2 范数,然后 扩展成与 mp 相同的 shape
                        temp = mp.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(mp.shape)
                        # # 使用标准正态分布采样，然后缩放标准差为：args.sigma × 每行范数
                        temp = torch.normal(0, args.sigma*temp).to(mp.data.device)
                    else:
                        temp = torch.empty_like(mp, device=mp.data.device)
                        temp.normal_(0, args.sigma*(mp.view(-1).norm().item() + 1e-16))
                    
                    # ---- Fisher 相关的缩放逻辑（任务感知噪声）----
                    if fisher_arr != []:
                    
                        # 使用 Fisher 信息缩放噪声项：
                        # temp ← temp / sqrt(1 + η × fisher)
                        # 使得在“确定性强”（fisher 大）的方向上抑制噪声，反之增强不确定方向上的噪声

                        temp /= torch.sqrt(1 + args.eta * fisher)
                        
                    noise.append(temp)
                    mp.data.add_(noise[-1])

            
            """


            sh = param.shape
            sh_mul = np.prod(sh[1:])
            if len(param.shape) > 1:
                # 计算每一行的 L2 范数，并广播为和 mp 相同形状
                row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)  # shape = [m, 1]
                std_matrix = row_norms.repeat(1, sh_mul).view(param.shape)

                # 构建噪声标准差张量
                scaler = std * std_matrix

                # 使用标准正态采样后乘以 std，实现与 normal(mean=0, std=...) 等效行为
                noise_temp = torch.randn_like(param) * scaler
            else:
                # 一维参数：整体 L2 范数 + ε 保证数值稳定
                scale = std * (param.view(-1).norm().item() + 1e-16)
                noise_temp = torch.randn_like(param) * scale

            # 在生成噪声的代码中添加
            # assert noise_temp.shape == fisher_param.shape, "Fisher 信息形状与噪声不匹配"
            if fisher_param is not None:
                # 按着原始实现，这里应该对 Fisher_param 按行来计算 
                # 按行 sum 得到每行一个 Fisher 值 , 广播扩展为完整参数 shape
                fisher =  fisher_param.view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
                # 按照 Fisher 信息缩放噪声项： ε ← ε / sqrt(1 + η × F)
                noise_temp /= torch.sqrt(1 +  fisher_scaler* fisher )

            return noise_temp
        

        elif mode == "flatLoRA":
            # 从 N(0, 1)进行缩放，而不是 torch.normal 
            if len(param.shape) > 1:
                sh = param.shape # 如果维度为 m * n
                sh_mul = np.prod(sh[1:]) # 这个就是 n

                # 每行的L2范数：[m, 1]
                row_norms = param.view(sh[0], -1).norm(dim=1, keepdim=True)
                # 将其扩展为与param相同形状
                std_matrix = row_norms.repeat(1, sh_mul).view(param.shape)
                # 使用 randn_like，然后缩放为每个元素 std = std * row_norm
                scaler = std / math.sqrt(param.shape[1]) * std_matrix 

                noise_temp = torch.randn_like(param) * scaler
            else:
                # 对于一维参数，先计算整体L2范数（加上稳定项）
                n = param.shape[0]  # 一维向量的“长度”就是输出维度
                scale = std / math.sqrt(n) * (param.view(-1).norm().item() + 1e-16)

                noise_temp = torch.randn_like(param) * scale

            return noise_temp

        else:
            raise ValueError(f"Unknown noise mode: {mode}")
        
    def _gather_grad_vector(self, model):
        """
        收集所有 requires_grad 且有 grad 的参数梯度，拼接为一维向量
        """
        grad_list = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_list.append(param.grad.detach().clone().view(-1))
        if len(grad_list) == 0:
            raise ValueError("未检测到任何梯度，请检查梯度是否正确反向传播。")
        return torch.cat(grad_list)

    def _assign_grad_vector(self, model, grad_vector):
        """
        将一维梯度向量 grad_vector 拆分并赋值回模型参数的 .grad
        """
        offset = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(grad_vector[offset:offset + numel].view_as(param))
                offset += numel
        if offset != grad_vector.numel():
            raise ValueError("Grad vector size 与参数维度不匹配！")
    
    
    
    
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

        # 在训练之前，将梯度进行归0
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



                if self.args.gradient_accumulation_steps == 1 and  ("rpw_single_jiou" in self.args.rwp_type):
                # 添加一个分支，专门处理 Rwp 按着论文，奇偶不同的情形，作为一起进行合并的情形

                    if step % 2 == 0:
                        pass

                    if step % 2 == 1:
                        pass

                
                else :

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
                            elif self.args.train_method == "finetune":
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
                                

                                logger.debug(f"loss: {loss.item():.4f}")

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
                            # PyTorch中，`backward()`方法
                            # 会计算当前张量（在这里是损失值`loss`）相对于各个需要梯度的参数的梯度，
                            # 并将这些梯度累积到参数的`.grad`属性中。
                            # 这是训练神经网络时的标准步骤，
                            # 为后续优化器更新参数（如 optimizer.step()）提供梯度信息
                            
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
                                
                            # detach()`会创建一个新的张量，与原始张量共享数据，但剥离计算图，
                            # 即新张量不再有梯度历史。这意味着对这个新张量的操作不会影响反向传播，也不会在计算梯度时被追踪。
                            # 这一步通常用于将损失值从计算图中分离出来，
                            # 以便进行后续的数值记录或日志，而不会保留不必要的计算图信息，从而节省内存。
                            tr_loss_step = loss.detach()
                            loss_cleaned_log = tr_loss_step



                            # 这里已经完成了 损失函数和梯度的计算，相当于完成了g0的计算
                            # 这里累积的是lora 部分的梯度信息，因为其他部分都被fixed住了，实际上没有梯度
                            g0 = self._gather_grad_vector(model)
                                
                            # ------------------- 
                            # -------------------
                            # 如果使用 rwp 来进行 更新梯度，在这一步要计算扰动损失 也就是g1
                            # 并且扰动损失要和原来的损失要进行混合
                            if self.args.rwp_type != "":
                                # pass
                                # 这里是 RWP 的损失计算过程 
                                # g0 = self._gather_grad_vector(model)
                                logger.info(f"go-loss-cleaned: {loss_cleaned_log}")
                                
                                
                                # ----------------------------
                                # Step 2: 计算扰动梯度 g1
                                # ----------------------------
                                # Step 2: g₁ - 添加扰动并计算噪声梯度
                                # === Step 2: g₁ ===
                                # 清除梯度缓冲区，确保后续计算 g1 时梯度从零开始累积
                                model.zero_grad()
                                disable_running_stats(model) # 禁用 BatchNorm 的运行统计

                                # 为参数添加扰动
                                rwp_noise_dict = {}
                                # logger.debug(f"RWP add noise- mode : {self.args.rwp_noise_type} noise_std : {self.args.noise_std}")
                                with torch.no_grad():
                                    for name, param in model.named_parameters():
                                        # if not param.requires_grad:
                                        #     continue
                                        apply_noise = False
                                        if "lora" in self.args.rwp_type and "loranew_" in name:
                                            apply_noise = True
                                            
                                        elif "full" in self.args.rwp_type:
                                            apply_noise = True
                                        # 已经训练好的lora部分也不会被添加噪声，只有原来的模型的参数会被添加噪声
                                        elif "origin"in self.args.rwp_type and  ("lora"  not in name):
                                            apply_noise = True


                                        if apply_noise:
                                            fisher_param = None
                                            fisher_scaler = 0
                                            if 'fisher' in self.args.rwp_noise_type:
                                                fisher_param = self.fisher_dict.get(name, None)
                                                fisher_scaler =  self.args.rwp_fisher_app_scaler
                                            
                                            noise = self.generate_noise(
                                                param=param,
                                                mode=self.args.rwp_noise_type,  # one of: standard, matrix, element, filter, fisher
                                                std=self.args.noise_std, 
                                                fisher_param=fisher_param,
                                                fisher_scaler= fisher_scaler
                                            )
                                            param.data.add_(noise)
                                            # 保存为每一层参数添加的噪声，用来还原模型
                                            rwp_noise_dict[name] = noise
                                            # logger.debug(f"[RWP-{self.args.rwp_noise_type}] Injected noise into: {name} | shape: {param.shape}")


                                    
                                # forward + backward with noise → 得到 g₁
                                with self.compute_loss_context_manager():
                                    loss_noisy = self.compute_loss(model, inputs)

                                #-----------
                                logger.info(f'training_step RWP model with {self.args.rwp_noise_type} g1--loss_noise : {loss_noisy.item():.4f}')

                                # 添加正则化处理，如果需要的话
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
                                        f"orthogonal_loss: {orthogonal_loss.item():.4f}; l2_loss: {l2_loss.item():.4f}; loss_noisy: {loss_noisy.item():.4f}; λ1: {lamda_1}; λ2: {lamda_2}")

                                    loss_noisy = loss_noisy + orthogonal_loss * lamda_1 + l2_loss * lamda_2
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

                                    logger.info(f"Nlora_loss: {l1_loss.item():.4f};   loss_noisy: {loss_noisy.item():.4f}; λ1: {lamda_1};")

                                    loss_noisy = loss_noisy + l1_loss * lamda_1
                                elif self.args.lora_strategy.lower() == "inclora":
                                    logger.info(f"inclora loss_noisy: {loss_noisy.item():.4f}")
                                elif self.args.lora_strategy.lower() == "lora_l2":
                                    l2_loss = 0.
                                    for name, param in model.named_parameters():
                                        if "loranew_" in name:
                                            l2_loss += torch.norm(param, p=2)

                                    logger.info(f"lora_l2 l2_loss: {l2_loss.item():.4f}; loss_noisy: {loss_noisy.item():.4f};  λ2: {lamda_2}")
                                    lamda_2 = self.args.lamda_2
                                    loss_noisy = loss_noisy + l2_loss * lamda_2
                                ######################################################################
                                logger.debug(f"g1---sum_loss(with some ): {loss_noisy.item():.4f}")

                                
                                #---多GPU，平均损失---------
                                if self.args.n_gpu > 1:
                                    loss_noisy = loss_noisy.mean()
                                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                                    loss_noisy = loss_noisy / self.args.gradient_accumulation_steps

                                


                                # loss 的 backward
                                if self.do_grad_scaling:
                                    self.scaler.scale(loss_noisy).backward()
                                elif self.use_apex:
                                    with amp.scale_loss(loss_noisy, self.optimizer) as scaled_loss:
                                        scaled_loss.backward()
                                elif self.deepspeed:
                                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                                    loss_noisy = self.deepspeed.backward(loss_noisy)
                                else:
                                    loss_noisy.backward()

                                loss_noisy_log = loss_noisy.detach()
                                logger.info(f"g1-loss_noisy: {loss_noisy_log.item():.4f}")

                                # 保存添加噪声扰动的模型的参数的fisher信息 
                                # 这里是根据 添加噪声扰动后的梯度计算得到的fisher信息矩阵
                                if 'fisher' in self.args.rwp_noise_type:
                                    with torch.no_grad():
                                        for name, param in self.model.named_parameters():
                                            if param.grad is not None:
                                                grad_squared = param.grad.detach() ** 2
                                                # 使用指数滑动平均更新 fisher_dict
                                                self.fisher_dict[name] = (self.fisher_dict[name] *  self.args.rwp_fisher_cal_scaler + grad_squared )


                                # 收集扰动梯度 g1
                                g1 = self._gather_grad_vector(model)

                                # ----------------------------
                                # Step 3: 恢复参数并混合梯度
                                # ----------------------------
                                # 恢复参数值
                                # 将模型从扰动状态还原回原始状态,然后进行训练
                                # 正确代码
                                with torch.no_grad():
                                    for name, noise in rwp_noise_dict.items():  # 使用 .items()
                                        param = model.get_parameter(name)       # 根据名称获取参数
                                        param.data.sub_(noise)                  # 减去噪声

                                # === Gradient mix ===
                                # Step 3: 梯度融合
                                # g ← a·g0 + b·g1
                                # 其中 g0 是没有扰动的， g1 是有扰动的 ,其对应的权重系数分别为  rwp_a，  rwp_b
                                loss_rwp  = self.args.rwp_a * loss_cleaned_log + self.args.rwp_b* loss_noisy_log
                                logger.info(f"loss_rwp：{loss_rwp}；g0-cleaned_loss: {loss_cleaned_log.item():.4f}; g1-noisy_loss: {loss_noisy_log.item():.4f}; a :{self.args.rwp_a}, b :{self.args.rwp_b}")
                                logger.info(f"{loss_rwp} = {self.args.rwp_a} * {loss_cleaned_log.item():.4f} + {self.args.rwp_b} *{loss_noisy_log.item():.4f} ")

                                tr_loss_step = loss_rwp
                                
                                # 进行梯度混合
                                mixed_grad = self.args.rwp_a * g0 + self.args.rwp_b * g1 

                                self._assign_grad_vector(model, mixed_grad)

                        elif self.args.train_method == "finetune":
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
                            

                            logger.debug(f"loss: {loss.item():.4f}")

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
                    # elif hasattr(self.optimizer, "first_step") and hasattr(self.optimizer, "second_step"):
                    elif 'sam' in self.args.optimizer_type:
                        # === For SAM Optimizer: first_step-second_step two phases ===
                  
                        
                        # first forward-backward
                        self.optimizer.first_step(zero_grad=True)

                        # second forward-backward
                        # 第二次计算loss，这里计算loss是只根据一个batch来计算loss
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

                        # 第二次反向传播, 
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

                    # 调用optimner更新梯度，结束之后进行梯度清除，
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
        self, eval_dataset: Dataset, output_dir, distrub_name = 'originModel',x_range=(-1, 1), y_range=(-1, 1), num_points=10, max_batches=2, sample_batches=False,

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
        # 是否是训练后的模型，还是加载已经训练好的模型
        if args.do_train:
            # 如果进行训练，那么模型的new_lora就是新的当前任务添加的lora
            # Flag_newtaskLoRA = ''  # 如果只干扰训练后的新的任务的lora
            # Flag_onlyLoRA = ''  # 如果只干扰lora 部分
            # Flag_fullModel =  'fullModel'  # 如果干扰模型的所有参数
            surf_file = os.path.join(
                    output_dir, f"{args.lora_strategy}_{distrub_name}_T5large_testData.h5")
        else:
            # 如果不进行训练，只展示加载模型的 lossland ，则对模型的所有参数(不包括new_lora)都进行干扰，
            surf_file = os.path.join(output_dir, f"{distrub_name}_T5large__testData.h5")


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

        model = model.to(device=device)
        model.eval()  # 确保模型在 eval 模式

        # --------------------- 生成参数扰动阶段阶段 ---------------------
        # 确定需要扰动的参数名称
        # 根据不同的微调方法（Nlora或lora）确定需要保存原始值的参数
        original_params_to_perturb = {}
        for name, param in model.named_parameters():
            if args.do_train:
                if distrub_name == 'originModel':
                    pass
            else:# 不是训练后直接评估，加载的模型不应该包括新添加的 new_lora 部分
                if distrub_name == 'originModel':
                    if "loranew_" not in name : 
                        original_params_to_perturb[name] = param.data.clone()
                elif distrub_name == 'trainedLoRA':
                    if "lora_" in name  and "loranew_" not in name: 
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


                # 计算损失 (统一使用 prediction_step)
                total_loss = 0.0
                for batch in all_batches:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    loss, _, _ = self.prediction_step(
                        model, 
                        inputs, 
                        prediction_loss_only=True, 
                        ignore_keys=None
                    )
                    # 兼容返回为张量/float
                    total_loss += loss.item() if hasattr(loss, "item") else float(loss)
                loss_grid[i, j] = total_loss / len(all_batches)

                # # 计算损失
                # total_loss = 0.0
                # for batch in all_batches:
                #     inputs = {k: v.to(device) for k, v in batch.items()}
                #     with torch.cuda.amp.autocast(enabled=args.fp16):  # 支持混合精度
                #         outputs = model(**inputs)
                #         loss = F.cross_entropy(
                #             outputs.logits.view(-1, outputs.logits.size(-1)),
                #             inputs["labels"].view(-1)
                #         )
                #     total_loss += loss.item()

                # loss_grid[i, j] = total_loss / len(all_batches)





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
        eval_dataset,
        output_dir,
        name="hessian",
        max_batches=10,
        sample_batches=False,
        use_gpu=True,

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

        model_type = 'lora'
        if model_type == 'fintune':
            #如果模型是经过微调的结果，那么整个模型都要计算Hessian矩阵
            pass
        if model_type == 'lora':
            # 如果模型是经过lora微调的结果，那么只需要计算lora部分的Hessian矩阵, 
            # 因为原来的模型是冻结的，没有更新所以不需要计算 Hessian 矩阵
            # 新添加的部分是默认初始化，也不需要计算 Hessian 矩阵
            hessian_file_lora = os.path.join(output_dir, f"{name}_lora_only-predictDataset_lanczos.h5")


        logger.info(f'***5***--5-3**1 begin init   ')
        logger.info(
            f'***5***--5-3**1 use distribute ---- {torch.distributed.is_initialized()}')
        args = self.args
        device = args.device

        # 创建独立模型副本（关键改进点1：隔离原始模型）
        model = self.model
        model = self._wrap_model(model, training=False)

        # 混合精度处理
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=device)

        model = model.to(device=device)
        model.eval()

        logger.info(f'***5***--LORA Hessian**1 finish init ')

        # --------------------- 数据准备阶段 ---------------------
        logger.info(f'***5***--5-3**2 begin load data   ')
        dataloader = self.get_eval_dataloader(eval_dataset)
        all_batches = list(dataloader)

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            # all_batches = all_batches[rank::world_size]  # 数据分片
            logger.info(
                f'***5***--LORA Hessian**2 distribute yes, rank:{rank}, total_batches:{len(all_batches)}')
        else:
            logger.info(f'***5***--5-3**4 distribute--no ')
            world_size = 1
            rank = 0

        # 批量采样逻辑（改进点3：内存优化）
        # logger.info(
        #     f'***5***--5-3**2-1 if  {len(all_batches)} , {max_batches} ')
        # if len(all_batches) > max_batches:
        #     if sample_batches:
        #         indices = np.random.choice(
        #             len(all_batches), max_batches, replace=False)
        #         all_batches = [all_batches[i] for i in indices]
        #     else:
        #         all_batches = all_batches[:max_batches]

        # logger.info(f'***5***--5-3**2 finish load data ')
        logger.info(f'***5***--5-3**2 finish load data ')
        def dataloader_stream_sampler(dataloader, max_batches, sample_batches, rank=0, world_size=1):
            """
            支持分布式的数据流式 batch 采样器。
            - dataloader: torch 的数据加载器
            - max_batches: 最多取几个 batch
            - sample_batches: True 随机采样，False 顺序取
            - rank/world_size: 分布式环境下的本地索引
            """
            # 把所有batch的index组成一个list
            indices = list(range(len(dataloader)))
            if sample_batches:
                np.random.shuffle(indices)
            indices = indices[:max_batches]

            # 分布式切分
            indices = indices[rank::world_size]

            # 只取指定的batch
            for i, batch in enumerate(dataloader):
                if i in indices:
                    yield batch
                if len(indices) > 0 and i > max(indices):
                    break
        logger.info(f'***5***--5-3**2 begin load data ')
        batch_iterator = dataloader_stream_sampler(dataloader, max_batches, sample_batches, rank=rank, world_size=world_size)
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

            def lanczos_algorithm(self, hvp_func, dim, order=5, num_splits=4, random_seed=0):
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

            def block_lanczos(self, hvp_func, dim, k=10, block_size=2):
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


        if model_type == 'lora':
            dom_eigs_lora = []
            # 获取参数集合（改进点8：动态参数处理）
            lora_params = {}
            for name, param in model.named_parameters():
                # 只需要关注哪些需要进行更新的参数
                if param.requires_grad == True:
                    lora_params[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )

        calculator = HessianCalculator(model, device)

        # 分布式通信初始化（改进点9：分布式支持）

        # if torch.distributed.is_initialized():
        #     logger.info(f'***5***--5-3**4 distribute--yes ')
        #     world_size = torch.distributed.get_world_size()
        #     rank = torch.distributed.get_rank()
        #     all_batches = all_batches[rank::world_size]  # 数据分片
        #     logger.info(f'***5***--5-3**4 {len(all_batches)} ')
        # else:
        #     logger.info(f'***5***--5-3**4 distribute--no ')
        #     world_size = 1
        #     rank = 0

        try:
            # for batch in tqdm(all_batches, desc=f"Rank {rank}: Processing"):
            for batch_idx, batch in enumerate(tqdm(batch_iterator, desc=f"Rank {rank}: Processing")):
                # 重置模型参数（关键改进点10：消除参数污染）
                # model.load_state_dict(original_state)
                # 恢复参数时仅操作需要修改的部分（优化点1）
                for name in original_params_to_calculate_hessian:
                    model.state_dict()[name].copy_(
                        original_params_to_calculate_hessian[name])

                
                if model_type == 'lora':
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
                    # eigvals = calculator.lanczos_algorithm(
                    #     hvp_lora, dim=sum(p.numel() for p in lora_params.values()), order=10, num_splits=4)
                    # eigvals = calculator.lanczos_algorithm(
                    #     hvp_lora, dim=sum(p.numel() for p in lora_params.values()), order=10, num_splits=4)
                    dom_eigs_lora.append(eigvals[-1])  # 或者 eigvals[0]，取决于实现
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
        
            if model_type == 'lora':
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
            if model_type == 'lora':
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

    def compute_task_aware_expected_sharpness(self,
        eval_dataset,
        output_dir,
        distrub_name='originModel',
        sigma=0.0015, 
        n_samples=1000, 
        max_batches=10,
        sample_batches=False,
        ):
        # import torch.distributed
        """
        计算 Task-Aware Expected Sharpness，支持精度切换、分布式和自适应 batch。
        """

        # ========== 初始化与精度设置 ==========
        logger.info('begin init')
        args = self.args
        device = args.device

        # 独立模型副本
        model = self.model
        model = self._wrap_model(model, training=False)
        
        model = model.to(device=device)
        model.eval()
        logger.info(f'***5***--LORA**1 finish init')

        # ========== 数据准备，分布式与采样 ==========
        logger.info(f'***5***--5-3**2 begin load data')
        dataloader = self.get_eval_dataloader(eval_dataset)
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            logger.info(f'***5***--LORA Hessian**2 distribute yes, rank:{rank}, max_batches:{max_batches}')
        else:
            logger.info(f'***5***--5-3**4 distribute--no')
            world_size = 1
            rank = 0

        # 定义 batch 采样器
        def dataloader_stream_sampler(dataloader, max_batches, sample_batches, rank=0, world_size=1):
            indices = list(range(len(dataloader)))
            if sample_batches:
                np.random.shuffle(indices)
            indices = indices[:max_batches]
            indices = indices[rank::world_size]
            for i, batch in enumerate(dataloader):
                if i in indices:
                    yield batch
                if len(indices) > 0 and i > max(indices):
                    break

        
        batch_iterator = dataloader_stream_sampler(dataloader,max_batches, sample_batches, rank=rank, world_size=world_size)

        # ========== 确定扰动参数集合 ==========
        original_params_to_perturb = {}
        logger.info(f"Adding {distrub_name} type nois")
        for name, param in model.named_parameters():
            if args.do_train:
                if distrub_name == 'originModel':
                    pass
            else:
                if distrub_name == 'originModel':
                    if "loranew_" not in name:
                        original_params_to_perturb[name] = param.data.clone()
                        
                elif distrub_name == 'trainedLoRA':
                    if "lora_" in name and "loranew_" not in name:
                        original_params_to_perturb[name] = param.data.clone()

        # ======= 2. 计算 baseline loss（用 prediction_step 统一接口） =======
        batch_iterator_baseline = dataloader_stream_sampler(
            dataloader, max_batches, sample_batches, rank=rank, world_size=world_size)
        baseline_loss_sum = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in batch_iterator_baseline:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _, _ = self.prediction_step(
                    model, batch, prediction_loss_only=True, ignore_keys=None
                )
                baseline_loss_sum += loss.item() if hasattr(loss, "item") else float(loss)
                num_batches += 1
        baseline_loss = baseline_loss_sum / max(1, num_batches)

        # ======= 3. 累加 expected sharpness（每次扰动都调用 prediction_step） =======
        sharpness_sum = 0.0
        sharpness_samples = []
        sharpness_samples_abs = []

        for i in range(n_samples):
            perturb_eps = {}
            with torch.no_grad():
                for name in original_params_to_perturb:
                    param = model.state_dict()[name]
                    epsilon = torch.normal(
                        mean=0.0, std=sigma, size=param.shape, device=param.device, dtype=param.dtype)
                    param.add_(epsilon)
                    perturb_eps[name] = epsilon

            # 采样 batch 重新迭代，loss计算用 prediction_step
            batch_iterator = dataloader_stream_sampler(
                dataloader, max_batches, sample_batches, rank=rank, world_size=world_size)
            perturbed_loss_sum = 0.0
            num_batches = 0
            with torch.no_grad():
                for batch in batch_iterator:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss, _, _ = self.prediction_step(
                        model, batch, prediction_loss_only=True, ignore_keys=None
                    )
                    perturbed_loss_sum += loss.item() if hasattr(loss, "item") else float(loss)
                    num_batches += 1
            perturbed_loss = perturbed_loss_sum / max(1, num_batches)

            # 恢复参数
            with torch.no_grad():
                for name in original_params_to_perturb:
                    param = model.state_dict()[name]
                    param.sub_(perturb_eps[name])

            this_sharpness = perturbed_loss - baseline_loss
            sharpness_sum += this_sharpness
            sharpness_samples.append(this_sharpness)
            sharpness_samples_abs.append(abs(this_sharpness))

        if torch.distributed.is_initialized():
            # 所有rank各自的sharpness_samples
            local_samples = [float(x) for x in sharpness_samples]
            # 用all_gather_object收集到主进程
            all_samples = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(all_samples, local_samples)
            # 扁平化
            all_samples = sum(all_samples, [])
            if rank == 0:
                sharpness_samples = all_samples
                sharpness_samples_abs = [abs(x) for x in all_samples]
        else:
            if rank == 0:
                sharpness_samples = [float(x) for x in sharpness_samples]
                sharpness_samples_abs = [float(x) for x in sharpness_samples_abs]

        # 然后再统一做 mean/max/percentile等统计
        # ==== 保存结果为 JSON ====
        # 只主进程保存
        if rank == 0:
            expected_sharpness = sharpness_sum / n_samples
            mean_absolute_sharpness = np.mean(sharpness_samples_abs)
            max_absolute_sharpness = np.max(sharpness_samples_abs)
            max_signed_sharpness = np.max(sharpness_samples)
            min_signed_sharpness = np.min(sharpness_samples)
            std_signed_sharpness = float(np.std(sharpness_samples))
            percentile90_abs = float(np.percentile(sharpness_samples_abs, 90))
            percentile95_abs = float(np.percentile(sharpness_samples_abs, 95))

            logger.info(
                f"Task-aware Expected Sharpness (sigma={sigma}, n={n_samples}): {expected_sharpness:.6f}, "
                f"MeanAbs: {mean_absolute_sharpness:.6f}, MaxAbs: {max_absolute_sharpness:.6f}"
            )
        
    
            result = {
                "distrub_name": distrub_name,
                "sigma": sigma,
                "n_samples": n_samples,
                "max_batches": max_batches,
                "expected_sharpness": expected_sharpness,
                "mean_absolute_sharpness": mean_absolute_sharpness,
                "max_absolute_sharpness": max_absolute_sharpness,
                "baseline_loss": baseline_loss,
                "output_dir": output_dir,
                "sharpness_samples": [float(x) for x in sharpness_samples],       # 转 float 以便json序列化
                "sharpness_samples_abs": [float(x) for x in sharpness_samples_abs],
                "max_signed_sharpness": float(max_signed_sharpness),
                "min_signed_sharpness": float(min_signed_sharpness),
                "std_signed_sharpness": std_signed_sharpness,
                "percentile90_abs": percentile90_abs,
                "percentile95_abs": percentile95_abs,
            }
            json_path = os.path.join(output_dir, f"task_aware_expected_sharpness_{distrub_name}.json")
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Sharpness result saved to: {json_path}")

        return expected_sharpness