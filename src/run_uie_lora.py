#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
from torch.nn import Linear
import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import torch
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig  # add

from uie_collator import DataCollatorForUIE
from uie_dataset_lora import gen_cache_path

from uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics
from sklearn.metrics import f1_score ,accuracy_score
from model.llama import LlamaForCausalLM_with_lossmask

# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    # added for AutoCL
    lora_dim: Optional[int] = field(
        default=8,
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={
                      "help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    # full_test_task_config_dir: Optional[str] = None  # NEW
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={
            "help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to preappend dataset name before the task input."}
    )


@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={
                          "help": "Whether to run the model as a demo in the terminal."})
    lamda_1: float = field(default=0.5)
    lamda_2: float = field(default=0)
    train_method: str = field(default="finetune")
    finetune_strategy: str = field(default="full")  # 'full', 'ewc', 'er'
    lora_strategy: str = field(default="lora")  # 'lora', 'Nlora', 'er'
    # optimizer_type: str = field(default="sam-adam")  # 'sgd' 或 'sam'
    momentum: float = field(default=0)          # 仅用于 sgd 或 sam
    rho: float = field(default=0.05)              # sam 特有超参数
    do_flatminal: bool = field(default=False)

    optimizer_type: str = field(default="adamw_hf", metadata={
                                "help": "Custom optimizer type, supports sam variants."})



    # ✅ 线性探针相关参数  training_args.train_method 参数设置为 use_probe':
    probe_num_classes: int = field(
        default=14,
        metadata={"help": "Number of classes for the linear probe head."},
    )
    probe_feature_mode: str = field(
        default="cls",  # 支持 cls, eos, mean
        metadata={"help": "Feature pooling strategy for probe. Options: [cls, eos, mean]."},
    )
    # use_probe: bool = field(default=False)
    zero_shot: bool = field(default=False)

    

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, UIETrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # training_args.get_process_log_level()
    log_level = 10
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"model_args {model_args}")
    logger.info(f"data_args parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # data_cache_dir = gen_cache_path('logs_and_outputs/order_1', data_args)
    # if getattr(training_args, "use_probe", False):
    # 这个设定下 训练集和测试集都只有一个
    if training_args.train_method.lower() == 'use_probe' or training_args.do_flatminal:
        data_cache_dir = gen_cache_path('logs_and_outputs_linearProbe/order_1', data_args)
    else:
        data_cache_dir = gen_cache_path('logs_and_outputs_ShowResults/order_1', data_args)

    # Get the UIE dataset
    # 输出所有传入参数信息
    logger.info("Loading dataset with the following parameters:")
    logger.info(
        f"Dataset script path: {os.path.join(CURRENT_DIR, 'uie_dataset_lora.py')}")
    logger.info(f"data_dir: {data_args.data_dir}")
    logger.info(f"task_config_dir: {data_args.task_config_dir}")
    logger.info(f"instruction_file: {data_args.instruction_file}")
    logger.info(f"instruction_strategy: {data_args.instruction_strategy}")
    logger.info(f"cache_dir: {data_cache_dir}")
    logger.info(
        f"max_num_instances_per_task: {data_args.max_num_instances_per_task}")
    logger.info(
        f"max_num_instances_per_eval_task: {data_args.max_num_instances_per_eval_task}")
    logger.info(f"num_examples: {data_args.num_examples}")
    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "uie_dataset_lora.py"),
        data_dir=data_args.data_dir,
        task_config_dir=data_args.task_config_dir,
        instruction_file=data_args.instruction_file,
        instruction_strategy=data_args.instruction_strategy,
        cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        num_examples=data_args.num_examples
    )

    # if getattr(training_args, "use_probe", False):
    if training_args.train_method.lower() == 'use_probe':
        # --- 自动构建 label2id ---
        all_labels = set()
        for split in raw_datasets.keys():
            for example in raw_datasets[split]:
                all_labels.add(example["Instance"]["label"])
        label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}

        logger.info(f"Label2id mapping: {label2id}")

        # --- 给每个 split 添加 class_label 字段 ---
        def encode_class_label(example):
            example["class_label"] = label2id[example["Instance"]["label"]]
            return example

        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].map(encode_class_label)

    # raw_datasets.cleanup_cache_files()

    # Load full test dataset for evaluation
    # full_test_datasets = None
    # if data_args.full_test_task_config_dir is not None:
    #     full_test_datasets = load_dataset(
    #         os.path.join(CURRENT_DIR, "uie_dataset_lora.py"),
    #         data_dir=data_args.data_dir,
    #         task_config_dir=data_args.full_test_task_config_dir,
    #         instruction_file=data_args.instruction_file,
    #         instruction_strategy=data_args.instruction_strategy,
    #         cache_dir=data_cache_dir,
    #         max_num_instances_per_task=data_args.max_num_instances_per_task,
    #         max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
    #         num_examples=data_args.num_examples,
    #     )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'adapter' in model_args.model_name_or_path:  # load lora-config
        logger.info(f'***1***-adapter  if {model_args.model_name_or_path}')
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        logger.info(f'***1***-adapter  config: {config}')
        if 'llama' in model_args.model_name_or_path.lower():
            tokenizer = transformers.LlamaTokenizer.from_pretrained(
                config.base_model_name_or_path)
            config.bos_token_id = 1
            config.eos_token_id = 2
            config.pad_token_id = 1
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            tokenizer.pad_token_id = 1
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        logger.info(f'***1***1-llama  {model_args.model_name_or_path.lower()}')
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.bos_token_id = 1
        config.eos_token_id = 2
        config.pad_token_id = 1
        logger.info(f'***1***1-llama  config: {config}')
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1
    else:  # load original config
        logger.info(f'***1***--1-else  {model_args.model_name_or_path}')
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    logger.info('*****2***Load model_class**')
    if 'llama' in model_args.model_name_or_path.lower():  # add llama
        logger.info(
            f'***2***--2-llama  {model_args.model_name_or_path.lower()}')
        model_class = LlamaForCausalLM_with_lossmask
        tokenizer.padding_side = 'left'
    else:
        logger.info(
            f'***2***--2-else  {model_args.model_name_or_path.lower()}')
        model_class = AutoModelForSeq2SeqLM

    if 'adapter' in model_args.model_name_or_path:  # add lora-adapter to the original model
        logger.info(f'***3***--3-adapter  {model_args.model_name_or_path}')
        model = model_class.from_pretrained(config.base_model_name_or_path)

        # === 🔥 这里就直接加 probe_head ===
        # if getattr(training_args, "use_probe", False):
        if training_args.train_method.lower() == 'use_probe':
            logger.info(f"Adding probe head in MAIN function (not only in Trainer)")
            hidden_size = model.config.hidden_size
            num_probe_classes = training_args.probe_num_classes
            model.probe_head = Linear(hidden_size, num_probe_classes)
            model.probe_head.to(training_args.device)  # 注意迁移到 GPU

            # 只冻结其他参数
            for name, param in model.named_parameters():
                if name.find("probe_head")!= -1:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        logger.info(
            f'***3***--3-llama  {model_args.model_name_or_path.lower()}')
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None
        )

        # === 🔥 这里就直接加 probe_head ===
        # if getattr(training_args, "use_probe", False):
        if training_args.train_method.lower() == 'use_probe':
            logger.info(f"Adding probe head in MAIN function (not only in Trainer)")
            hidden_size = model.config.hidden_size
            num_probe_classes = training_args.probe_num_classes
            model.probe_head = Linear(hidden_size, num_probe_classes)
            model.probe_head.to(training_args.device)  # 注意迁移到 GPU

            # 只冻结其他参数
            for name, param in model.named_parameters():
                if name.find("probe_head")!= -1:
                    param.requires_grad = True
                else:
                    param.requires_grad = False


        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
    else:
        logger.info(f'***3***--3-else  {model_args.model_name_or_path}')
        logger.info(f'***3***--3-else  config:{config}')
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # === 🔥 这里就直接加 probe_head ===
        # if getattr(training_args, "use_probe", False):
        if training_args.train_method.lower() == 'use_probe':
            logger.info(f"Adding probe head in MAIN function (not only in Trainer)")
            hidden_size = model.config.hidden_size
            num_probe_classes = training_args.probe_num_classes
            model.probe_head = Linear(hidden_size, num_probe_classes)
            model.probe_head.to(training_args.device)  # 注意迁移到 GPU

            # 只冻结其他参数
            for name, param in model.named_parameters():
                if name.find("probe_head")!= -1:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
        logger.info(
            f'***3***--3- LoraConfig** task_type:{TaskType.SEQ_2_SEQ_LM},  ')
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1
        )
        logger.info(f'***3***--3- LoraConfig** peft_config:{peft_config},  ')

        # logger.info(f'***3***--3- isinstance(peft_config, PromptLearningConfig):{isinstance(peft_config, PromptLearningConfig)},  ')

        model = get_peft_model(model, peft_config)

    model.resize_token_embeddings(len(tokenizer))

    if 'llama' in model_args.model_name_or_path.lower():
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = 1

    

    # fix lora_A/B (bases of previous LoRA parameters, loaded in "load_adapter"[peft_momdel.py])
    # fine-tune loranew_A/B (initialized in "update_layer"[lora.py])
    # optional: lora_A/B is trainable but should not move too far from lorapre_A/B
    # (constrained in "training_step"[uie_trainer_lora.py])
    if training_args.do_train:
        if training_args.train_method.lower() == 'lora':
            if training_args.lora_strategy.lower() == 'olora':
                logger.debug(f'---set paramter:olora---')
                for name, param in model.named_parameters():
                    if name.find("loranew_") != -1:
                        param.requires_grad = True
                    elif name.find("lora_") != -1:
                        param.requires_grad = False
                    # this module should always be frozen because we change the vocabulary
                    elif name.find("shared") != -1:
                        param.requires_grad = False
                    
                    logger.info(f"Param: {name}, requires_grad: {param.requires_grad}")
            elif training_args.lora_strategy.lower() == 'nlora':
                logger.debug(f'---set paramter:nlora---')
                for name, param in model.named_parameters():
                    if name.find("loranew_") != -1:
                        param.requires_grad = True
                    elif name.find("lora_") != -1:
                        param.requires_grad = False
                    # this module should always be frozen because we change the vocabulary
                    elif name.find("shared") != -1:
                        param.requires_grad = False
                    logger.info(f"Param: {name}, requires_grad: {param.requires_grad}")
            elif training_args.lora_strategy.lower() == 'inclora':
                logger.debug(f'---set paramter:inclora---')
                for name, param in model.named_parameters():
                    if name.find("loranew_") != -1:
                        param.requires_grad = True
                    elif name.find("lora_") != -1:
                        param.requires_grad = False
                    # this module should always be frozen because we change the vocabulary
                    elif name.find("shared") != -1:
                        param.requires_grad = False
                    logger.info(f"Param: {name}, requires_grad: {param.requires_grad}")




        # 添加线性探针之后的表现
        elif training_args.train_method.lower() == 'use_probe':
            logger.debug(f'---set paramter:use_probe---')
            for name, param in model.named_parameters():
                if name.find("probe_head")!= -1:
                    param.requires_grad = True
                elif name.find("loranew_") != -1:
                    param.requires_grad = False
                elif name.find("lora_") != -1:
                    param.requires_grad = False
                # this module should always be frozen because we change the vocabulary
                elif name.find("shared") != -1:
                    param.requires_grad = False

                logger.info(f"Param: {name}, requires_grad: {param.requires_grad}")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(
                range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples))
            
    

    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        # pad_to_multiple_of=8 if training_args.fp16 else None,
        pad_to_multiple_of=None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file,
        use_probe=getattr(training_args, "use_probe", False), #   # 兼容旧版 training_args
    )

    if training_args.do_predict:
        sample = predict_dataset[0]
        batch = data_collator([sample])
        logger.info(f"Sample batch keys: {batch.keys()}")
        logger.info(
            f"Sample input_ids decoded: {tokenizer.decode(batch['input_ids'][0])}")

    if training_args.do_train:
        sample = train_dataset[0]
        batch = data_collator([sample])
        logger.info(f"Sample batch keys: {batch.keys()}")
        logger.info(
            f"Sample input_ids decoded: {tokenizer.decode(batch['input_ids'][0])}")

    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(
            predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result
    
    def compute_probe_metrics(dataset, preds, save_prefix=None):
        """
        计算 Linear Probe 的分类任务指标，包括 Accuracy、Macro-F1、Micro-F1。
        同时按 Task 和 Dataset 进行细粒度统计。
        
        参数：
            dataset: hf Dataset对象，包含每条样本的 Task, Dataset, Instance(label)
            preds: 预测 logits，(N, num_classes) numpy array
            save_prefix: 保存预测文件前缀（可选）
        返回：
            metrics: 包含总体指标和分组指标的字典
        """
        # --- 预测类别 ---
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        pred_labels = np.argmax(preds, axis=-1)

        # --- 真实标签 ---
        references = []
        for example in dataset:
            references.append(example["Instance"]["label"])

        # 将 ground truth 转为索引
        # 注意：这里需要传入 label2id 字典，如果你的 label 本身就是int就不用转
        if isinstance(references[0], str):
            # 默认假设label是str，需要建立label2id
            unique_labels = sorted(list(set(references)))
            label2id = {label: idx for idx, label in enumerate(unique_labels)}
            references = [label2id[label] for label in references]

        references = np.array(references)

        # --- 计算总体指标 ---
        acc = accuracy_score(references, pred_labels)
        macro_f1 = f1_score(references, pred_labels, average='macro')
        micro_f1 = f1_score(references, pred_labels, average='micro')

        result = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1
        }

        # --- 按 Task 分组 ---
        groups = [example["Task"] for example in dataset]
        result_per_task = {}
        for task in set(groups):
            task_mask = [g == task for g in groups]
            task_preds = pred_labels[task_mask]
            task_refs = references[task_mask]
            result_per_task[f"task_{task}_acc"] = accuracy_score(task_refs, task_preds)
            result_per_task[f"task_{task}_macro_f1"] = f1_score(task_refs, task_preds, average='macro')

        result.update(result_per_task)

        # --- 按 Dataset 分组 ---
        categories = [example["Dataset"] for example in dataset]
        result_per_category = {}
        for cat in set(categories):
            cat_mask = [g == cat for g in categories]
            cat_preds = pred_labels[cat_mask]
            cat_refs = references[cat_mask]
            result_per_category[f"dataset_{cat}_acc"] = accuracy_score(cat_refs, cat_preds)
            result_per_category[f"dataset_{cat}_macro_f1"] = f1_score(cat_refs, cat_preds, average='macro')

        result.update(result_per_category)

        # --- 可选：保存每条预测 ---
        if save_prefix is not None:
            output_dir = os.path.join(training_args.output_dir, f"{save_prefix}_probe_predictions.jsonl")
            with open(output_dir, "w", encoding="utf-8") as fout:
                for example, pred_idx in zip(dataset, pred_labels):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": int(pred_idx)
                    }) + "\n")

        return {k: round(v, 4) for k, v in result.items()}

    print(
        f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    if getattr(training_args, "use_probe", False):
        compute_metrics_fn = compute_probe_metrics
    else:
        compute_metrics_fn = compute_rouge_metrics

    
    trainer = UIETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            DenserEvalCallback] if training_args.denser_evaluation else None
    )

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        peft_model_id = training_args.output_dir + "/adapter"
        trainer.model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)


        # === 添加：保存 probe_head 权重 ===
        # if getattr(training_args, "use_probe", False):
        if training_args.train_method.lower() == 'use_probe':
            probe_save_dir = os.path.join(training_args.output_dir, "probe_head")
            os.makedirs(probe_save_dir, exist_ok=True)

            torch.save({
                "state_dict": {
                    "weight": trainer.model.probe_head.weight.detach().cpu(),
                    "bias": trainer.model.probe_head.bias.detach().cpu()
                },
                "config": {
                    "in_features": trainer.model.probe_head.in_features,
                    "out_features": trainer.model.probe_head.out_features
                }
            }, os.path.join(probe_save_dir, "probe_head.bin"))

            logger.info(f"Saved probe head to {probe_save_dir}/probe_head.bin")


        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)



    # Evaluation
    results = {}
    # in case the batch is shorter than max length, the output should be padded
    max_new_tokens = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    repetition_penalty = data_args.repetition_penalty

    if training_args.do_predict:
        logger.info("*** Prediction ***")
        logger.info("*** Loading CheckPoint ***")

        if (not training_args.do_train):
            # 即使不进行训练，输出结果也是training_args.output_dir
            trainer.args.output_dir = training_args.output_dir

        # predict_dataset = full_test_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples))

        print(f"-------[DEBUG] Predict dataset size: {len(predict_dataset)}")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )

        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(
                predict_dataset)
        )
        metrics["predict_samples"] = min(
            max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)

    if training_args.do_flatminal:
        pass

        # 
        analyse_result_pth = training_args.output_dir
        logger.debug(f"*** do_flatminal  save result in {analyse_result_pth} ***")

        # Debug 查看模型的参数到底是啥样子的，然后后面设置对哪些参数进行扰动
        logger.debug("check before eval_flatminal")
        for name, param in model.named_parameters():
            logger.debug(f' name:{name}, requires_grad:{param.requires_grad}')

        logger.debug(f'***5***--5-1 begin analyse_flat_minima  ')
        flatminal_dataset = predict_dataset


        # **1. 直接使用 trainer.model（已包含 LoRA 适配器）**
        # **3. 计算损失景观**
        logger.debug(
            f'***5***--5-2 compute_loss_landscape output_dir={analyse_result_pth}  ')
        trainer.compute_loss_landscape(flatminal_dataset, analyse_result_pth)

        

    # return results


if __name__ == "__main__":
    main()
