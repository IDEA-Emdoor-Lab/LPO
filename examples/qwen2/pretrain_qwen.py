# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import os
from functools import partial
from typing import Union

import torch
import torch.nn.functional as F
import torch._dynamo
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetXPO,
    GPTDatasetConfig,
    MockGPTDataset,
)
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron_patch.arguments import get_patch_args
from megatron_patch.data import build_pretrain_dataset_from_original

from megatron_patch.data.utils import get_batch_on_this_tp_rank_original, get_batch_on_this_tp_rank_idxmap_sft, get_batch_on_this_tp_rank_idxmap_xpo
from megatron_patch.model.qwen2.layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron_patch.model.qwen2.model import GPTModel
from megatron_patch.model.qwen2.transformer_config import Qwen2TransformerConfig
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron.core.packed_seq_params import PackedSeqParams

torch._dynamo.config.suppress_errors = True


def model_provider(
    pre_process=True, post_process=True
) -> Union[GPTModel]:

    args = get_args()
    build_tokenizer(args)
    print_rank_0("building qwen2 model ...")

    config = core_transformer_config_from_args(args, Qwen2TransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"

    if use_te:
        print_rank_0("building qwen2 model in TE...")
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
    else:
        print_rank_0("building qwen2 model in Mcore...")
        transformer_layer_spec = get_gpt_layer_local_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None

    args = get_args()

    if "-Raw" in args.dataset:
        if args.train_mode == "pretrain":
            raise ValueError('The LLama-SFT-Raw dataset should only be used for finetuning!')
        # get batches based on the TP rank you are on
        # per_seq_average=False做数据时候已经计算好了seq的平均值
        batch = get_batch_on_this_tp_rank_original(data_iterator, per_seq_average=False)
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs')
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            None
        )
    elif "-Idxmap" in args.dataset:
        # get batches based on the TP rank you are on
        if args.train_mode == "pretrain":
            batch = get_batch_on_this_tp_rank(data_iterator)
        elif args.train_mode == 'sft':
            batch = get_batch_on_this_tp_rank_idxmap_sft(data_iterator, per_seq_average=True)
        
        packed_seq_params = None
        if args.reset_position_ids:
            # sequence-packing, build cu_seqlens
            position_ids = batch.get('position_ids', None)
            if position_ids is not None:
                # mbs = 1
                position_ids = position_ids[0] # shape: [seq_length]
                start_indices = (position_ids == 0).nonzero(as_tuple=True)[0]
                seqlens = start_indices[1:] - start_indices[:-1]
                # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
                cu_seqlens = torch.zeros(start_indices.shape[0] + 1, device=position_ids.device, dtype=torch.int)
                cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
                cu_seqlens[-1] = position_ids.shape[0]
                packed_seq_params = PackedSeqParams(
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    qkv_format='thd'
                )
        
        if packed_seq_params is not None and args.context_parallel_size > 1:
            raise ValueError('Sequence Packing is not supported when CP>1 !')
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs', None)
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            packed_seq_params
        )
    else:
        raise ValueError("please set correct --dataset ")
    

def get_batch_xpo(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None, None

    args = get_args()

    if "-Raw" in args.dataset:
        if args.train_mode == "pretrain":
            raise ValueError('The LLama-SFT-Raw dataset should only be used for finetuning!')
        # get batches based on the TP rank you are on
        # per_seq_average=False做数据时候已经计算好了seq的平均值
        batch = get_batch_on_this_tp_rank_original(data_iterator, per_seq_average=False)
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs')
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            num_seqs,
            None
        )
    elif "-Idxmap" in args.dataset:
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank_idxmap_xpo(data_iterator, per_seq_average=False)
        
        packed_seq_params = None
        if args.reset_position_ids:
            # sequence-packing, build cu_seqlens
            position_ids = batch.get('position_ids', None)
            if position_ids is not None:
                # mbs = 1
                position_ids = position_ids[0] # shape: [seq_length]
                start_indices = (position_ids == 0).nonzero(as_tuple=True)[0]
                seqlens = start_indices[1:] - start_indices[:-1]
                # NOTE: cu_seqlens: [0, A1, A1+A2, A1+A2+A3, ..., seq_len]
                cu_seqlens = torch.zeros(start_indices.shape[0] + 1, device=position_ids.device, dtype=torch.int)
                cu_seqlens[1:-1] = torch.cumsum(seqlens, dim=0)
                cu_seqlens[-1] = position_ids.shape[0]
                packed_seq_params = PackedSeqParams(
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    qkv_format='thd'
                )
        
        if packed_seq_params is not None and args.context_parallel_size > 1:
            raise ValueError('Sequence Packing is not supported when CP>1 !')
        # slice batch along sequence dimension for context parallelism
        num_seqs = batch.pop('num_seqs', None)
        batch = get_batch_on_this_cp_rank(batch)

        return (
            batch['tokens'],
            batch['labels'],
            batch['loss_mask'],
            batch['attention_mask'],
            batch['position_ids'],
            batch['ref_logprobs'],
            num_seqs,
            packed_seq_params
        )
    else:
        raise ValueError("please set correct --dataset ")


def loss_func(loss_mask: torch.Tensor, num_seqs: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    loss = torch.stack([torch.sum(losses.view(-1) * loss_mask), loss_mask.sum()])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    # NOTE: The grad will be scaled down by CP size later, should not remove this multilication factor
    # LINK: https://github.com/NVIDIA/Megatron-LM/issues/906
    # The issue is solved since 0926

    if num_seqs is None:
        return loss[0] * args.context_parallel_size, {"lm loss": averaged_loss}
    
    return loss[0] * args.context_parallel_size, num_seqs.sum(), {"lm loss": averaged_loss}


def loss_func_xpo(
        loss_mask: torch.Tensor, 
        num_seqs: torch.Tensor, 
        ref_logprobs: torch.Tensor, 
        output_tensor: torch.Tensor):

    args = get_args()

    output_dict = {}

    loss_type=args.loss_type
    beta=args.beta
    gamma=args.gamma
    r1=args.r1
    r2=args.r2
    label_smoothing=args.label_smoothing
    add_sft_loss=args.add_sft_loss
    is_average = args.is_average

    if is_average:
        policy_logprobs = (-output_tensor * loss_mask).sum(-1, keepdim=True) / loss_mask.sum(dim=1, keepdim=True)
    else:
        policy_logprobs = (-output_tensor * loss_mask).sum(-1, keepdim=True)
        
    policy_logprobs = policy_logprobs.nan_to_num(1.0)
    policy_chosen_logprobs, policy_reject_logprobs = torch.chunk(policy_logprobs, chunks=2, dim=0)
    policy_logratios = policy_chosen_logprobs - policy_reject_logprobs
    
    ref_chosen_logprobs, ref_reject_logprobs = torch.chunk(ref_logprobs, chunks=2, dim=0)
    ref_logratios = ref_chosen_logprobs - ref_reject_logprobs
    # print(f'Policy-Reference logprobs: {policy_logprobs} {ref_logprobs}', flush=True)
    logits = policy_logratios - ref_logratios

    pl_minus_ref_choose = (policy_chosen_logprobs - ref_chosen_logprobs).mean()
    # print(f'Pl minus ref logprobs: {pl_minus_ref_choose}', flush=True)
    pl_minus_ref_choose_reduced = average_losses_across_data_parallel_group(losses=[pl_minus_ref_choose])[0]
    output_dict["rewards/pl_minus_ref_choose"] = pl_minus_ref_choose_reduced.tolist()
    
    pl_minus_ref_reject = (policy_reject_logprobs - ref_reject_logprobs).mean()
    pl_minus_ref_reject_reduced = average_losses_across_data_parallel_group(losses=[pl_minus_ref_reject])[0]
    output_dict["rewards/pl_minus_ref_reject"] = pl_minus_ref_reject_reduced.tolist()

    policy_chosen_sft_loss = -policy_chosen_logprobs
    ref_chosen_sft_loss = -ref_chosen_logprobs
    policy_choose_loss_reduced = average_losses_across_data_parallel_group(losses=[policy_chosen_sft_loss.mean()])[0]
    output_dict["train/pl_sft_loss_choose"] = policy_choose_loss_reduced.tolist()
    # print(f'Loss type: {loss_type}')
    if loss_type == 'sigmoid':
        xpo_loss = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "hinge":
        xpo_loss = torch.relu(1 - beta * logits)
    elif loss_type == "ipo":
        # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
        ipo_factor = ((2 * beta) ** 2) * 2.0 if add_sft_loss else 1.0
        xpo_loss = ipo_factor * (logits - 1 / (2 * beta)) ** 2
    elif loss_type == 'lpo':
        lpo_factor = (2 * beta) * 2.0
        xpo_loss = lpo_factor * torch.abs(logits - 1 / (2 * beta))
    elif loss_type == 'lpo-r':
        lpo_factor = (2 * beta) * 2.0
        x1 = policy_chosen_logprobs - ref_chosen_logprobs
        x1 = x1 * r1
        x2 = policy_reject_logprobs - ref_reject_logprobs
        x2 = x2 * r2
        xpo_loss = lpo_factor * torch.abs(x1 - x2 - 1 / (2 * beta))
    elif loss_type == 'lpo-ste':
        x1 = policy_chosen_logprobs - ref_chosen_logprobs
        x2 = policy_reject_logprobs - ref_reject_logprobs
    
        x1_ste = r1 * torch.abs(x1 - x2.detach() - 1 / (2 * beta))
        x2_ste = r2 * torch.abs(x1.detach() - x2 - 1 / (2 * beta))
        lpo_factor = (2 * beta) * (2.0 / (r1 + r2))
        xpo_loss = lpo_factor * (x1_ste + x2_ste)
    elif loss_type == 'lpo-ste-dynamic':
        x1 = policy_chosen_logprobs - ref_chosen_logprobs
        x2 = policy_reject_logprobs - ref_reject_logprobs

        r1 = r2 = 1.0
        x1_e = torch.exp(x1.detach())
        x2_e = torch.exp(x2.detach())
        r2_dynamic = (x2_e / x1_e) * r1
        r2_dynamic_reduced = average_losses_across_data_parallel_group(losses=[r2_dynamic.mean()])[0]
        output_dict['r2_dynamic'] = r2_dynamic_reduced.tolist()
    
        x1_ste = r1 * torch.abs(x1 - x2.detach() - 1 / (2 * beta))
        x2_ste = r2_dynamic * torch.abs(x1.detach() - x2 - 1 / (2 * beta))
        lpo_factor = (2 * beta) * (2.0 / (r1 + r2_dynamic))
        xpo_loss = lpo_factor * (x1_ste + x2_ste)
    elif loss_type == "kto_pair":
        # eqn (7) of the HALOs paper
        chosen_KL = (policy_chosen_logprobs - ref_chosen_logprobs).mean().clamp(min=0)
        rejected_KL = (policy_reject_logprobs - ref_reject_logprobs).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logprobs - ref_chosen_logprobs
        rejected_logratios = policy_reject_logprobs - ref_reject_logprobs
        # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
        xpo_loss = torch.cat(
            (
                1 - F.sigmoid(beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
        )
    xpo_wo_sft_loss_reduced = average_losses_across_data_parallel_group(losses=[xpo_loss.mean()])[0]
    output_dict["train/xpo_wo_sft"] = xpo_wo_sft_loss_reduced.tolist()

    if add_sft_loss:
        chosen_policy_minus_ref = policy_chosen_sft_loss - ref_chosen_sft_loss
        smaug_loss = torch.maximum(chosen_policy_minus_ref, 
                                   torch.zeros_like(chosen_policy_minus_ref))
        xpo_loss += gamma * smaug_loss
        xpo_loss_reduced = average_losses_across_data_parallel_group(losses=[xpo_loss.mean()])[0]
        output_dict["train/xpo_with_sft"] = xpo_loss_reduced.tolist()

    loss = xpo_loss.mean()
    # print(f'Policy logprobs: {policy_logprobs} Ref logprobs: {ref_logprobs} Loss mask sum: {loss_mask.sum(dim=1, keepdim=True)}')
    if num_seqs is None:
        return loss * args.context_parallel_size, output_dict
    
    return loss * args.context_parallel_size, num_seqs.sum(), output_dict


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    args = get_args()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    if args.train_mode != 'xpo':
        tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params = get_batch(data_iterator)
        timers("batch-generator").stop()
        # [micro_bsz, seq_length]
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)

        return output_tensor, partial(loss_func, loss_mask, num_seqs)
    else:
        tokens, labels, loss_mask, attention_mask, position_ids, ref_logprobs, num_seqs, packed_seq_params = get_batch_xpo(data_iterator)
        timers("batch-generator").stop()
        # [micro_bsz, seq_length]
        # print(f'R{torch.distributed.get_rank()} After get batch token shape: {tokens.shape}')
        # print(f'R{torch.distributed.get_rank()} After get batch position shape: {position_ids.shape}')
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)

        return output_tensor, partial(loss_func_xpo, loss_mask, num_seqs, ref_logprobs)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path),
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    if "-Raw" in args.dataset:
        train_ds, valid_ds, test_ds = build_pretrain_dataset_from_original(args.dataset)
    else:
        config = core_gpt_dataset_config_from_args(args)

        # NOTE: in preparation scripts, the sequence is collect into (seq, labels)
        # therefore we need to double the seqlen
        if args.train_mode == "sft":
            config.sequence_length = config.sequence_length * 2
        elif args.train_mode == "xpo":
            config.sequence_length = (config.sequence_length + 1 + 3) * 2
        
        if config.mock:
            dataset_type = MockGPTDataset
        else:
            if args.train_mode == "xpo":
                dataset_type = GPTDatasetXPO
            else:
                dataset_type = GPTDataset
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
        ).build()

        print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )
