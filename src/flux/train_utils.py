import argparse
from src.flux.modules.layers import DoubleStreamBlockLoraProcessorDualAttnMod, SingleStreamBlockLoraProcessorDualAttnMod, DoubleStreamBlockPoseLoraProcessor, SingleStreamBlockTripleLoraProcessor, DoubleStreamBlockPoseLoraProcessorOneStream, SingleStreamBlockTripleLoraProcessorOneStream, DoubleStreamBlockPoseLoraProcessorAR, SingleStreamBlockTripleLoraProcessorAR, DoubleStreamBlockPoseLoraProcessorAR_version2, SingleStreamBlockTripleLoraProcessorAR_version2, DoubleStreamBlockPoseLoraProcessorAR_version2_crossattn, SingleStreamBlockTripleLoraProcessorAR_version2_crossattn
import re
import math
import os
import wandb
from pathlib import Path
import torch.distributed as dist


def generate_exp_name_layout_prediction(name, args, *param_names):
    """Generate experiment name based on configuration parameters.
    
    Args:
        args: Configuration arguments
        *param_names: Names of parameters to include in the experiment name
    """
    exp_name = args.exp_name
    save_dir = args.save_dir
    
    # Only create mappings for parameters that were requested
    param_values = {}
    if 'lr' in param_names:
        param_values['lr'] = f"_lr_{args.lr}"
    if 'bs' in param_names:
        param_values['bs'] = f"_bs_{args.local_batch_size * dist.get_world_size() * args.gradient_accumulation_steps}"
    
    for param in param_names:
        if param in param_values:
            exp_name += param_values[param]
            # Only add attn to save_dir if it was explicitly requested

            
    #save_dir += f"_{name}"
    # save_dir += param_values['attn']

    return exp_name, save_dir


def generate_exp_name_layout_prediction_sample(name, args, *param_names):
    """Generate experiment name based on configuration parameters.
    
    Args:
        args: Configuration arguments
        *param_names: Names of parameters to include in the experiment name
    """
    exp_name = args.exp_name
    save_dir = args.save_dir
    
    # Only create mappings for parameters that were requested
    param_values = {}
    if 'lr' in param_names:
        param_values['lr'] = f"_lr_{args.lr}"
    if 'bs' in param_names:
        param_values['bs'] = f"_bs_{args.local_batch_size * args.gradient_accumulation_steps}"
    
    for param in param_names:
        if param in param_values:
            exp_name += param_values[param]
            # Only add attn to save_dir if it was explicitly requested

            
    #save_dir += f"_{name}"
    # save_dir += param_values['attn']

    return exp_name, save_dir


def generate_exp_name(name, args, *param_names):
    """Generate experiment name based on configuration parameters.
    
    Args:
        args: Configuration arguments
        *param_names: Names of parameters to include in the experiment name
    """
    exp_name = args.exp_name
    save_dir = args.save_dir
    
    # Only create mappings for parameters that were requested
    param_values = {}
    if 'pre_real_lora' in param_names:
        param_values['pre_real_lora'] = f'_pre_real_lora' if args.get('pre_real_lora', False) else 'pose_lora_only'
    if 'guidance' in param_names:
        param_values['guidance'] = f'_guidance_{args.guidance}'
    if 'unlimited_canvas' in param_names:
        param_values['unlimited_canvas'] = '_unlimited_canvas2' if args.unlimited_canvas else ''
    if 'pretend_pe' in param_names:
        param_values['pretend_pe'] = '_pretend_pe' if args.pretend_pe else ''
    if 'pe_learnable' in param_names:
        param_values['pe_learnable'] = '_pe_learnable' if args.pe_learnable else '_pe_unlearnable'
    if 'use_learned_shift_scale_ratio' in param_names:
        param_values['use_learned_shift_scale_ratio'] = f'_use_learned_shift_scale_ratio_{args.use_learned_shift_scale_ratio}'
    if 'num_people_indicator' in param_names:
        param_values['num_people_indicator'] = '_w_num_people_indicator' if args.num_people_indicator else ''
    if 'embed_id' in param_names:
        param_values['embed_id'] = f'_embed_id_{args.num_embed_id}' if args.embed_id else ''
    if "embed_id_realnotadded" in param_names:
        param_values["embed_id_realnotadded"] = f'_embed_id_realnotadded_{args.num_embed_id}' if args.embed_id else ''
    if 'embed_id_init' in param_names:
        param_values['embed_id_init'] = f'_{args.embed_id_init}' if args.get('embed_id_init', None) is not None else ''
    if 'embed_id_plan' in param_names:
        param_values['embed_id_plan'] = f'plan_from_{args.continue_step}' if args.get("embed_id_plan", None) is not None else ''
    if 'learn_pe' in param_names:
        param_values['learn_pe'] = '_learn_pe'
    if 'background_color' in param_names:
        param_values['background_color'] = f'_background_color_{args.background_color}' if args.background_color != "black" else ''
    if 'loss_pose_background_lambda' in param_names:
        param_values['loss_pose_background_lambda'] = f'_loss_pose_background_lambda_{args.loss_pose_background_lambda}' if args.loss_pose_background_lambda != 1.0 else ''
    if 'resize_to_square' in param_names:
        param_values['resize_to_square'] = '' if args.resize_to_square else 'use_original_size'
    if 'resolution' in param_names:
        param_values['resolution'] = f'_resolution_{args.resolution}'
    if 'apply_smaller_pose' in param_names:
        param_values['apply_smaller_pose'] = f'_smaller_pose_{args.apply_smaller_pose}' if args.get('apply_smaller_pose', None) is not None and args.apply_smaller_pose != 1 else ''
    if 'customize_img_ids' in param_names:
        param_values['customize_img_ids'] = f'_customize_img_ids' if args.customize_img_ids else ''
    if 'customize_txt_ids' in param_names:
        param_values['customize_txt_ids'] = f'_customize_txt_ids' if args.customize_txt_ids else ''
    if 'generate_img_ids_type' in param_names:
        assert isinstance(args.generate_img_ids_type, int), "generate_img_ids_type must be an integer"
        param_values['generate_img_ids_type'] = f'_customize_img_ids_type_{args.generate_img_ids_type}' if args.generate_img_ids_type != 1 else ''
    if 'double_real_lora' in param_names:
        param_values['double_real_lora'] = f'_double_real_lora' if args.double_real_lora else ''
    if 'single_real_lora' in param_names:
        param_values['single_real_lora'] = f'_single_real_lora' if args.single_real_lora else ''
    if 'real_lr_scale' in param_names:
        param_values['real_lr_scale'] = f'_real_lr_scale_{args.real_lr_scale}' if args.real_lr_scale != 1.0 else ''
    if 'attn_mask_type' in param_names:
        param_values['attn_mask_type'] = f'_{args.attn_mask_type}' if args.get('attn_mask_type', None) is not None else ''
    if 'signal_to_learn' in param_names:
        param_values['signal_to_learn'] = f'_signal_to_learn_{args.signal_to_learn}'
    if 'use_dataset' in param_names:
        param_values['use_dataset'] = f'_usedataset_{args.use_dataset}'
    if 'prompt_type' in param_names:
        param_values['prompt_type'] = f'_prompttype_{args.prompt_type}'
    if 'add_t2p_anno' in param_names:
        param_values['add_t2p_anno'] = f'_add_downward_dog' if args.get('add_t2p_anno', None) else '' # TODO temporal specific
    if 'training_resolution' in param_names:
        param_values['training_resolution'] = f'_training_resolution_{args.training_height}'
    if 'use_general_prompt' in param_names:
        param_values['use_general_prompt'] = '_use_general_prompt' if args.get('use_general_prompt', False) else '_detailedanno_prompt'
    if 'trainable_names' in param_names:
        param_values['trainable_names'] = f'_trainable_names_{args.trainable_names}'
    if 'cfg_ratio' in param_names:
        param_values['cfg_ratio'] = f'_cfg_ratio_{args.cfg_ratio}' if args.get('cfg_ratio', None) else 'cfg_ratio_0.0'
    if 'lambda_real' in param_names:
        param_values['lambda_real'] = f'_lambda_real_{args.lambda_real}'
    if 'lambda_pose' in param_names:
        param_values['lambda_pose'] = f'_lambda_pose_{args.lambda_pose}'
    if 't2p_noise_key' in param_names:
        param_values['t2p_noise_key'] = f'_t2p_{args.t2p_noise_key}noise'
    if 'use_data' in param_names:
        param_values['use_data'] = f'_use_{args.use_data}'
    if 'add_pose_clip' in param_names:
        param_values['add_pose_clip'] = '_add_pose_clip' if args.add_pose_clip else '_use_same_clip'
    if 'text_lora' in param_names:
        param_values['text_lora'] = '_text_lora' if args.text_lora else '_no_text_lora'
    if 'pose_lora' in param_names:
        param_values['pose_lora'] = '_pose_lora' if args.pose_lora else '_no_pose_lora'
    if 'real_lora' in param_names:
        param_values['real_lora'] = '_real_lora' if args.real_lora else '_no_real_lora'
    if 'train_modulation' in param_names:
        param_values['train_modulation'] = '_train_modulation' if args.train_modulation else '_no_train_modulation'
    if 'add_h' in param_names:
        param_values['add_h'] = 'connected_pe' if args.add_h else 'same_pe'
    if 'pose_id' in param_names:
        param_values['pose_id'] = f'pose_id_{args.pose_id_value}'
    if 'attn' in param_names:
        param_values['attn'] = '_attn_masked' if args.use_attn_mask else '_attn_full'
    if 'data' in param_names:
        param_values['data'] = '_w_t2p' if args.t2p else '_hf_only'
    if 'time' in param_names:
        param_values['time'] = f"_time_{args.time_key}"
    if 'ratio' in param_names:
        param_values['ratio'] = f"_ratio_{args.ratio}"
    if 'rank' in param_names:
        param_values['rank'] = f"_rank_{args.rank}"
    if 'lr' in param_names:
        param_values['lr'] = f"_lr_{args.lr}"
    if 'bs' in param_names:
        param_values['bs'] = f"_bs_{args.local_batch_size * dist.get_world_size() * args.gradient_accumulation_steps}" if not args.infer else f"_bs_{args.local_batch_size * args.gradient_accumulation_steps}"
    if 't2p_time' in param_names and hasattr(args, 't2p_time_key'):
        param_values['t2p_time'] = f"_t2p_time_{args.t2p_time_key}"
    if 'train_pose_head' in param_names:
        if args.trainable_names == ['img_in.pose_in', 'final_layer.adaLN_modulation_pose', 'final_layer.linear_pose_img']:
            param_values['train_pose_head'] = '_train_pose_head'
        else:
            assert len(args.trainable_names) == 0
            param_values['train_pose_head'] = '_lock_pose_head'
    # Add parameters to the name
    for param in param_names:
        if param in param_values:
            exp_name += param_values[param]
            # Only add attn to save_dir if it was explicitly requested

            
    #save_dir += f"_{name}"
    # save_dir += param_values['attn']

    return exp_name, save_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config

def replace_attn_processor(dit, args):
    print("using pose first blocks")
    double_processor = DoubleStreamBlockLoraProcessorDualAttnMod
    single_processor = SingleStreamBlockLoraProcessorDualAttnMod

    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]
    
    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)

            lora_attn_procs[name] = double_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora
            )
            
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = single_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora
            )

        else:
            lora_attn_procs[name] = attn_processor

    print("setting LoRA Processor for all attn processors")
    dit.set_attn_processor(lora_attn_procs)

def replace_attn_processor_singlestream(dit, args):
    print("using pose first blocks")
    double_processor = DoubleStreamBlockPoseLoraProcessorOneStream # pose modulation is already added to double_blocks
    single_processor = SingleStreamBlockTripleLoraProcessorOneStream

    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]
    
    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)

            lora_attn_procs[name] = double_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora
            )
            
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = single_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora
            )

        else:
            lora_attn_procs[name] = attn_processor

    print("setting LoRA Processor for all attn processors")
    dit.set_attn_processor(lora_attn_procs)


def replace_attn_processor_triplelora(dit, args):
    print("using pose first blocks")
    double_processor = DoubleStreamBlockPoseLoraProcessor # pose modulation is already added to double_blocks
    single_processor = SingleStreamBlockTripleLoraProcessor 

    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]
    
    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)

            lora_attn_procs[name] = double_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora, pose_lora=args.pose_lora, real_lora=args.real_lora
            )
            
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = single_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora, pose_lora=args.pose_lora, real_lora=args.real_lora
            )

        else:
            lora_attn_procs[name] = attn_processor

    print("setting LoRA Processor for all attn processors")
    dit.set_attn_processor(lora_attn_procs)

def replace_attn_processor_triplelora_ar(dit, args):
    print("using pose first blocks")
    double_processor = DoubleStreamBlockPoseLoraProcessorAR # pose modulation is already added to double_blocks
    single_processor = SingleStreamBlockTripleLoraProcessorAR 

    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]
    
    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)

            lora_attn_procs[name] = double_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora, pose_lora=args.pose_lora, real_lora=args.get('double_real_lora', False)
            )
            if args.get('double_real_lora', False):
                print("Adding double_real_lora")
            
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = single_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora, pose_lora=args.pose_lora, real_lora=args.get('single_real_lora', False)
            )
            if args.get('single_real_lora', False):
                print("Adding single_real_lora")

        else:
            lora_attn_procs[name] = attn_processor

    print("setting LoRA Processor for all attn processors")
    dit.set_attn_processor(lora_attn_procs)

def replace_attn_processor_triplelora_ar_version2(dit, args):
    print("using pose first blocks")
    double_processor = DoubleStreamBlockPoseLoraProcessorAR_version2  # pose modulation is already added to double_blocks
    single_processor = SingleStreamBlockTripleLoraProcessorAR_version2 

    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]
    
    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)

            lora_attn_procs[name] = double_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora, pose_lora=args.pose_lora, real_lora=args.get('double_real_lora', False), 
                pre_real_lora=args.get('pre_real_lora', False)
            )
            if args.get('double_real_lora', False):
                print("Adding double_real_lora")
            if args.get('pre_real_lora', False):
                print("Adding pre_real_lora")
            
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = single_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora, pose_lora=args.pose_lora, real_lora=args.get('single_real_lora', False), 
                pre_real_lora=args.get('pre_real_lora', False)
            )
            if args.get('single_real_lora', False):
                print("Adding single_real_lora")
            if args.get('pre_real_lora', False):
                print("Adding pre_real_lora")

        else:
            lora_attn_procs[name] = attn_processor

    print("setting LoRA Processor for all attn processors")
    dit.set_attn_processor(lora_attn_procs)


def replace_attn_processor_triplelora_ar_version2_crossattn(dit, args):
    print("using pose first blocks")
    double_processor = DoubleStreamBlockPoseLoraProcessorAR_version2_crossattn  # pose modulation is already added to double_blocks
    single_processor = SingleStreamBlockTripleLoraProcessorAR_version2_crossattn 

    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]
    
    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)

            lora_attn_procs[name] = double_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora, pose_lora=args.pose_lora, real_lora=args.get('double_real_lora', False)
            )
            if args.get('double_real_lora', False):
                print("Adding double_real_lora")
            
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = single_processor(
                dim=3072, rank=args.rank, text_lora=args.text_lora, pose_lora=args.pose_lora, real_lora=args.get('single_real_lora', False)
            )
            if args.get('single_real_lora', False):
                print("Adding single_real_lora")

        else:
            lora_attn_procs[name] = attn_processor

    print("setting LoRA Processor for all attn processors")
    dit.set_attn_processor(lora_attn_procs)

def get_lr(it, args):
    min_lr = args.lr * args.lr_decay_frac
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.lr * (it+1) / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.max_train_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.max_train_steps - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (args.lr - min_lr)



def disable_grad(dit, traiable_names):
    dit.train()
    for n, param in dit.named_parameters():
        if not any(trainable_name in n for trainable_name in traiable_names):
            param.requires_grad = False
        else:
            print("trainable -- ", n)


def setup_wandb(args, rank):
    wandb_dit = Path.cwd() / args.save_dir / args.exp_name / "wandb"
    os.makedirs(wandb_dit, exist_ok=True)
    wandb_run = None
    wandb_key = getattr(args, "wandb_key", "")
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
    resume_wandb = getattr(args, "resume_wandb", False)

    if args.use_wandb:
        os.environ["WANDB_MODE"] = "offline"
        if rank == 0:
            wandb_id = None

            if resume_wandb:
                wandb_id_file = wandb_dit / "wandb_run_id.txt"
                print("!!!!!!!!!!!!!!!!!")
                print(f"[W&B] Resuming from run ID: {wandb_id_file}")
                print("!!!!!!!!!!!!!!!!!")
                if wandb_id_file.exists():
                    with open(wandb_id_file, "r") as f:
                        wandb_id = f.read().strip()
                        print(f"[W&B] Resuming from run ID: {wandb_id}")
                else:
                    print("[W&B] Resume requested but wandb_run_id.txt not found. Starting new run.")

            wandb_run = wandb.init(
                config=dict(args),
                dir=str(wandb_dit),
                id=wandb_id,
                resume="must" if wandb_id else None,
                mode="online",
                entity=args.wandb_entry,
                project=args.wandb_project,
            )
            wandb.run.log_code(".")
            wandb.run.name = args.exp_name
            print(f"[W&B] Run dir: {wandb_run.dir}")
            os.makedirs(wandb_run.dir, exist_ok=True)

            if not resume_wandb or wandb_id is None:
                with open(wandb_dit / "wandb_run_id.txt", "w") as f:
                    f.write(wandb_run.id)

    return wandb_run
