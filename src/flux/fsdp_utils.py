import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    LocalStateDictConfig,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools         
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from pathlib import Path
import os
import torch.distributed as dist
from datetime import timedelta
import logging
import re

from src.flux.modules.layers import DoubleStreamBlock, SingleStreamBlock
from torch.utils.data import DataLoader, DistributedSampler

# TODO-TODO: global seeding?

def get_policy_base(blocks):
    recursive_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=blocks, # layer_cls = {DoubleStreamBlock, SingleStreamBlock}
    )
    return recursive_policy


def fsdp_checkpointing_base(model, blocks):
    non_reentrant = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda m: isinstance(m, blocks)
    apply_activation_checkpointing(model, non_reentrant, check_fn)


bfloatPolicy = MixedPrecision(
    # param_dtype=torch.bfloat16,
    param_dtype=torch.bfloat16,
    # gradient communication precision
    reduce_dtype=torch.bfloat16,
    # buffer precision
    buffer_dtype=torch.float32, # use float32 instead
)

comboPolicy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

# --------------------------------------------------------------------------
# 0.  ONE global save-policy object – create once, reuse everywhere
# --------------------------------------------------------------------------
FULL_STATE_SAVE_POLICY = FullStateDictConfig(
    offload_to_cpu=True,   # stream shards to rank-0 CPU RAM
    rank0_only=True        # only rank-0 writes the file
)

# --------------------------------------------------------------------------
# 1.  ------------  LOADING  ------------
# --------------------------------------------------------------------------
# def load_model_checkpoint(model, rank: int, args, verbose=True):
#     """
#     Load model checkpoint based on resume flag and filename pattern.
#     Returns the global step for resuming training.
#     """
#     startfrom_0step = getattr(args, "startfrom_0step", False)
#     resume_wandb = getattr(args, "resume_wandb", False)
#     global_step = 0
#     ckpt_path = None

#     if rank == 0:
#         exp_dir = Path.cwd() / args.save_dir / args.exp_name

#         if resume_wandb:
#             # Try to find latest checkpoint file
#             pattern = re.compile(rf"{re.escape(args.model_save_name)}-(\d+)\.pt$")
#             all_ckpts = list(exp_dir.glob(f"{args.model_save_name}-*.pt"))
#             valid_ckpts = [(f, int(m.group(1))) for f in all_ckpts if (m := pattern.match(f.name))]
#             if valid_ckpts:
#                 # Get file with max numeric suffix
#                 ckpt_path, global_step = max(valid_ckpts, key=lambda x: x[1])
#                 if verbose:
#                     print(f"🔁 [resume_wandb=True] Found latest checkpoint: {ckpt_path}")
#             else:
#                 # Fallback to provided checkpoint filename
#                 fallback_ckpt = exp_dir / args.checkpoint.model_filename
#                 if fallback_ckpt.is_file():
#                     ckpt_path = fallback_ckpt
#                     if verbose:
#                         print(f"📂 [resume_wandb=True] Using fallback checkpoint: {ckpt_path}")
#                     if not startfrom_0step:
#                         try:
#                             global_step = int(ckpt_path.stem.split("-")[-1])
#                         except Exception:
#                             global_step = 0
#                 else:
#                     if verbose:
#                         print("⚠ No valid checkpoints found — starting from scratch.")

#         else:
#             # resume_wandb=False → try only default filename
#             ckpt_path = exp_dir / args.checkpoint.model_filename
#             if ckpt_path.is_file():
#                 if verbose:
#                     print(f"📂 [resume_wandb=False] Loading from {ckpt_path}")
#                 if not startfrom_0step:
#                     try:
#                         global_step = int(ckpt_path.stem.split("-")[-1])
#                     except Exception:
#                         global_step = 0
#             else:
#                 if verbose:
#                     print(f"⚠ model checkpoint {ckpt_path} not found — starting from scratch.")
#                 ckpt_path = None

#         if ckpt_path and ckpt_path.is_file():
#             state = torch.load(ckpt_path, map_location="cpu")
#             model.load_state_dict(state)
#             if verbose:
#                 print(f"✅ model weights loaded on rank-0, resuming from global step {global_step}")

#     # Sync global_step across ranks
#     global_step_tensor = torch.tensor(global_step, device="cuda")
#     dist.broadcast(global_step_tensor, src=0)
#     global_step = global_step_tensor.item()

#     return global_step
# --------------------------------------------------------------------------
# 1.  ------------  LOADING  ------------
# --------------------------------------------------------------------------
def load_model_checkpoint(model, rank: int, args, verbose=True):
    """
    Load model checkpoint based on resume flag and filename pattern.
    Returns the global step for resuming training.
    """
    startfrom_0step = getattr(args, "startfrom_0step", False)
    resume_wandb = getattr(args, "resume_wandb", False)
    global_step = 0
    ckpt_path = None

    if rank == 0:
        exp_dir = Path.cwd() / args.save_dir / args.exp_name

        # First priority: Check if specific model_filename exists and is filled
        specified_ckpt = exp_dir / args.checkpoint.model_filename
        if args.checkpoint.model_filename and specified_ckpt.is_file():
            ckpt_path = specified_ckpt
            if verbose:
                print(f"📂 Using specified model checkpoint: {ckpt_path}")
            if not startfrom_0step:
                try:
                    global_step = int(ckpt_path.stem.split("-")[-1])
                except Exception:
                    global_step = 0
        
        # Second priority: If no specific file found and resume_wandb is True, look for latest
        elif resume_wandb:
            # Try to find latest checkpoint file
            pattern = re.compile(rf"{re.escape(args.model_save_name)}-(\d+)\.pt$")
            all_ckpts = list(exp_dir.glob(f"{args.model_save_name}-*.pt"))
            valid_ckpts = [(f, int(m.group(1))) for f in all_ckpts if (m := pattern.match(f.name))]
            if valid_ckpts:
                # Get file with max numeric suffix
                ckpt_path, global_step = max(valid_ckpts, key=lambda x: x[1])
                if verbose:
                    print(f"🔁 [resume_wandb=True] Found latest checkpoint: {ckpt_path}")
            else:
                if verbose:
                    print("⚠ No valid checkpoints found — starting from scratch.")
        
        # Third priority: If resume_wandb is False and no specific file found
        else:
            if verbose:
                print(f"⚠ model checkpoint {specified_ckpt} not found — starting from scratch.")
            ckpt_path = None

        if ckpt_path and ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state)
            if verbose:
                print(f"✅ model weights loaded on rank-0, resuming from global step {global_step}")

    # Sync global_step across ranks
    global_step_tensor = torch.tensor(global_step, device="cuda")
    dist.broadcast(global_step_tensor, src=0)
    global_step = global_step_tensor.item()

    return global_step


def load_optimizer_checkpoint(model, optimizer, rank: int, args, verbose=True):
    """
    Load a FULL-STATE optimizer checkpoint after FSDP wrapping.
    Uses `scatter_full_optim_state_dict` so only rank-0 touches disk.
    """
    resume_wandb = getattr(args, "resume_wandb", False)
    opt_ckpt_path = None
    exp_dir = Path.cwd() / args.save_dir / args.exp_name

    if resume_wandb:
        # Try to find latest matching optimizer checkpoint
        pattern = re.compile(rf"{re.escape(args.optimizer_name)}-(\d+)\.pt$")
        all_ckpts = list(exp_dir.glob(f"{args.optimizer_name}-*.pt"))
        valid_ckpts = [(f, int(m.group(1))) for f in all_ckpts if (m := pattern.match(f.name))]
        if valid_ckpts:
            # Get file with highest numeric suffix
            opt_ckpt_path, _ = max(valid_ckpts, key=lambda x: x[1])
            if verbose and rank == 0:
                print(f"🔁 [resume_wandb=True] Found latest optimizer checkpoint: {opt_ckpt_path}")
        else:
            fallback = exp_dir / args.checkpoint.optimizer_filename
            if fallback.is_file():
                opt_ckpt_path = fallback
                if verbose and rank == 0:
                    print(f"📂 [resume_wandb=True] Using fallback optimizer checkpoint: {opt_ckpt_path}")
            else:
                if verbose and rank == 0:
                    print("⚠ No optimizer checkpoint found — skipping restore.")
                return
    else:
        # resume_wandb=False → load default filename
        opt_ckpt_path = exp_dir / args.checkpoint.optimizer_filename
        if not opt_ckpt_path.is_file():
            if verbose and rank == 0:
                print(f"⚠ optimizer checkpoint {opt_ckpt_path} not present — skipping")
            return
        if verbose and rank == 0:
            print(f"📂 [resume_wandb=False] Loading optimizer from {opt_ckpt_path}")

    full_osd = None
    if rank == 0:
        if verbose:
            print(f"🔄 loading optimizer checkpoint on rank-0 from {opt_ckpt_path}")
        full_osd = torch.load(opt_ckpt_path, map_location="cpu")

    # All ranks participate
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
    optimizer.load_state_dict(sharded_osd)
    if verbose:
        print(f"✅ optimizer shard restored on rank {rank}")


# --------------------------------------------------------------------------
# 2.  ------------  SAVING  ------------
# --------------------------------------------------------------------------
def save_model_checkpoint(model, rank: int, args, global_step: int = 0, verbose=True):
    """
    Gather FULL-STATE model weights to rank-0 CPU and save one *.pt* file.
    Call *after* the current training step has completed.
    """
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FULL_STATE_SAVE_POLICY,
    ):
        cpu_state = model.state_dict()     # gathers & offloads automatically

    if rank == 0:
        save_dir = Path.cwd() / args.save_dir / args.exp_name
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{args.model_save_name}-{global_step:06d}.pt"
        torch.save(cpu_state, save_dir / fname)
        if verbose:
            print(f"💾 saved model checkpoint → {save_dir / fname}")


def save_optimizer_checkpoint(model, optimizer, rank: int, args,
                              global_step: int = 0, verbose=True):
    """
    Gather FULL-STATE optimizer state to rank-0 CPU and save.
    Large optimizers (e.g. AdamW on 10 B params) can take minutes—print status!
    """
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        if verbose:
            print(f"📥 gathering optimizer state on rank {rank}")
        optim_state = FSDP.full_optim_state_dict(model, optimizer)

    if rank == 0:
        save_dir = Path.cwd() / args.save_dir / args.exp_name
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{args.optimizer_name}-{args.model_save_name}-{global_step:06d}.pt"
        torch.save(optim_state, save_dir / fname)
        if verbose:
            print(f"💾 saved optimizer checkpoint → {save_dir / fname}")



def launch_distributed_job(backend: str = "nccl"):
    """
    When launched with torchrun --nproc_per_node=<> --nnodes=<>  --rdzv-endpoint=${master_addr}:${master_port}
    The following environment variables are set:
    LOCAL_RANK: 0 ~ nproc_per_node - 1
    RANK: 0 ~ world_size - 1
    WORLD_SIZE: nproc_per_node * nnodes
    MASTER_ADDR: master_addr
    MASTER_PORT: master_port
    GROUP_RANK: 0 ~ nnodes - 1
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    ddp_local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    ddp_node_rank = int(os.environ["GROUP_RANK"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)

    print(
        f"Process {rank}/{world_size} is using device or local rank {local_rank}/{ddp_local_world_size} on node {ddp_node_rank}"
    )

    dist.barrier()

    return rank, world_size, local_rank


def init_logger():
    rank = int(os.getenv("RANK", 0))       # torchrun injects this
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
        datefmt="%Y-%m-%d %H:%M:%S",  # Define the date format      
        handlers=[
            logging.StreamHandler(),                           # console
        ],
    )
    # Inject the rank into the log record:
    logging.LoggerAdapter(logging.getLogger(), {"rank": rank})

init_logger()
launch_distributed_job()



def setup_model_train_real_only(model, args):
    """
    when to call this function?
    - build model, load pretrained weights/checkpoints, replace lora and in-last layers, set requires_grad
    - then call this function to wrap the model with FSDP.
    config: config object should have the following keys:
    - lr: learning rate.
    - weight_decay: weight decay.
    - beta1: beta1 for the AdamW optimizer.
    - beta2: beta2 for the AdamW optimizer.
    Setup the model for training.  Model is already on CPU or meta device. 
    Step-1: extract a dict of the model params, keep track of shape and requires_grad. 
    Step-2: Warp the model with FSDP.
    Step-3: Warp the model with activation checkpointing.
    Step-4: Compile the model [optional]
    Step-5: Setup the optimizer.
    Step-6: Setup the ema.
    Step-7: TODO: support for checkpointing manager and resume! 
    """
    global_step = 0 # TODO resume training step also!
    num_total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"=> Number of total parameters: {num_total_params / 1e6}M")
    name_to_orig_shape = {} # for determin weight decay. No weight decay for 1D params.
    name_to_orig_numel = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_to_orig_shape[name] = param.shape
        name_to_orig_numel[name] = param.numel()

    ######## Step-1: Load checkpoint if exists ########
    rank = dist.get_rank()
    global_step = load_model_checkpoint(model, rank, args)
    dist.barrier()
    # TODO-TODO where is model.to(f"cuda:{rank}")?

    ### Step-2: Compile the model ####
    # TODO notice that compile need to be done before the model is wrapped with FSDP.!
    # torch.compile(model)

    ######## Step-3: Sharding the model with FSDP ########
    fsdp_modules = (DoubleStreamBlock, SingleStreamBlock)
    fsdp_wrap_policy = get_policy_base(fsdp_modules)
    model = FSDP(
        model,
        auto_wrap_policy = fsdp_wrap_policy,
        mixed_precision  = bfloatPolicy,
        device_id=torch.cuda.current_device(),
        use_orig_params  = True,     # PyTorch ≥ 2.2 → reuse param shards between fwd/bwd reuses the fp16/bf16 parameter shard in-place during the optimizer step; again, no separate master weights.
        forward_prefetch = True,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        limit_all_gathers=True,
        sync_module_states=True, #You load the weights only on rank 0 **before** wrapping, so the other ranks must receive them here.
    )
    ### Step-4: Warp the model with activation checkpointing ####
    fsdp_checkpointing_base(model, fsdp_modules)


    ######## Step-5: Setup the optimizer and lr-scheduler ########
    name_to_trainable_params = {}
    params_with_weight_decay = []
    params_no_weight_decay = []

    params_real_wd = []
    params_real_nwd = []

    wd_params_name_list = [_name for _name, _shape in name_to_orig_shape.items() if len(_shape) > 1]
    def rename_param(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "") \
                   .replace("_checkpoint_wrapped_module.", "") \
                   .replace("_orig_mod.", "") \
                   .replace("module.", "")

    num_trainable_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        renamed_name = rename_param(name)

        ####### use different lr for real stream #######
        is_real_lora = "qkv_lora_real" in renamed_name  # or any substring you prefer

        name_to_trainable_params[renamed_name] = param
        if renamed_name in wd_params_name_list:
            if is_real_lora:
                params_real_wd.append(param)
                print(f"Adding real weight decay param: {renamed_name}")
            else:
                # freeze the params
                param.requires_grad = False
        else:
            if is_real_lora:
                params_real_nwd.append(param)
                print(f"Adding real no weight decay param: {renamed_name}")
            else:
                param.requires_grad = False
        logging.info(
            f"FSDP param name: {name}, Original param name: {renamed_name}. | "
            f"Param shape: {param.shape}, numel: {param.numel()}, dtype: {param.dtype} | "
            f"weight decay? : {renamed_name in wd_params_name_list}"
        )
        num_trainable_params += name_to_orig_numel[renamed_name]
    
    logging.info(f"=> Number of trainable parameters: {num_trainable_params / 1e6}M")
    
    weight_decay = args.get("weight_decay", 0.0)


    optimizer_param_groups = [
        {
            "params": params_real_wd,
            "weight_decay": weight_decay,
            "lr": args.lr * args.get('real_lr_scale', 1.0),
        },
        {
            "params": params_real_nwd,
            "weight_decay": 0.0,
            "lr": args.lr * args.get('real_lr_scale', 1.0),
        },
    ]
    # drop empty groups
    optimizer_param_groups = [g for g in optimizer_param_groups if g["params"]]
    

    optimizer = torch.optim.AdamW(optimizer_param_groups, betas=(args.beta1, args.beta2)) # this lr is only for groups that have no lr yet
    load_optimizer_checkpoint(model, optimizer, rank, args)
    dist.barrier()

    return model, optimizer, global_step

# how to use
def setup_model(model, args):
    """
    when to call this function?
    - build model, load pretrained weights/checkpoints, replace lora and in-last layers, set requires_grad
    - then call this function to wrap the model with FSDP.
    config: config object should have the following keys:
    - lr: learning rate.
    - weight_decay: weight decay.
    - beta1: beta1 for the AdamW optimizer.
    - beta2: beta2 for the AdamW optimizer.
    Setup the model for training.  Model is already on CPU or meta device. 
    Step-1: extract a dict of the model params, keep track of shape and requires_grad. 
    Step-2: Warp the model with FSDP.
    Step-3: Warp the model with activation checkpointing.
    Step-4: Compile the model [optional]
    Step-5: Setup the optimizer.
    Step-6: Setup the ema.
    Step-7: TODO: support for checkpointing manager and resume! 
    """
    global_step = 0 # TODO resume training step also!
    num_total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"=> Number of total parameters: {num_total_params / 1e6}M")
    name_to_orig_shape = {} # for determin weight decay. No weight decay for 1D params.
    name_to_orig_numel = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_to_orig_shape[name] = param.shape
        name_to_orig_numel[name] = param.numel()

    ######## Step-1: Load checkpoint if exists ########
    rank = dist.get_rank()
    global_step = load_model_checkpoint(model, rank, args)
    dist.barrier()

    ### Step-2: Compile the model ####
    # TODO notice that compile need to be done before the model is wrapped with FSDP.!
    # torch.compile(model)

    ######## Step-3: Sharding the model with FSDP ########
    fsdp_modules = (DoubleStreamBlock, SingleStreamBlock)
    fsdp_wrap_policy = get_policy_base(fsdp_modules)
    model = FSDP(
        model,
        auto_wrap_policy = fsdp_wrap_policy,
        mixed_precision  = bfloatPolicy,
        device_id=torch.cuda.current_device(),
        use_orig_params  = True,     # PyTorch ≥ 2.2 → reuse param shards between fwd/bwd reuses the fp16/bf16 parameter shard in-place during the optimizer step; again, no separate master weights.
        forward_prefetch = True,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        limit_all_gathers=True,
        sync_module_states=True, #You load the weights only on rank 0 **before** wrapping, so the other ranks must receive them here.
    )
    ### Step-4: Warp the model with activation checkpointing ####
    fsdp_checkpointing_base(model, fsdp_modules)


    ######## Step-5: Setup the optimizer and lr-scheduler ########
    name_to_trainable_params = {}
    params_with_weight_decay = []
    params_no_weight_decay = []

    params_real_wd = []
    params_real_nwd = []

    wd_params_name_list = [_name for _name, _shape in name_to_orig_shape.items() if len(_shape) > 1]
    def rename_param(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "") \
                   .replace("_checkpoint_wrapped_module.", "") \
                   .replace("_orig_mod.", "") \
                   .replace("module.", "")

    num_trainable_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        renamed_name = rename_param(name)

        ####### use different lr for real stream #######
        is_real_lora = "qkv_lora_real" in renamed_name  # or any substring you prefer

        name_to_trainable_params[renamed_name] = param
        if renamed_name in wd_params_name_list:
            if is_real_lora:
                params_real_wd.append(param)
                print(f"Adding real weight decay param: {renamed_name}")
            else:
                params_with_weight_decay.append(param)
        else:
            if is_real_lora:
                params_real_nwd.append(param)
                print(f"Adding real no weight decay param: {renamed_name}")
            else:
                params_no_weight_decay.append(param)
        logging.info(
            f"FSDP param name: {name}, Original param name: {renamed_name}. | "
            f"Param shape: {param.shape}, numel: {param.numel()}, dtype: {param.dtype} | "
            f"weight decay? : {renamed_name in wd_params_name_list}"
        )
        num_trainable_params += name_to_orig_numel[renamed_name]
    
    logging.info(f"=> Number of trainable parameters: {num_trainable_params / 1e6}M")
    
    weight_decay = args.get("weight_decay", 0.0)

    optimizer_param_groups = [
        {
            "params": params_with_weight_decay,
            "weight_decay": weight_decay,
            "lr": args.lr,
        },
        {
            "params": params_no_weight_decay,
            "weight_decay": 0.0,
            "lr": args.lr,
        },
        {
            "params": params_real_wd,
            "weight_decay": weight_decay,
            "lr": args.lr * args.get('real_lr_scale', 1.0),
        },
        {
            "params": params_real_nwd,
            "weight_decay": 0.0,
            "lr": args.lr * args.get('real_lr_scale', 1.0),
        },
    ]
    # drop empty groups
    optimizer_param_groups = [g for g in optimizer_param_groups if g["params"]]
    

    optimizer = torch.optim.AdamW(optimizer_param_groups, betas=(args.beta1, args.beta2), lr=args.lr) # this lr is only for groups that have no lr yet
    load_optimizer_checkpoint(model, optimizer, rank, args)
    dist.barrier()

    return model, optimizer, global_step
    # TODO resume training step also!

def build_dataloader(dataset, args, collate_fn=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=args.is_train,
        seed=args.seed,
        drop_last=args.is_train,       # keep same micro-batch size
    )
    return DataLoader(
        dataset,
        batch_size=args.local_batch_size,    # per-GPU micro-batch
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=lambda w_id: torch.manual_seed(args.seed + w_id),
        collate_fn=collate_fn,     # or omit if default is fine
    )

