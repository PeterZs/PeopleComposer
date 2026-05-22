# DDP template codes
# # train_ddp.py
import os, math, time, json, random, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from datetime import timedelta
import re
import sys
from torch.utils.data import DataLoader, DistributedSampler

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    seed_all(42)

    return rank, world_size, local_rank




init_logger()
launch_distributed_job()
'''
you must call this before constructing either DDP or FSDP.
Both wrappers need an initialized ProcessGroup (from dist.init_process_group) to know world size, ranks, and to perform parameter sync.

Here’s the clean lifecycle you want:

1. Launch with torchrun (it sets RANK, LOCAL_RANK, etc.).
2. Initialize distributed (your launch_distributed_job):
    a. reads env vars
    b. calls dist.init_process_group(...)
    c. sets CUDA device for this rank (torch.cuda.set_device(local_rank))
3. Build the model on CPU (safe on all ranks).
4. (Optional) Rank-0 loads checkpoint on CPU, others skip; then dist.barrier().
5. Move model to this rank’s GPU (model.to(f"cuda:{local_rank}")).
6. Wrap with DDP or FSDP.
7. Train.
'''



def load_model_checkpoint(model, rank: int, args, verbose=True): # load/resume model weights on the model on cpu
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

     # --- Rank-0 loads on CPU; others set None ---
    obj_list = [None]
    if rank == 0:
        if verbose:
            print(f"🔄 loading optimizer checkpoint on rank-0 from {opt_ckpt_path}")
        full_osd = torch.load(opt_ckpt_path, map_location="cpu")
        obj_list[0] = full_osd

    # --- Broadcast to all ranks ---
    # Make sure torch.cuda.set_device(local_rank) was called earlier in the process.
    dist.broadcast_object_list(obj_list, src=0)

    # --- Everyone loads the same full state ---
    full_osd = obj_list[0]
    optimizer.load_state_dict(full_osd)
    if verbose:
        print(f"✅ optimizer state restored on rank {rank}")



def _unwrap_ddp(model):
    return model.module if isinstance(model, DDP) else model

def _cpu_state_dict(module: torch.nn.Module):
    # Move all tensors in the state_dict to CPU to avoid GPU memory spikes while saving
    sd = module.state_dict()
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}

def save_model_checkpoint_ddp(model, rank: int, args, global_step: int = 0, verbose=True):
    """
    Save a FULL (non-sharded) model checkpoint from rank-0.
    Works with DDP-wrapped or plain models.
    Call after the current training step has completed.
    """
    model_to_save = _unwrap_ddp(model)
    cpu_state = _cpu_state_dict(model_to_save)

    if rank == 0:
        save_dir = Path.cwd() / args.save_dir / args.exp_name
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{args.model_save_name}-{global_step:06d}.pt"
        path = save_dir / fname
        torch.save(cpu_state, path)
        if verbose:
            print(f"💾 saved model checkpoint → {path}")
    # Optional: keep everyone in lock-step
    dist.barrier()

def _optimizer_state_to_cpu(optim_state: dict):
    """
    Move all tensor leaves of an optimizer.state_dict() to CPU.
    (state_dict structure: {'state': {param_id: {...}}, 'param_groups': [...]})
    """
    state_cpu = {}
    for k, v in optim_state.items():
        if k == "state":
            inner = {}
            for pid, slot in v.items():
                inner[pid] = {sk: (sv.detach().cpu() if torch.is_tensor(sv) else sv)
                              for sk, sv in slot.items()}
            state_cpu["state"] = inner
        elif k == "param_groups":
            # param_groups is metadata; keep as-is
            state_cpu["param_groups"] = v
        else:
            state_cpu[k] = v
    return state_cpu

def save_optimizer_checkpoint_ddp(optimizer, rank: int, args,
                                  global_step: int = 0, verbose=True):
    """
    Save the FULL optimizer state (non-sharded) from rank-0.
    Create the optimizer from the (DDP-wrapped) model before calling this,
    so param_groups ordering matches on restore.
    """
    # In DDP there is no sharded optimizer state; just take optimizer.state_dict()
    optim_state = optimizer.state_dict()
    optim_state = _optimizer_state_to_cpu(optim_state)

    if rank == 0:
        save_dir = Path.cwd() / args.save_dir / args.exp_name
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{args.optimizer_name}-{args.model_save_name}-{global_step:06d}.pt"
        path = save_dir / fname
        torch.save(optim_state, path)
        if verbose:
            print(f"💾 saved optimizer checkpoint → {path}")
    dist.barrier()

def setup_model_DDP(model, args):
    global_step = 0
    num_total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"=> Number of total parameters: {num_total_params / 1e6}M")
    name_to_orig_shape = {} # for determin weight decay. No weight decay for 1D params.
    name_to_orig_numel = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_to_orig_shape[name] = param.shape
        name_to_orig_numel[name] = param.numel()
    
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    '''
    Typical flow (single-node or multi-node; same idea)

    All ranks build the model on CPU.
    (If you used a meta device, materialize to CPU before loading.)

    Rank 0 loads the checkpoint from disk to CPU (your load_model_checkpoint(..., map_location="cpu")).
    Other ranks do nothing here.

    Barrier: dist.barrier() so everyone waits for rank 0 to finish.

    Each rank moves its own model to its own GPU:
    torch.cuda.set_device(local_rank) → model.to(f"cuda:{local_rank}").

    Wrap in DDP:
    DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True).
    DDP then broadcasts parameters from rank 0 so every rank is bit-identical. (Buffers sync too if broadcast_buffers=True.)
    '''
    global_step = load_model_checkpoint(model, rank, args) # if rank == 0 written inside load_model_checkpoint
    dist.barrier()
    model.to(f"cuda:{local_rank}")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True)
    # TODO optionally checkpointing -- not used first
    ##### Setup the optimizer and lr-scheduler ########
    name_to_trainable_params = {}
    params_with_weight_decay = []
    params_no_weight_decay = []
    wd_params_name_list = [_name for _name, _shape in name_to_orig_shape.items() if len(_shape) > 1]
    def rename_param(name: str) -> str:
        return name.replace("module.", "")

    num_trainable_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        renamed_name = rename_param(name)
        name_to_trainable_params[renamed_name] = param
        if renamed_name in wd_params_name_list:
            params_with_weight_decay.append(param)
        else:
            params_no_weight_decay.append(param)
        logging.info(
            f"DDP param name: {name}, Original param name: {renamed_name}. | "
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
    ]
    # drop empty groups
    optimizer_param_groups = [g for g in optimizer_param_groups if g["params"]]
    
    '''
    Where does the optimizer live?
    An optimizer object (torch.optim.AdamW, SGD, etc.) itself is just a Python class living on the CPU side.
    What matters is the states it manages (momentum buffers, exp moving averages, etc.). Those states are created on the same device as the parameters it optimizes.
    '''

    optimizer = torch.optim.AdamW(optimizer_param_groups, betas=(args.beta1, args.beta2), lr=args.lr) # this lr is only for groups that have no lr yet
    load_optimizer_checkpoint(model, optimizer, rank, args)
    dist.barrier()

    return model, optimizer, global_step







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

