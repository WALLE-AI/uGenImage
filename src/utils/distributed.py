"""DDP 辅助。

原实现的 scripts/train_transformer.sh 调用 torch.distributed.launch，
但训练脚本没有任何 rank 逻辑 —— 4 个进程各自独立训练并抢写同一个 checkpoint。
这里补上真正的 DDP，单卡运行时全部退化为 no-op。
"""
import os

import torch
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def init_distributed():
    """按 torchrun 注入的环境变量初始化。返回 (rank, world_size, local_rank)。"""
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return 0, 1, 0
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 1 and not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup():
    if is_distributed():
        dist.destroy_process_group()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main():
    return get_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


def all_reduce_mean(value, device):
    """把标量在所有 rank 上求平均，用于日志与验证指标。"""
    if not is_distributed():
        return float(value)
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / get_world_size()).item()


def unwrap(model):
    return model.module if hasattr(model, 'module') else model
