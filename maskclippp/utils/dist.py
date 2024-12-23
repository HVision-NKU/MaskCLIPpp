from typing import List, Tuple, Dict, Union, Optional

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed.nn.functional import _AllGather

from detectron2.utils import comm


# class AllGather(Function):
#     @staticmethod
#     def forward(ctx, group, tensor, out_tensor_list):
#         tensor = tensor.contiguous()
#         ctx.group = group
#         dist.all_gather(out_tensor_list, tensor, group=group)
#         return tuple(out_tensor_list)

#     @staticmethod
#     def backward(ctx, *grad_outputs):
#         if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
#             rank = dist.get_rank(group=ctx.group)
#             gx = torch.empty_like(grad_outputs[rank])
#             gx = _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
#         else:
#             tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
#             gxs = _AlltoAll.apply(ctx.group, tensor_list, grad_outputs)
#             gx = torch.sum(torch.stack(gxs), dim=0)
#         return (None, gx, None)

# NOTE: torch.distributed.all_gather does not carry gradients, we need to implement our own version

def gather_tensors(tensor: Tensor, nums: List[int], dim: int) -> Tensor:
    world_size = len(nums)
    if world_size == 1:
        return tensor
    tensor_shapes = [list(tensor.shape) for _ in range(world_size)]
    for i, num in enumerate(nums):
        tensor_shapes[i][dim] = num
    tensor_list = [tensor.new_empty(size) for size in tensor_shapes]
    # dist.all_gather(tensor_list, tensor)
    _AllGather.apply(dist.group.WORLD, tensor, tensor_list)
    return torch.cat(tensor_list, dim=dim)
    

def scatter_tensors(all_tensors: Tensor, nums: List[int], dim: int) -> Tensor:
    world_size = len(nums)
    if world_size == 1:
        return all_tensors
    rank = comm.get_local_rank()
    cur_shape = list(all_tensors.shape)
    cur_shape[dim] = nums[rank]
    # cur_tensor = all_tensors.new_empty(cur_shape)
    prefix_sum = [0]
    for num in nums:
        prefix_sum.append(prefix_sum[-1] + num)
    indices = torch.arange(prefix_sum[rank], prefix_sum[rank+1], device=all_tensors.device)
    cur_tensor = torch.index_select(all_tensors, dim, indices)
    return cur_tensor