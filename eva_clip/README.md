# Modification record

The code for the current directory is derived from [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP/rei/eva_clip).

Compared to the original code, we have made the following adjustments.

- Removes the dependency on `deepspeed`.
- Remove the dependence on `xformers` and replace them with [sdpa](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), which was supported after `Pytorch 2.0`.
- Removes the dependency on `FusedLayerNorm`.
- Add interpolation to the [RoPE](rope.py).
