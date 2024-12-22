import torch
import int4_kernel._CUDA

__all__ = [ 
  "matmul", #int-4 matmul
  "sym_quant", "sym_dequant", "PackedQuantizedTensor", # Quantization
  "sym_dequant_row_only", "sym_dequant_col_only", # Quantization
]

class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)

def flatten_last_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    return x, shape_excl_last

def flatten_first_dim_and_return_shape(x: torch.Tensor):
    shape_excl_first = x.shape[1:]
    x = x.view(x.shape[0], -1)
    return x, shape_excl_first

def matmul(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return int4_kernel._CUDA.matmul(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def sym_quant(x, scale):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return int4_kernel._CUDA.sym_quant(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_dequant(q, scale_row, scale_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return int4_kernel._CUDA.sym_dequant(q, scale_row.view(-1), scale_col, bits).view(*q_shape_excl_last, -1)

def sym_dequant_row_only(q, scale_row, bits=32):
    assert q.dtype == torch.int8
    assert scale_row.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return int4_kernel._CUDA.sym_dequant_row_only(q, scale_row.view(-1), bits).view(*q_shape_excl_last, -1)

def sym_dequant_col_only(q, scale_col, bits=32):
    assert q.dtype == torch.int8
    assert scale_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_first_dim_and_return_shape(q)
    return int4_kernel._CUDA.sym_dequant_col_only(q, scale_col.view(-1), bits).view(-1, *q_shape_excl_last)

def sym_dequantize_quantize(q_in, scale_row, scale_col, bits=32):
    assert q_in.dtype == torch.int8
    assert scale_row.dtype == scale_col.dtype == torch.float16
    q_in, q_in_shape_excl_last = flatten_last_dim_and_return_shape(q_in)
    return int4_kernel._CUDA.sym_dequantize_quantize(q_in, scale_row, scale_col) #.view(*q_in_shape_excl_last, -1)


class PackedQuantizedTensor:
    def __init__(self, 
                 quantized_x: torch.Tensor, 
                 scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype
