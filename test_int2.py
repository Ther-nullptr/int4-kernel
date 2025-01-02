import torch
import int4_kernel

if __name__ == '__main__':
  # randomly set seed
  torch.manual_seed(0)
  x = torch.randn((2, 1024, 1024), dtype=torch.bfloat16, device='cuda')
  s = (torch.max(x, dim=-1)[0].unsqueeze(1) - torch.min(x, dim=-1)[0].unsqueeze(1)) / (2 ** 2 - 1) + torch.finfo(torch.bfloat16).smallest_normal 
  # quantize
  x_quant = int4_kernel.sym_quant_int2(x, s)
  print(x_quant)
  # dequantize
  x_dequant = int4_kernel.sym_dequant_row_only_int2(x_quant, s)
  print(x_dequant.abs().mean())
  print(x.abs().mean())
  print((x - x_dequant).abs().mean())