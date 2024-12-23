import torch
import int4_kernel

if __name__ == '__main__':
  # torch.set_printoptions(edgeitems=64, linewidth=1000)
  activation_bit = 4
  x = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
  scale = (torch.max(x, dim=-1)[0].unsqueeze(1) - torch.min(x, dim=-1)[0].unsqueeze(1)) / (2 ** activation_bit - 1) + torch.finfo(torch.float16).smallest_normal
  
  x_quantized = int4_kernel.sym_quant(x, scale)
  x_dequantized = int4_kernel.sym_dequant_row_only(x_quantized, scale)
  print(x)
  print(x_dequantized)
  