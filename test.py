import torch
import int4_kernel

if __name__ == '__main__':
  torch.set_printoptions(edgeitems=64, linewidth=1000)
  # # row test
  # w = torch.randint(-32, 31, (1024, 512), device='cuda', dtype=torch.int8)
  # scale_row = torch.randn((1024,), device='cuda', dtype=torch.float16)
  # w_dequant = int4_kernel.sym_dequant_row_only(w, scale_row)
  # print(w_dequant.shape)
  # print(w_dequant)
  
  # w_unpack = torch.zeros((1024, 1024), device='cuda', dtype=torch.int8)
  # w_unpack[:, ::2] = w & 0xf
  # w_unpack[:, 1::2] = (w >> 4) & 0xf
  # w_dequant_torch = w_unpack * scale_row.view(-1, 1)
  # print(w_dequant_torch.shape)
  # print(w_dequant_torch)
  
  # # col test
  # w = torch.randint(-32, 31, (256, 512), device='cuda', dtype=torch.int8)
  # scale_col = torch.randn((512), device='cuda', dtype=torch.float16)
  # w_dequant = int4_kernel.sym_dequant_col_only(w, scale_col)
  # print(w_dequant.shape)
  # print(w_dequant)
  
  # w_unpack = torch.zeros((512, 512), device='cuda', dtype=torch.int8)
  # w_unpack[::2] = w & 0xf
  # w_unpack[1::2] = (w >> 4) & 0xf
  # w_dequant_torch = w_unpack * scale_col.view(1, -1)
  # print(w_dequant_torch.shape)
  # print(w_dequant_torch)
  
  # quantize dequantize seperate test
  w = torch.randn((2048, 4096), device='cuda', dtype=torch.float16)
  scale_row = torch.randn((2048,), device='cuda', dtype=torch.float16)
  scale_col = torch.randn((4096,), device='cuda', dtype=torch.float16)
  w_quant = int4_kernel.sym_quant(w, scale_row)
  w_dequant = int4_kernel.sym_dequant_row_only(w_quant, scale_row)
  w_dequant = w_dequant.T.contiguous()
  w_dequant_quant = int4_kernel.sym_quant(w_dequant, scale_col)
  print(w_dequant_quant.shape)
  print(w_dequant_quant)
  
  # quantize dequantize combined test
  w_dequant_quant_2 = int4_kernel.sym_dequantize_quantize(w_quant, scale_row, scale_col)
  print(w_dequant_quant_2.shape)
  print(w_dequant_quant_2)
  
  print(w_dequant_quant - w_dequant_quant_2)
  
  