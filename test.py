import torch
import int4_kernel

if __name__ == '__main__':
  torch.set_printoptions(edgeitems=8, linewidth=100)
  grad_y = torch.randn((2, 256, 512), dtype=torch.bfloat16).cuda()
  w = torch.randn((512, 1024), dtype=torch.bfloat16).cuda()
  bit = 4
  
  grad_y_scale = (torch.max(grad_y, dim=-1)[0].unsqueeze(1) - torch.min(grad_y, dim=-1)[0].unsqueeze(1)) / (2 ** bit - 1)
  grad_y_scale = grad_y_scale.transpose(-2, -1)
  
  # forward direction
  w_row_scale = (torch.max(w, dim=-1)[0].unsqueeze(1) - torch.min(w, dim=-1)[0].unsqueeze(1)) / (2 ** bit - 1)
  # backward direction
  w_col_scale = (torch.max(w, dim=0)[0].unsqueeze(0) - torch.min(w, dim=0)[0].unsqueeze(0)) / (2 ** bit - 1)
  print(w_row_scale.shape, w_col_scale.shape)
  
  grad_y_quant_dequant = torch.clamp(torch.round(grad_y / grad_y_scale), -2 ** (bit - 1), 2 ** (bit - 1) - 1) * grad_y_scale
  
  #print(w.shape, w_row_scale.shape)
  w_row_quant_dequant = torch.clamp(torch.round(w / w_row_scale), -2 ** (bit - 1), 2 ** (bit - 1) - 1) * w_row_scale
  #print(w.shape, w_col_scale.shape)
  w_col_quant_dequant = torch.clamp(torch.round(w_row_quant_dequant / w_col_scale), -2 ** (bit - 1), 2 ** (bit - 1) - 1) * w_col_scale
  
  # backward-naive
  grad_x = grad_y @ w
  
  # backward-quantized
  grad_x_quant = grad_y_quant_dequant @ w_col_quant_dequant
  
  # backward-pure int4
  grad_y_int4 = int4_kernel.sym_quant(grad_y, grad_y_scale)
  w_int4 = int4_kernel.sym_quant(w, w_row_scale) # first, quantize in forward direction
  w_int4_transpose = int4_kernel.sym_dequantize_quantize(w_int4, w_row_scale, w_col_scale.transpose(-2, -1)) # then, dequantize in backward direction

  grad_x_int4 = int4_kernel.matmul(grad_y_int4, w_int4_transpose)
  grad_x_int4_dequant = int4_kernel.sym_dequant(grad_x_int4, grad_y_scale, w_col_scale.transpose(-2, -1))
  
  # print('grad_x', grad_x)
  # print('grad_x_quant', grad_x_quant)
  print('grad_x - grad_x_quant', grad_x - grad_x_quant)
  print((grad_x - grad_x_quant).abs().mean())
  print('grad_x_quant - grad_x_int4_dequant', grad_x_quant - grad_x_int4_dequant)
  print((grad_x_quant - grad_x_int4_dequant).abs().mean())
  
  
  '''
  w_row_scale = (torch.max(w, dim=-1)[0].unsqueeze(1) - torch.min(w, dim=-1)[0].unsqueeze(1)) / (2 ** bit - 1)
  w_col_scale = (torch.max(w, dim=0)[0].unsqueeze(0) - torch.min(w, dim=0)[0].unsqueeze(0)) / (2 ** bit - 1)
  
  w_row_quant = int4_kernel.sym_quant(w, w_row_scale)
  w_row_dequant = int4_kernel.sym_dequant_row_only(w_row_quant, w_row_scale)
  w_row_dequant = w_row_dequant.T.contiguous()
  w_col_quant = int4_kernel.sym_quant(w_row_dequant, w_col_scale)
  
  w_int4 = int4_kernel.sym_quant(w, w_row_scale) # first, quantize in forward direction
  w_int4_transpose = int4_kernel.sym_dequantize_quantize(w_int4, w_row_scale, w_col_scale.transpose(-2, -1)) # then, dequantize in backward direction
  
  print(w_col_quant - w_int4_transpose)
  '''
  