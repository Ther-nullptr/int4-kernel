#include <torch/extension.h>

// Include all files
#include <gemm.h>
#include <quant.h>

torch::Tensor matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

  torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}});
  uint32_t M = A.size(0);
  uint32_t N = B.size(0);
  uint32_t K = A.size(1) * kElementsPerVector; // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  matmul_host(A.data_ptr<Int4Storage>(), B.data_ptr<Int4Storage>(), M, N, K,
              C.data_ptr<int32_t>());

  return C;
}

torch::Tensor sym_quant(const torch::Tensor &x, const torch::Tensor &scale) {
  torch::checkAllContiguous("sym_quant", {{x, "x", 0}, {scale, "scale", 1}});
  torch::checkDeviceType("sym_quant", {x, scale}, at::DeviceType::CUDA);

  torch::checkSameGPU("sym_quant", {x, "x", 0}, {scale, "scale", 1});
  torch::checkSize("sym_quant", torch::TensorArg{scale, "scale", 1}, 0,
                   x.size(0));
  uint32_t rows = x.size(0);
  uint32_t colsSrc = x.size(1);
  uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

  auto q = torch::empty({rows, colsDst},
                        torch::dtype(torch::kUInt8).device(x.device()));

  sym_quant_host((half *)x.data_ptr(), (half *)scale.data_ptr(), rows, colsSrc,
                 colsDst, q.data_ptr<Int4Storage>());

  return q;
}

torch::Tensor sym_dequant(const torch::Tensor &q,
                          const torch::Tensor &scale_row,
                          const torch::Tensor &scale_col, const int bits) {
  torch::checkAllContiguous(
      "sym_dequant",
      {{q, "q", 0}, {scale_row, "scale_row", 1}, {scale_col, "scale_col", 2}});
  torch::checkDeviceType("sym_dequant", {q, scale_row, scale_col},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU(
      "sym_dequant",
      {{q, "q", 0}, {scale_row, "scale_row", 1}, {scale_col, "scale_col", 2}});

  uint32_t rows = q.size(0);
  uint32_t cols = q.size(1);

  torch::checkSize("sym_dequant", torch::TensorArg{scale_row, "scale_row", 1},
                   0, rows);
  torch::checkSize("sym_dequant", torch::TensorArg{scale_col, "scale_col", 2},
                   0, cols);

  auto x =
      torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));

  switch (bits) {
  case 32:
    sym_dequant_host(q.data_ptr<int32_t>(), (half *)scale_row.data_ptr(),
                     (half *)scale_col.data_ptr(), rows, cols,
                     (half *)x.data_ptr());
    break;
  default:
    TORCH_CHECK(false, "Unsupported data type")
  }

  return x;
}

torch::Tensor sym_dequant_col_only(const torch::Tensor &q,
                                   const torch::Tensor &scale_col, const int bits) {
  torch::checkAllContiguous("sym_dequant_col_only", {{q, "q", 0}, {scale_col, "scale_col", 1}});
  torch::checkDeviceType("sym_dequant_col_only", {q, scale_col},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant_col_only", {{q, "q", 0}, {scale_col, "scale_col", 1}});

  uint32_t rowsSrc = q.size(0);
  uint32_t rowsDst = rowsSrc * kElementsPerVector;
  uint32_t cols = q.size(1);

  torch::checkSize("sym_dequant_col_only", torch::TensorArg{scale_col, "scale_col", 1},
                   0, cols);

  auto x =
      torch::empty({rowsDst, cols}, torch::dtype(torch::kHalf).device(q.device()));

  switch (bits) {
  case 32:
    sym_dequant_col_only_host(q.data_ptr<int8_t>(), (half *)scale_col.data_ptr(),
                              rowsSrc, rowsDst, cols, (half *)x.data_ptr());
    break;
  default:
    TORCH_CHECK(false, "Unsupported data type")
  }

  return x;
}

torch::Tensor sym_dequant_row_only(const torch::Tensor &q,
                                   const torch::Tensor &scale_row, const int bits) {
  torch::checkAllContiguous("sym_dequant_row_only", {{q, "q", 0}, {scale_row, "scale_row", 1}});
  torch::checkDeviceType("sym_dequant_row_only", {q, scale_row},
                         at::DeviceType::CUDA);

  torch::checkAllSameGPU("sym_dequant_row_only", {{q, "q", 0}, {scale_row, "scale_row", 1}});

  uint32_t rows = q.size(0);
  uint32_t colsSrc = q.size(1);
  uint32_t colsDst = colsSrc * kElementsPerVector;

  torch::checkSize("sym_dequant_row_only", torch::TensorArg{scale_row, "scale_row", 1},
                   0, rows);

  auto x =
      torch::empty({rows, colsDst}, torch::dtype(torch::kHalf).device(q.device()));

  switch (bits) {
  case 32:
    sym_dequant_row_only_host(q.data_ptr<int8_t>(), (half *)scale_row.data_ptr(),
                              rows, colsSrc, colsDst, (half *)x.data_ptr());
    break;
  default:
    TORCH_CHECK(false, "Unsupported data type")
  }

  return x;
}



//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("matmul", &matmul,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));

  m.def("sym_quant", &sym_quant,
        "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
        "torch.Tensor(M x 1, FP16, CUDA))"
        "bits: int\n"
        "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"));

  m.def("sym_dequant", &sym_dequant,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, "
        "FP16), scale_col: torch.Tensor(1 x N, FP16)"
        "bits: int\n"
        "output: torch.Tensor(M x N, FP16)\n"
        "output = x * scale_row * scale_col"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is FP16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("q"), py::arg("scale_row"), py::arg("scale_col"),
        py::arg("bits"));

  m.def("sym_dequant_col_only", &sym_dequant_col_only,
        "input (x: torch.Tensor(M x N), scale_col: torch.Tensor(1 x N, FP16)"
        "bits: int\n"
        "output: torch.Tensor(M x N, FP16)\n"
        "output = x * scale_col"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is FP16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("q"), py::arg("scale_col"), py::arg("bits"));
  
  m.def("sym_dequant_row_only", &sym_dequant_row_only,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, FP16)"
        "bits: int\n"
        "output: torch.Tensor(M x N, FP16)\n"
        "output = x * scale_row"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is FP16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("q"), py::arg("scale_row"), py::arg("bits"));
}
