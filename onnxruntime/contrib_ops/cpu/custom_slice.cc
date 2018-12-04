#include "custom_slice.h"
#include "core/providers/cpu/tensor/utils.h"
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx

ONNX_OPERATOR_KERNEL_EX(
    CustomSlice,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int32_t>()),
    CustomSlice<float, int32_t>);

template <typename T, typename Tind>
Status CustomSlice<T, Tind>::Compute(OpKernelContext* ctx) const {
  auto& input_tensor = *ctx->Input<Tensor>(0);
  auto& input_dimensions = input_tensor.Shape().GetDims();

  auto& begin_tensor = *ctx->Input<Tensor>(1);
  auto& end_tensor = *ctx->Input<Tensor>(2);

  const Tind* begins = begin_tensor.template Data<Tind>();
  const Tind* ends = end_tensor.template Data<Tind>();

  // Initialize the starts & ends to the actual tensor shape
  const size_t dimension_count = input_dimensions.size();
  std::vector<int64_t> starts(dimension_count, 0);
  std::vector<int64_t> output_dims(input_dimensions);

  for (size_t i = 0; i < dimension_count; ++i) {
      starts[i] = static_cast<int64_t>(begins[i]);
      output_dims[i] = static_cast<int64_t>(ends[i] - begins[i]);
  }

  TensorShape output_shape(output_dims);
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto* output = output_tensor.template MutableData<T>();
  const auto* output_end = output + output_shape.Size();

  SliceIterator<T> input_iterator(input_tensor, starts, output_dims);
  while (output != output_end)
    *output++ = *input_iterator++;

  return Status::OK();
}

}
}
