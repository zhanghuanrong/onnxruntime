#include "range.h"
#include "onnx/defs/schema.h"

#include <cmath>

namespace onnxruntime {
namespace contrib {

bool IsScalarLike(const Tensor* t) {
  return (t == nullptr || (t->Shape().NumDimensions() <= 1 && t->Shape().Size() == 1));
}

template <typename T>
Status TemplatedCompute(OpKernelContext* ctx) {
  if (!IsScalarLike(ctx->Input<Tensor>(0))) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  
                                   "start in Range operator should be scalar like tensor, yet got shape:",
                                   ctx->Input<Tensor>(0)->Shape());
  }
  if (!IsScalarLike(ctx->Input<Tensor>(1))) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  
                                   "limit in Range operator should be scalar like tensor, yet got shape:",
                                   ctx->Input<Tensor>(1)->Shape());
  }
  if (!IsScalarLike(ctx->Input<Tensor>(2))) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  
                                   "delta in Range operator should be scalar like tensor, yet got shape:",
                                   ctx->Input<Tensor>(2)->Shape());
  }

  T start = *(ctx->Input<Tensor>(0)->template Data<T>());
  T limit = *(ctx->Input<Tensor>(1)->template Data<T>());
  T delta = (ctx->Input<Tensor>(2) == nullptr) ? T{1} : *(ctx->Input<Tensor>(2)->template Data<T>());
  
  if (delta == T{0}) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  "delta in Range operator can not be zero");
  }

  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
  if (n <= 0) n = 1;
  TensorShape shape = {n};
  T* y = ctx->Output(0, shape)->template MutableData<T>();
  for (int64_t i = 0; i < n; ++i) {
      *y++ = start;
      start += delta;
  }

  return Status::OK();
}

Status Range::Compute(OpKernelContext* ctx) const {
  auto data_type = ctx->Input<Tensor>(0)->DataType();
  if (data_type == DataTypeImpl::GetType<int32_t>()) {
      return TemplatedCompute<int32_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int16_t>()) {
      return TemplatedCompute<int16_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int64_t>()) {
      return TemplatedCompute<int64_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<float>()) {
      return TemplatedCompute<float>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<double>()) {
      return TemplatedCompute<double>(ctx);
  }
  return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  
                                 "Unsupportted tensor data type:",
                                 data_type);
}

/* Range operator */
ONNX_OPERATOR_KERNEL_EX(
    Range,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {
        DataTypeImpl::GetTensorType<float>(), 
        DataTypeImpl::GetTensorType<double>(),
        DataTypeImpl::GetTensorType<int16_t>(), 
        DataTypeImpl::GetTensorType<int32_t>(), 
        DataTypeImpl::GetTensorType<int64_t>()}),
    Range);


}  // namespace contrib
}  // namespace onnxruntime
