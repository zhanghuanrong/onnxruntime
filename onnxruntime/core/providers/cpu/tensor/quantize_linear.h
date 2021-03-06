// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
class DequantizeLinear final : public OpKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : OpKernel(info) { }

  Status Compute(OpKernelContext* context) const override; 
};

template <typename T>
class QuantizeLinear final : public OpKernel {
 public:
  QuantizeLinear(const OpKernelInfo& info) : OpKernel(info) { }

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace onnxruntime
