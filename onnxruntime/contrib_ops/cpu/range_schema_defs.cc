// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "range_schema_defs.h"

#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace contrib {

using ::ONNX_NAMESPACE::OPTIONAL;
using ::ONNX_NAMESPACE::OpSchema;

// This Doc based on LSTM_ver7, and modification
static const char* Range_ver1_doc = R"DOC(
Creates a sequence of numbers that begins at `start` and extends by increments of `delta`
up to but not including `limit`.
)DOC";

OpSchema& RegisterRangeOpSchema(OpSchema&& op_schema){
  return op_schema
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .TypeConstraint(
        "T",
        {"tensor(float)", "tensor(double)", "tensor(int16)", "tensor(int32)", "tensor(int64)" },
        "Constrain input and output types.")
    .Input(
        0,
        "start",
        "Tensor(scalar, or dims=[1]). First entry in the range.",
        "T")
    .Input(
        1,
        "limit",
        "Tensor(scalar, or dims=[1]). Upper limit of sequence, exclusive.",
        "T")
    .Input(
        2,
        "delta",
        "Tensor(scalar, or dims=[1]). Number that increments start. Defaults to 1.",
        "T",
        OpSchema::Optional)
    .Output(
        0,
        "Y",
        "1-D Tensor of the range.",
        "T")
    .SetDoc(Range_ver1_doc);
}

}  // namespace contrib
}  // namespace onnxruntime
