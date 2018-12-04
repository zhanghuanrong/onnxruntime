// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "slice_schema_defs.h"

#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace contrib {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::OPTIONAL;
using ::ONNX_NAMESPACE::OpSchema;

static const char* CustomSlice_ver1_doc = R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `axes`, `starts` and `ends` inputs to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
)DOC";


OpSchema& RegisterCustomSliceOpSchema(OpSchema&& op_schema){
  return op_schema
    .SetDomain(kMSDomain)
    .SetDoc(CustomSlice_ver1_doc)
    .Input(0, "data", "Tensor of data to extract slices from.", "T")
    .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
    .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
    .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
    .Output(0, "output", "Sliced data tensor.", "T")
    .TypeConstraint(
        "T",
        OpSchema::all_tensor_types(),
        "Constrain input and output types to all tensor types.")
    .TypeConstraint(
        "Tind",
        {"tensor(int32)", "tensor(int64)"},
        "Constrain indices to integer types");
}

}
}
