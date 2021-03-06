# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_optimizer_srcs
    "${ONNXRUNTIME_ROOT}/core/optimizer/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/*.cc"
    )

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_optimizer_srcs})

add_library(onnxruntime_optimizer ${onnxruntime_optimizer_srcs})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/optimizer  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
onnxruntime_add_include_to_target(onnxruntime_optimizer onnxruntime_common onnxruntime_framework gsl onnx onnx_proto protobuf::libprotobuf)
target_include_directories(onnxruntime_optimizer PRIVATE ${ONNXRUNTIME_ROOT})
add_dependencies(onnxruntime_optimizer ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_optimizer PROPERTIES FOLDER "ONNXRuntime")
