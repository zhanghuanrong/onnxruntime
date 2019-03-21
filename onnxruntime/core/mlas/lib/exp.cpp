/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    exp.cpp

Abstract:

    This module implements routines to compute the exp function.

    This implementation uses the same polynomial coefficients and algorithm as
    found in Eigen. Our usage requires building platform specific versions of
    the algorithm to target different instruction sets. The implementation below
    targets the base instruction set (typically SSE2) while assembly
    implementations target newer instruction sets (such as FMA3).

--*/

#include "mlasi.h"
#include <cmath>

//
// Bundles the floating point constants for use by kernels written in assembly.
//

extern "C" const struct {
    float UpperRange;
    float LowerRange;
    float LOG2EF;
    float C1;
    float C2;
    float P0;
    float P1;
    float P2;
    float P3;
    float P4;
    float P5;
    float One;
    float Half;
    int32_t X7f;
} MlasExpConstants = {
    88.3762626647950f, 
    -88.3762626647949f, 
    1.44269504088896341f, 
    0.693359375f, 
    -2.12194440e-4f, 
    1.9875691500E-4f, 
    1.3981999507E-3f, 
    8.3334519073E-3f, 
    4.1665795894E-2f, 
    1.6666665459E-1f, 
    5.0000001201E-1f,
    1.0f,
    0.5f,
    0x7f
};

void
MLASCALL
MlasExpKernel(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the exp() function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    while (N >= 4) {
        MLAS_FLOAT32X4 _x = MlasLoadFloat32x4(Input);
        MLAS_FLOAT32X4 x = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(MlasExpConstants.LowerRange), _x);
        x = MlasMinimumFloat32x4(MlasBroadcastFloat32x4(MlasExpConstants.UpperRange), x);

        MLAS_FLOAT32X4 fx = MlasMultiplyAddFloat32x4(x, MlasBroadcastFloat32x4(MlasExpConstants.LOG2EF), MlasBroadcastFloat32x4(MlasExpConstants.Half));
        fx = MlasFloorFloat32x4(fx);
        MLAS_FLOAT32X4 tmp = MlasMultiplyFloat32x4(fx, MlasBroadcastFloat32x4(MlasExpConstants.C1));
        MLAS_FLOAT32X4 z = MlasMultiplyFloat32x4(fx, MlasBroadcastFloat32x4(MlasExpConstants.C2));
        x = MlasSubtractFloat32x4(x, tmp);
        x = MlasSubtractFloat32x4(x, z);
        z = MlasMultiplyFloat32x4(x, x);

        MLAS_FLOAT32X4 y = MlasBroadcastFloat32x4(MlasExpConstants.P0);
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P1));
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P2));
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P3));
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P4));
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P5));
        y = MlasMultiplyAddFloat32x4(y, z, x);
        y = MlasAddFloat32x4(y, MlasBroadcastFloat32x4(MlasExpConstants.One));

        // build 2^n
        MLAS_FLOAT32X4 emm0 = MlasPowerOf2Float32x4(fx);
        y = MlasMaximumFloat32x4(MlasMultiplyFloat32x4(y, emm0), _x);

        MlasStoreFloat32x4(Output, y);

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {
        float _x = *Input++;

        float x = (std::min)(MlasExpConstants.UpperRange, (std::max)(MlasExpConstants.LowerRange, _x));

        float fx = std::floor(x * MlasExpConstants.LOG2EF + MlasExpConstants.Half);
        float tmp = fx * MlasExpConstants.C1;
        float z = fx * MlasExpConstants.C2;
        
        x = x - tmp - z;
        z = x * x;

        float y = MlasExpConstants.P0 * x + MlasExpConstants.P1;
        y = y * x + MlasExpConstants.P2;
        y = y * x + MlasExpConstants.P3;
        y = y * x + MlasExpConstants.P4;
        y = y * x + MlasExpConstants.P5;
        y = y * z + x;
        y = y + MlasExpConstants.One;
        
        y = ldexpf(y, static_cast<int>(fx));
        y = (std::max)(y, _x);

        *Output++ = y;

        --N;
    }
}

void
MLASCALL
MlasComputeExp(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the hyperbolic tangent function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.ExpKernelRoutine(Input, Output, N);
#else
    MlasExpKernel(Input, Output, N);
#endif
}
