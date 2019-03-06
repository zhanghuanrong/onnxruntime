/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    TanhKernelFma3.s

Abstract:

    This module implements a kernel for computing the hyperbolic tangent
    function for a buffer of elements.

    This implementation uses AVX fused multiply/add instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

//
// Structure layout for the exp constants block.
//

        .equ    ExpConstants_LowerRange, 0
        .equ    ExpConstants_UpperRange, 4
        .equ    ExpConstants_LOG2EF, 8
        .equ    ExpConstants_C1, 12
        .equ    ExpConstants_C2, 16
        .equ    ExpConstants_P0, 20
        .equ    ExpConstants_P1, 24
        .equ    ExpConstants_P2, 28
        .equ    ExpConstants_P3, 32
        .equ    ExpConstants_P4, 36
        .equ    ExpConstants_P5, 40
        .equ    ExpConstants_One, 44
        .equ    ExpConstants_Half, 48
        .equ    ExpConstants_X7f, 52

//
// Stack frame layout for the tanh kernel.
//

        .equ    ExpKernelFrame_CountN, -8
        .equ    ExpKernelFrame_ReturnAddress, 0

/*++

Routine Description:

    This routine implements a vectorized kernel for the exp()
    function.

Arguments:

    Input (rdi) - Supplies the input buffer.

    Output (rsi) - Supplies the output buffer.

    N (rdx)  - Supplies the number of elements to process.

Return Value:

    None.

--*/

        .globl  C_UNDERSCORE(MlasExpKernelFma3)
C_UNDERSCORE(MlasExpKernelFma3):

        lea     rax,C_UNDERSCORE(MlasExpConstants)[rip]
        vbroadcastss ymm4,ExpConstants_LowerRange[rax]
        vbroadcastss ymm5,ExpConstants_UpperRange[rax]
        vbroadcastss ymm7,ExpConstants_C1[rax]
        vbroadcastss ymm8,ExpConstants_C2[rax]
        vbroadcastss ymm9,ExpConstants_P0[rax]
        vbroadcastss ymm10,ExpConstants_P1[rax]
        vbroadcastss ymm11,ExpConstants_P2[rax]
        vbroadcastss ymm12,ExpConstants_P3[rax]
        vbroadcastss ymm13,ExpConstants_P4[rax]
        vbroadcastss ymm14,ExpConstants_P5[rax]
        vbroadcastss ymm15,ExpConstants_LOG2EF[rax]

        sub     rdx,8
        jb      .LProcessRemainingCount

.LComputeExpBy8Loop:
        vmovups ymm4, YMMWORD PTR [rdi]
        vmaxps  ymm0,ymm5,ymm4                  # clamp lower bound
        vminps  ymm0,ymm6,ymm0                  # clamp upper bound
        vbroadcastss ymm3,ExpConstants_Half[rax]
        vmovaps ymm2,ymm15                      # MlasExpConstants.LOG2EF
        vfmadd231ps ymm2,ymm0,ymm3              # fx = x * MlasExpConstants.LOG2EF + 0.5f)
        vroundps ymm2,ymm2,0xB                  # fx = floor(fx)
        vmulps ymm1,ymm2,ymm7                   # tmp(ymm1) = fx * c1
        vmulps ymm3,ymm2,ymm8                   # z(ymm3) = fx * c2
        vsubps ymm0,ymm0,ymm1                   # x = x - tmp
        vsubps ymm0,ymm0,ymm3                   # x = x - z
        vmulps ymm3,ymm0,ymm0                   # z = x * x

        vmovaps ymm1,ymm9                       # y = MlasExpConstants.P0 
        vfmadd231ps ymm1,ymm0,ymm10             # y = y * x + MlasExpConstants.P1;
        vfmadd231ps ymm1,ymm0,ymm11             # y = y * x + MlasExpConstants.P2;
        vfmadd231ps ymm1,ymm0,ymm12             # y = y * x + MlasExpConstants.P3;
        vfmadd231ps ymm1,ymm0,ymm13             # y = y * x + MlasExpConstants.P4;
        vfmadd231ps ymm1,ymm0,ymm14             # y = y * x + MlasExpConstants.P5;
        vfmadd231ps ymm1,ymm3,ymm0              # y = y * z + x;
        vbroadcastss ymm3,ExpConstants_One[rax] 
        vaddps ymm1, ymm3                       # y = y + 1.0

        vcvttps2dq ymm2,ymm2
        vbroadcastss ymm3,ExpConstants_X7f[rax]
        vpaddd ymm2,ymm2,ymm3                   # emm0 =  Covert_to_Int(fx) + 0x7f 
        pslld ymm2, 23                          # emm0 << 23
        vmulps ymm1,ymm1,ymm2                   # y = y * log2(fx)
        vmaxps ymm0,ymm1,ymm4                   # exp = max(y, input)

        add     rdi,8*4                         # advance input by 8 elements
        vmovups YMMWORD PTR [rsi],ymm0
        add     rsi,8*4                         # advance output by 8 elements
        sub     rdx,8
        jae     .LComputeTanhBy8Loop

.LProcessRemainingCount:
        add     rdx,8                           # correct for over-subtract above
        jz      .LExitKernel
        mov     DWORD PTR ExpKernelFrame_CountN[rsp],edx
        vbroadcastss ymm4,DWORD PTR ExpKernelFrame_CountN[rsp]
        vpcmpgtd ymm4,ymm4,YMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]

        vmaxps  ymm0,ymm5,YMMWORD PTR [rdi]     # clamp lower bound
        vminps  ymm0,ymm6,ymm0                  # clamp upper bound
        vbroadcastss ymm3,ExpConstants_Half[rax]
        vmovaps ymm2,ymm15                      # MlasExpConstants.LOG2EF
        vfmadd231ps ymm2,ymm0,ymm3              # fx = x * MlasExpConstants.LOG2EF + 0.5f)
        vroundps ymm2,ymm2,0xB                  # fx = floor(fx)
        vmulps ymm1,ymm2,ymm7                   # tmp(ymm1) = fx * c1
        vmulps ymm3,ymm2,ymm8                   # z(ymm3) = fx * c2
        vsubps ymm0,ymm0,ymm1                   # x = x - tmp
        vsubps ymm0,ymm0,ymm3                   # x = x - z
        vmulps ymm3,ymm0,ymm0                   # z = x * x

        vmovaps ymm1,ymm9                       # y = MlasExpConstants.P0 
        vfmadd231ps ymm1,ymm0,ymm10             # y = y * x + MlasExpConstants.P1;
        vfmadd231ps ymm1,ymm0,ymm11             # y = y * x + MlasExpConstants.P2;
        vfmadd231ps ymm1,ymm0,ymm12             # y = y * x + MlasExpConstants.P3;
        vfmadd231ps ymm1,ymm0,ymm13             # y = y * x + MlasExpConstants.P4;
        vfmadd231ps ymm1,ymm0,ymm14             # y = y * x + MlasExpConstants.P5;
        vfmadd231ps ymm1,ymm3,ymm0              # y = y * z + x;
        vbroadcastss ymm3,ExpConstants_One[rax] 
        vaddps ymm1, ymm3                       # y = y + 1.0

        vcvttps2dq ymm2,ymm2
        vbroadcastss ymm3,ExpConstants_X7f[rax]
        vpaddd ymm2,ymm2,ymm3                   # emm0 =  Covert_to_Int(fx) + 0x7f 
        pslld ymm2, 23                          # emm0 << 23
        vmulps ymm1,ymm1,ymm2                   # y = y * log2(fx)
        vmaxps ymm0,ymm1,YMMWORD PTR [rdi]      # exp = max(y, input)

        vmaskmovps YMMWORD PTR [rsi],ymm4,ymm0

.LExitKernel:
        vzeroupper
        ret

        .end
