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
// Structure layout for the tanh constants block.
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
        .equ    ExpConstants_ExponetZeroInFloat, 52

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
        vbroadcastss ymm6,ExpConstants_alpha_13[rax]
        vbroadcastss ymm7,ExpConstants_alpha_11[rax]
        vbroadcastss ymm8,ExpConstants_alpha_9[rax]
        vbroadcastss ymm9,ExpConstants_alpha_7[rax]
        vbroadcastss ymm10,ExpConstants_alpha_5[rax]
        vbroadcastss ymm11,ExpConstants_alpha_3[rax]
        vbroadcastss ymm12,ExpConstants_alpha_1[rax]
        vbroadcastss ymm13,ExpConstants_beta_6[rax]
        vbroadcastss ymm14,ExpConstants_beta_2[rax]
        vbroadcastss ymm15,ExpConstants_beta_0[rax]

        sub     rdx,8
        jb      .LProcessRemainingCount

.LComputeTanhBy8Loop:
        vmaxps  ymm0,ymm4,YMMWORD PTR [rdi]     # clamp lower bound
        vmovaps ymm2,ymm7
        vminps  ymm0,ymm5,ymm0                  # clamp upper bound
        vmulps  ymm1,ymm0,ymm0                  # x2
        vbroadcastss ymm3,ExpConstants_beta_4[rax]
        vfmadd231ps ymm2,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11
        vfmadd213ps ymm2,ymm1,ymm8              # p = x2 * p + alpha_9
        vfmadd213ps ymm2,ymm1,ymm9              # p = x2 * p + alpha_7
        vfmadd213ps ymm2,ymm1,ymm10             # p = x2 * p + alpha_5
        vfmadd213ps ymm2,ymm1,ymm11             # p = x2 * p + alpha_3
        vfmadd213ps ymm2,ymm1,ymm12             # p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0
        vmulps  ymm2,ymm0,ymm2                  # p = x * p
        vdivps  ymm0,ymm2,ymm3                  # tanh = p / q
        add     rdi,8*4                         # advance input by 8 elements
        vmovups YMMWORD PTR [rsi],ymm0
        add     rsi,8*4                         # advance output by 8 elements
        sub     rdx,8
        jae     .LComputeTanhBy8Loop

.LProcessRemainingCount:
        add     rdx,8                           # correct for over-subtract above
        jz      .LExitKernel
        mov     DWORD PTR TanhKernelFrame_CountN[rsp],edx
        vbroadcastss ymm2,DWORD PTR TanhKernelFrame_CountN[rsp]
        vpcmpgtd ymm2,ymm2,YMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        vmaskmovps ymm0,ymm2,YMMWORD PTR [rdi]
        vmaxps  ymm0,ymm4,ymm0                  # clamp lower bound
        vminps  ymm0,ymm5,ymm0                  # clamp upper bound
        vmulps  ymm1,ymm0,ymm0                  # x2
        vbroadcastss ymm3,ExpConstants_beta_4[rax]
        vfmadd231ps ymm7,ymm1,ymm6              # p = x2 * alpha_13 + alpha_11
        vfmadd213ps ymm7,ymm1,ymm8              # p = x2 * p + alpha_9
        vfmadd213ps ymm7,ymm1,ymm9              # p = x2 * p + alpha_7
        vfmadd213ps ymm7,ymm1,ymm10             # p = x2 * p + alpha_5
        vfmadd213ps ymm7,ymm1,ymm11             # p = x2 * p + alpha_3
        vfmadd213ps ymm7,ymm1,ymm12             # p = x2 * p + alpha_1
        vfmadd231ps ymm3,ymm1,ymm13             # q = x2 * beta_6 + beta_4
        vfmadd213ps ymm3,ymm1,ymm14             # q = x2 * q + beta_2
        vfmadd213ps ymm3,ymm1,ymm15             # q = x2 * q + beta_0
        vmulps  ymm7,ymm0,ymm7                  # p = x * p
        vdivps  ymm0,ymm7,ymm3                  # tanh = p / q
        vmaskmovps YMMWORD PTR [rsi],ymm2,ymm0

.LExitKernel:
        vzeroupper
        ret

        .end
