#include "textflag.h"

// func dotProductNEON(a, b []float32) float32
TEXT ·dotProductNEON(SB), NOSPLIT, $0-52
	MOVD a_base+0(FP), R0     // pointer to a
	MOVD a_len+8(FP), R1      // length (element count)
	MOVD b_base+24(FP), R2    // pointer to b

	// Zero accumulators V0-V3
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	// Main loop: 16 float32s per iteration (4 accumulators)
	CMP  $16, R1
	BLT  dot_tail4

dot_loop16:
	VLD1.P 32(R0), [V4.S4, V5.S4]
	VLD1.P 32(R0), [V6.S4, V7.S4]
	VLD1.P 32(R2), [V8.S4, V9.S4]
	VLD1.P 32(R2), [V10.S4, V11.S4]

	// V0 += V8*V4, V1 += V9*V5, V2 += V10*V6, V3 += V11*V7
	VFMLA V4.S4, V8.S4, V0.S4
	VFMLA V5.S4, V9.S4, V1.S4
	VFMLA V6.S4, V10.S4, V2.S4
	VFMLA V7.S4, V11.S4, V3.S4

	SUB $16, R1
	CMP $16, R1
	BGE dot_loop16

dot_tail4:
	// Process 4 float32s at a time
	CMP $4, R1
	BLT dot_reduce

dot_loop4:
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R2), [V5.S4]
	VFMLA V4.S4, V5.S4, V0.S4
	SUB $4, R1
	CMP $4, R1
	BGE dot_loop4

dot_reduce:
	// Combine 4 accumulators into V0
	WORD $0x4E21D400 // fadd v0.4s, v0.4s, v1.4s
	WORD $0x4E23D442 // fadd v2.4s, v2.4s, v3.4s
	WORD $0x4E22D400 // fadd v0.4s, v0.4s, v2.4s

	// Horizontal sum: v0.4s -> s0
	WORD $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7E30D800 // faddp s0, v0.2s

	// Scalar tail: remaining 0-3 elements
	CBZ R1, dot_done

dot_scalar:
	FMOVS (R0), F4
	FMOVS (R2), F5
	FMULS F4, F5, F4
	FADDS F4, F0, F0
	ADD   $4, R0
	ADD   $4, R2
	SUB   $1, R1
	CBNZ  R1, dot_scalar

dot_done:
	FMOVS F0, ret+48(FP)
	RET

// func euclideanDistanceSquaredNEON(a, b []float32) float32
TEXT ·euclideanDistanceSquaredNEON(SB), NOSPLIT, $0-52
	MOVD a_base+0(FP), R0     // pointer to a
	MOVD a_len+8(FP), R1      // length (element count)
	MOVD b_base+24(FP), R2    // pointer to b

	// Zero accumulators V0-V3
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	// Main loop: 16 float32s per iteration
	CMP  $16, R1
	BLT  euc_tail4

euc_loop16:
	VLD1.P 32(R0), [V4.S4, V5.S4]
	VLD1.P 32(R0), [V6.S4, V7.S4]
	VLD1.P 32(R2), [V8.S4, V9.S4]
	VLD1.P 32(R2), [V10.S4, V11.S4]

	// diff = a - b
	WORD $0x4EA8D484 // fsub v4.4s, v4.4s, v8.4s
	WORD $0x4EA9D4A5 // fsub v5.4s, v5.4s, v9.4s
	WORD $0x4EAAD4C6 // fsub v6.4s, v6.4s, v10.4s
	WORD $0x4EABD4E7 // fsub v7.4s, v7.4s, v11.4s

	// acc += diff * diff
	VFMLA V4.S4, V4.S4, V0.S4
	VFMLA V5.S4, V5.S4, V1.S4
	VFMLA V6.S4, V6.S4, V2.S4
	VFMLA V7.S4, V7.S4, V3.S4

	SUB $16, R1
	CMP $16, R1
	BGE euc_loop16

euc_tail4:
	CMP $4, R1
	BLT euc_reduce

euc_loop4:
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R2), [V5.S4]
	WORD $0x4EA5D484 // fsub v4.4s, v4.4s, v5.4s
	VFMLA V4.S4, V4.S4, V0.S4
	SUB $4, R1
	CMP $4, R1
	BGE euc_loop4

euc_reduce:
	// Combine 4 accumulators into V0
	WORD $0x4E21D400 // fadd v0.4s, v0.4s, v1.4s
	WORD $0x4E23D442 // fadd v2.4s, v2.4s, v3.4s
	WORD $0x4E22D400 // fadd v0.4s, v0.4s, v2.4s

	// Horizontal sum: v0.4s -> s0
	WORD $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	WORD $0x7E30D800 // faddp s0, v0.2s

	// Scalar tail: remaining 0-3 elements
	CBZ R1, euc_done

euc_scalar:
	FMOVS (R0), F4
	FMOVS (R2), F5
	FSUBS F5, F4, F4  // F4 = F4 - F5 (a - b)
	FMULS F4, F4, F4  // F4 = diff * diff
	FADDS F4, F0, F0  // F0 += diff^2
	ADD   $4, R0
	ADD   $4, R2
	SUB   $1, R1
	CBNZ  R1, euc_scalar

euc_done:
	FMOVS F0, ret+48(FP)
	RET
