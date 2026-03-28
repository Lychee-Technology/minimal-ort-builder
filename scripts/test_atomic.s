	.text
	.file	"test_atomic.c"
	.globl	increment                       // -- Begin function increment
	.p2align	2
	.type	increment,@function
increment:                              // @increment
	.cfi_startproc
// %bb.0:
	mov	w8, #1                          // =0x1
	adrp	x9, counter
	add	x9, x9, :lo12:counter
	ldadd	w8, w8, [x9]
	ret
.Lfunc_end0:
	.size	increment, .Lfunc_end0-increment
	.cfi_endproc
                                        // -- End function
	.globl	increment_seq_cst               // -- Begin function increment_seq_cst
	.p2align	2
	.type	increment_seq_cst,@function
increment_seq_cst:                      // @increment_seq_cst
	.cfi_startproc
// %bb.0:
	mov	w8, #1                          // =0x1
	adrp	x9, counter
	add	x9, x9, :lo12:counter
	ldaddal	w8, w8, [x9]
	ret
.Lfunc_end1:
	.size	increment_seq_cst, .Lfunc_end1-increment_seq_cst
	.cfi_endproc
                                        // -- End function
	.globl	cas_test                        // -- Begin function cas_test
	.p2align	2
	.type	cas_test,@function
cas_test:                               // @cas_test
	.cfi_startproc
// %bb.0:
	mov	w8, w1
	casal	w8, w2, [x0]
	cmp	w8, w1
	cset	w0, eq
	ret
.Lfunc_end2:
	.size	cas_test, .Lfunc_end2-cas_test
	.cfi_endproc
                                        // -- End function
	.globl	exchange_test                   // -- Begin function exchange_test
	.p2align	2
	.type	exchange_test,@function
exchange_test:                          // @exchange_test
	.cfi_startproc
// %bb.0:
	mov	w8, #42                         // =0x2a
	swpa	w8, w0, [x0]
	ret
.Lfunc_end3:
	.size	exchange_test, .Lfunc_end3-exchange_test
	.cfi_endproc
                                        // -- End function
	.globl	fetch_or_test                   // -- Begin function fetch_or_test
	.p2align	2
	.type	fetch_or_test,@function
fetch_or_test:                          // @fetch_or_test
	.cfi_startproc
// %bb.0:
	mov	w8, #255                        // =0xff
	ldset	w8, w8, [x0]
	ret
.Lfunc_end4:
	.size	fetch_or_test, .Lfunc_end4-fetch_or_test
	.cfi_endproc
                                        // -- End function
	.type	counter,@object                 // @counter
	.bss
	.globl	counter
	.p2align	2, 0x0
counter:
	.word	0                               // 0x0
	.size	counter, 4

	.ident	"Apple clang version 17.0.0 (clang-1700.6.4.2)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym counter
