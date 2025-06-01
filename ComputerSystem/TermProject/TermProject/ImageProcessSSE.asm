; void ConvertRGBAToBW_SSE(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight)
; RCX = bufIn, RDX = bufOut, R8D = nWidth, R9D = nHeight

.data
g_fWeightR  dd 3e991687h, 3e991687h, 3e991687h, 3e991687h ; 0.299f
g_fWeightG  dd 3f1645a2h, 3f1645a2h, 3f1645a2h, 3f1645a2h ; 0.587f
g_fWeightB  dd 3df7ced9h, 3df7ced9h, 3df7ced9h, 3df7ced9h ; 0.114f
g_maskR     db 0,0,255,0, 0,0,255,0, 0,0,255,0, 0,0,255,0
g_maskG     db 0,255,0,0, 0,255,0,0, 0,255,0,0, 0,255,0,0
g_maskB     db 255,0,0,0, 255,0,0,0, 255,0,0,0, 255,0,0,0
g_maskA     db 0,0,0,255, 0,0,0,255, 0,0,0,255, 0,0,0,255

.code
ConvertRGBAToBW_SSE_ PROC
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32

    mov     rsi, rcx                ; bufIn
    mov     rdi, rdx                ; bufOut
    mov     ecx, r8d                ; nWidth
    mov     edx, r9d                ; nHeight

    imul    ecx, 4                  ; nWidth4 = nWidth * 4
    mov     r10d, ecx               ; r10d = nWidth4
    xor     r11d, r11d              ; j = 0

    movaps  xmm4, g_fWeightR
    movaps  xmm5, g_fWeightG
    movaps  xmm6, g_fWeightB

RowLoop:
    cmp     r11d, edx
    jge     Done

    xor     r8d, r8d                ; i = 0

PixelLoop:
    cmp     r8d, r10d
    jge     NextRow

    mov     rax, r11
    imul    rax, r10
    add     rax, r8
    movdqu  xmm0, xmmword ptr [rsi + rax]

    ; ÇÈ¼¿ ºÐ¸®
    movdqa  xmm1, xmm0
    pand    xmm1, xmmword ptr g_maskR
    psrld   xmm1, 16

    movdqa  xmm2, xmm0
    pand    xmm2, xmmword ptr g_maskG
    psrld   xmm2, 8

    movdqa  xmm3, xmm0
    pand    xmm3, xmmword ptr g_maskB

    pxor    xmm7, xmm7
    punpcklbw xmm1, xmm7
    punpcklbw xmm2, xmm7
    punpcklbw xmm3, xmm7

    cvtdq2ps xmm1, xmm1
    cvtdq2ps xmm2, xmm2
    cvtdq2ps xmm3, xmm3

    mulps   xmm1, xmm4
    mulps   xmm2, xmm5
    mulps   xmm3, xmm6

    addps   xmm1, xmm2
    addps   xmm1, xmm3

    cvtps2dq xmm1, xmm1

    movdqa  xmm2, xmm1
    packssdw xmm1, xmm2
    movdqa  xmm2, xmm1
    packuswb xmm1, xmm2

    movdqa  xmm2, xmmword ptr [rsi + rax]
    pand    xmm2, xmmword ptr g_maskA

    movdqa  xmm3, xmm1
    pslld   xmm3, 8
    por     xmm1, xmm3
    movdqa  xmm3, xmm1
    pslld   xmm3, 8
    por     xmm1, xmm3
    por     xmm1, xmm2

    movdqu  xmmword ptr [rdi + rax], xmm1

    add     r8d, 16
    jmp     PixelLoop

NextRow:
    inc     r11d
    jmp     RowLoop

Done:
    add     rsp, 32
    pop     rbp
    ret
ConvertRGBAToBW_SSE_ ENDP
end