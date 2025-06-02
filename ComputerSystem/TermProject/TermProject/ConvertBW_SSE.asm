; void ConvertRGBAToBW_SSE(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight)
; RCX = bufIn, RDX = bufOut, R8D = nWidth, R9D = nHeight

.data
zero16      dq 0, 0
g_fWeightR  dd 3e991687h, 3e991687h, 3e991687h, 3e991687h ; 0.299f
g_fWeightG  dd 3f1645a2h, 3f1645a2h, 3f1645a2h, 3f1645a2h ; 0.587f
g_fWeightB  dd 3df7ced9h, 3df7ced9h, 3df7ced9h, 3df7ced9h ; 0.114f
g_maskR     db 0,0,255,0, 0,0,255,0, 0,0,255,0, 0,0,255,0
g_maskG     db 0,255,0,0, 0,255,0,0, 0,255,0,0, 0,255,0,0
g_maskB     db 255,0,0,0, 255,0,0,0, 255,0,0,0, 255,0,0,0
g_maskA     db 0,0,0,255, 0,0,0,255, 0,0,0,255, 0,0,0,255

.code
ConvertRGBAToBW_SSE_ PROC
    ; Windows x64 ABI callee-saved
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rsi, rcx                ; bufIn
    mov     rdi, rdx                ; bufOut
    mov     ecx, r8d                ; nWidth
    mov     edx, r9d                ; nHeight

    imul    ecx, 4                  ; nWidth4 = nWidth * 4
    mov     r10d, ecx               ; r10d = nWidth4
    xor     r11d, r11d              ; j = 0 (row)

    movaps  xmm4, g_fWeightR
    movaps  xmm5, g_fWeightG
    movaps  xmm6, g_fWeightB

HeightLoop:
    cmp     r11d, edx
    jge     Done

    xor     r8d, r8d                ; i = 0 (col)
WidthLoop:
    cmp     r8d, r10d
    jge     NextRow

    mov     rax, r11
    imul    rax, r10
    add     rax, r8

    movdqu  xmm0, xmmword ptr [rsi + rax] ; 4픽셀(RGBA*4) 로드

    ; R 추출 및 변환
    movdqa  xmm1, xmm0                        ; xmm0(입력)에서 복사하여 xmm1에 저장
    pand    xmm1, xmmword ptr g_maskR         ; R 채널만 남기고 나머지 비트 0으로 마스킹
    psrld   xmm1, 16                          ; R 채널을 하위 바이트로 이동 (각 픽셀별 16비트 오른쪽 시프트)
    packusdw xmm1, xmm1                       ; 32비트 값을 16비트로 압축(포장)
    punpcklbw xmm1, xmmword ptr [zero16]      ; 8비트 값을 16비트로 zero-extend
    cvtdq2ps xmm1, xmm1                       ; 16비트 정수를 float로 변환
    mulps   xmm1, xmm4                        ; R 가중치(0.299f) 곱셈

    ; G 추출 및 변환
    movdqa  xmm2, xmm0                        ; xmm0(입력)에서 복사하여 xmm2에 저장
    pand    xmm2, xmmword ptr g_maskG         ; G 채널만 남기고 나머지 비트 0으로 마스킹
    psrld   xmm2, 8                           ; G 채널을 하위 바이트로 이동 (각 픽셀별 8비트 오른쪽 시프트)
    packusdw xmm2, xmm2                       ; 32비트 값을 16비트로 압축(포장)
    punpcklbw xmm2, xmmword ptr [zero16]      ; 8비트 값을 16비트로 zero-extend
    cvtdq2ps xmm2, xmm2                       ; 16비트 정수를 float로 변환
    mulps   xmm2, xmm5                        ; G 가중치(0.587f) 곱셈

    ; B 추출 및 변환
    movdqa  xmm3, xmm0                        ; xmm0(입력)에서 복사하여 xmm3에 저장
    pand    xmm3, xmmword ptr g_maskB         ; B 채널만 남기고 나머지 비트 0으로 마스킹
    packusdw xmm3, xmm3                       ; 32비트 값을 16비트로 압축(포장)
    punpcklbw xmm3, xmmword ptr [zero16]      ; 8비트 값을 16비트로 zero-extend
    cvtdq2ps xmm3, xmm3                       ; 16비트 정수를 float로 변환
    mulps   xmm3, xmm6                        ; B 가중치(0.114f) 곱셈

    ; R+G+B 합산
    addps   xmm1, xmm2
    addps   xmm1, xmm3

    ; float → int 변환 (각 픽셀별 gray)
    cvtps2dq xmm1, xmm1

    ; 각 gray값을 32비트 픽셀의 R,G,B에 복사
    ; 예: gray | gray | gray | gray (각 32비트)
    ; A 채널은 원본에서 추출해서 OR

    ; gray값을 8비트씩 각 픽셀의 B,G,R에 복사
    ; (SSE로 하려면 shuffle/pack 등 사용, 아니면 C++에서 처리)

    ; 예시: gray값을 32비트로 확장
    movdqa  xmm2, xmm1
    pslld   xmm2, 8
    por     xmm1, xmm2
    movdqa  xmm2, xmm1
    pslld   xmm2, 8
    por     xmm1, xmm2

    ; A 채널 추출 및 OR
    movdqa  xmm2, xmm0
    pand    xmm2, xmmword ptr g_maskA
    por     xmm1, xmm2

    movdqu  xmmword ptr [rdi + rax], xmm1

    add     r8d, 16
    jmp     WidthLoop

NextRow:
    inc     r11d
    jmp     HeightLoop

Done:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbp
    pop     rbx
    ret

ConvertRGBAToBW_SSE_ ENDP
end