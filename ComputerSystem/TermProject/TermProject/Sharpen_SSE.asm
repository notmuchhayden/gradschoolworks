.data
    NineVec dq 0009000900090009h, 0009000900090009h ; 16바이트(128비트)로 선언

.code
Sharpen_SSE_ PROC
    push    rsi
    push    rdi
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rsi, rcx
    mov     rdi, rdx
    mov     r12d, r8d
    imul    r12d, 4
    mov     r13d, r9d

    mov     r14d, 1
HeightLoop:
    cmp     r14d, r13d
    jge     end_func

    mov     r15d, 4
WidthLoop:
    mov     eax, r12d
    sub     eax, 8
    cmp     r15d, eax
    jg      next_row

    mov     eax, r14d
    imul    eax, r12d
    add     eax, r15d

    ; 9개 이웃 픽셀 8바이트씩 로드
    mov     ebx, eax
    sub     ebx, r12d
    sub     ebx, 4
    movq    xmm0, qword ptr [rsi + rbx]

    mov     ebx, eax
    sub     ebx, r12d
    movq    xmm1, qword ptr [rsi + rbx]

    mov     ebx, eax
    sub     ebx, r12d
    add     ebx, 4
    movq    xmm2, qword ptr [rsi + rbx]

    mov     ebx, eax
    sub     ebx, 4
    movq    xmm3, qword ptr [rsi + rbx]

    movq    xmm4, qword ptr [rsi + rax]

    mov     ebx, eax
    add     ebx, 4
    movq    xmm5, qword ptr [rsi + rbx]

    mov     ebx, eax
    add     ebx, r12d
    sub     ebx, 4
    movq    xmm6, qword ptr [rsi + rbx]

    mov     ebx, eax
    add     ebx, r12d
    movq    xmm7, qword ptr [rsi + rbx]

    mov     ebx, eax
    add     ebx, r12d
    add     ebx, 4
    movq    xmm8, qword ptr [rsi + rbx]

    ; zero-extend
    pxor    xmm9, xmm9

    punpcklbw xmm0, xmm9
    punpcklbw xmm1, xmm9
    punpcklbw xmm2, xmm9
    punpcklbw xmm3, xmm9
    punpcklbw xmm4, xmm9
    punpcklbw xmm5, xmm9
    punpcklbw xmm6, xmm9
    punpcklbw xmm7, xmm9
    punpcklbw xmm8, xmm9

    ; 샤프닝 커널 적용
    movdqa  xmm10, xmm4
    movdqa  xmm12, oword ptr [NineVec]
    pmullw  xmm10, xmm12

    movdqa  xmm11, xmm0
    paddw   xmm11, xmm1
    paddw   xmm11, xmm2
    paddw   xmm11, xmm3
    paddw   xmm11, xmm5
    paddw   xmm11, xmm6
    paddw   xmm11, xmm7
    paddw   xmm11, xmm8

    psubw   xmm10, xmm11

    ; 0~255로 클램핑
    packuswb xmm10, xmm9

    ; 결과 저장 (2픽셀, 8바이트)
    movq    qword ptr [rdi + rax], xmm10

    add     r15d, 8
    jmp     WidthLoop

next_row:
    inc     r14d
    jmp     HeightLoop

end_func:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rdi
    pop     rsi
    ret

Sharpen_SSE_ ENDP
end