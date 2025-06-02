;=======================================================================
; extern "C" void Sharpen_SSE(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
;=======================================================================

; 16비트 short 8개짜리 9 상수 벡터
.data
    NineVec dq 0009000900090009h, 0009000900090009h

.code
Sharpen_SSE_ PROC
    ; rcx: bufIn
    ; rdx: bufOut
    ; r8d: nWidth
    ; r9d: nHeight

    push    rsi
    push    rdi
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rsi, rcx            ; rsi = bufIn
    mov     rdi, rdx            ; rdi = bufOut
    mov     r12d, r8d
    imul    r12d, 4             ; r12d = nImageWidth4
    mov     r13d, r9d           ; r13d = nHeight

    mov     r14d, 1             ; j = 1
row_loop:
    cmp     r14d, r13d
    jge     end_func            ; if (j >= nHeight - 1) break

    mov     r15d, 4             ; i = 4
col_loop:
    mov     eax, r12d
    sub     eax, 16
    cmp     r15d, eax
    jg      next_row            ; if (i > nImageWidth4 - 16) break

    ; center = i + j * nImageWidth4
    mov     eax, r14d
    imul    eax, r12d
    add     eax, r15d           ; eax = center offset

    ; SSE로 4픽셀(16바이트)씩 처리
    ; 9개 이웃 픽셀을 각각 xmm 레지스터에 로드
    mov     ebx, eax
    sub     ebx, r12d
    sub     ebx, 4
    movdqu  xmm0, [rsi + rbx]   ; top left

    mov     ebx, eax
    sub     ebx, r12d
    movdqu  xmm1, [rsi + rbx]   ; top

    mov     ebx, eax
    sub     ebx, r12d
    add     ebx, 4
    movdqu  xmm2, [rsi + rbx]   ; top right

    mov     ebx, eax
    sub     ebx, 4
    movdqu  xmm3, [rsi + rbx]   ; left

    movdqu  xmm4, [rsi + rax]   ; center

    mov     ebx, eax
    add     ebx, 4
    movdqu  xmm5, [rsi + rbx]   ; right

    mov     ebx, eax
    add     ebx, r12d
    sub     ebx, 4
    movdqu  xmm6, [rsi + rbx]   ; bottom left

    mov     ebx, eax
    add     ebx, r12d
    movdqu  xmm7, [rsi + rbx]   ; bottom

    mov     ebx, eax
    add     ebx, r12d
    add     ebx, 4
    movdqu  xmm8, [rsi + rbx]   ; bottom right

    ; unsigned char -> unsigned short 변환 (zero-extend)
    pxor    xmm9, xmm9
    movdqa  xmm10, xmm0
    movdqa  xmm11, xmm1
    movdqa  xmm12, xmm2
    movdqa  xmm13, xmm3
    movdqa  xmm14, xmm4
    movdqa  xmm15, xmm5

    punpcklbw xmm0, xmm9
    punpcklbw xmm1, xmm9
    punpcklbw xmm2, xmm9
    punpcklbw xmm3, xmm9
    punpcklbw xmm4, xmm9
    punpcklbw xmm5, xmm9
    punpcklbw xmm6, xmm9
    punpcklbw xmm7, xmm9
    punpcklbw xmm8, xmm9

    ; 샤프닝 커널 적용: -1, -1, -1, -1, 9, -1, -1, -1, -1
    ; total = 9*center - (topL+top+topR+left+right+botL+bot+botR)
    movdqa  xmm10, xmm4         ; xmm10 = center
    movdqa  xmm12, oword ptr [NineVec] ; NineVec를 xmm12에 로드
    pmullw  xmm10, xmm12               ; 9*center

    movdqa  xmm11, xmm0
    paddw   xmm11, xmm1
    paddw   xmm11, xmm2
    paddw   xmm11, xmm3
    paddw   xmm11, xmm5
    paddw   xmm11, xmm6
    paddw   xmm11, xmm7
    paddw   xmm11, xmm8         ; sum of 8 neighbors

    psubw   xmm10, xmm11        ; total = 9*center - sum(neighbors)

    ; 0~255로 클램핑
    packuswb xmm10, xmm9

    ; 결과 저장
    movdqu  [rdi + rax], xmm10

    add     r15d, 16            ; 4픽셀(16바이트)씩 이동
    jmp     col_loop

next_row:
    inc     r14d
    jmp     row_loop

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