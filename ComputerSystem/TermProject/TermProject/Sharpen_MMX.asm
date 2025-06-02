;=======================================================================
; extern "C" void Sharpen_MMX(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
;=======================================================================

.code
Sharpen_MMX_ PROC
    ; Windows x64 calling convention
    ; rcx: bufIn
    ; rdx: bufOut
    ; r8d: nWidth
    ; r9d: nHeight

    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rsi, rcx            ; rsi = bufIn
    mov     rdi, rdx            ; rdi = bufOut
    mov     r12d, r8d
    imul    r12d, 4             ; r12d = nImageWidth4 (int)
    mov     r13d, r9d           ; r13d = nHeight

    mov     r14d, 1             ; j = 1
row_loop:
    cmp     r14d, r13d
    jge     end_func            ; if (j >= nHeight - 1) break
    cmp     r14d, r13d
    je      end_func
    mov     r15d, 4             ; i = 4
col_loop:
    cmp     r15d, r12d
    jge     next_row            ; if (i >= nImageWidth4) break

    mov     rbx, 0              ; channel = 0 (R)
channel_loop:
    cmp     rbx, 3
    jge     next_pixel

    ; center = i + channel + j * nImageWidth4
    mov     rax, r14
    imul    rax, r12
    add     rax, r15
    add     rax, rbx            ; rax = center index (64bit)

    pxor    mm0, mm0            ; total = 0

    ; kernel: -1, -1, -1, -1, 9, -1, -1, -1, -1
    ; top left
    mov     rcx, rax
    sub     rcx, r12
    sub     rcx, 4
    movzx   rdx, byte ptr [rsi + rcx]
    movd    mm1, edx
    psubw   mm0, mm1
    ; top
    mov     rcx, rax
    sub     rcx, r12
    movzx   rdx, byte ptr [rsi + rcx]
    movd    mm1, edx
    psubw   mm0, mm1
    ; top right
    mov     rcx, rax
    sub     rcx, r12
    add     rcx, 4
    movzx   rdx, byte ptr [rsi + rcx]
    movd    mm1, edx
    psubw   mm0, mm1
    ; left
    mov     rcx, rax
    sub     rcx, 4
    movzx   rdx, byte ptr [rsi + rcx]
    movd    mm1, edx
    psubw   mm0, mm1
    ; center (kernel 9)
    movzx   rdx, byte ptr [rsi + rax]
    imul    edx, 9
    movd    mm1, edx
    paddw   mm0, mm1
    ; right
    mov     rcx, rax
    add     rcx, 4
    movzx   rdx, byte ptr [rsi + rcx]
    movd    mm1, edx
    psubw   mm0, mm1
    ; bottom left
    mov     rcx, rax
    add     rcx, r12
    sub     rcx, 4
    movzx   rdx, byte ptr [rsi + rcx]
    movd    mm1, edx
    psubw   mm0, mm1
    ; bottom
    mov     rcx, rax
    add     rcx, r12
    movzx   rdx, byte ptr [rsi + rcx]
    movd    mm1, edx
    psubw   mm0, mm1
    ; bottom right
    mov     rcx, rax
    add     rcx, r12
    add     rcx, 4
    movzx   rdx, byte ptr [rsi + rcx]
    movd    mm1, edx
    psubw   mm0, mm1

    ; total 값 범위 제한 (0~255)
    movd    edx, mm0
    cmp     edx, 0
    jge     no_min
    mov     edx, 0
no_min:
    cmp     edx, 255
    jle     no_max
    mov     edx, 255
no_max:
    mov     byte ptr [rdi + rax], dl

    inc     rbx
    jmp     channel_loop

next_pixel:
    add     r15d, 4
    jmp     col_loop

next_row:
    inc     r14d
    jmp     row_loop

end_func:
    emms
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    ret
Sharpen_MMX_ ENDP
end