;=======================================================================
; extern "C" void ConvertRGBAToBW_MMX(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
; nWidth는 반드시 2의 배수(2픽셀 단위)라고 가정
;=======================================================================

.data
    maskB   dq 000000FF000000FFh
    maskG   dq 0000FF000000FF00h
    maskR   dq 00FF000000FF0000h
    maskA   dq 0FF000000FF000000h ; 여기서 맨 앞에 0을 추가!
    weightB dq 001D001D001D001Dh  ; 29
    weightG dq 0096009600960096h  ; 150
    weightR dq 004C004C004C004Ch  ; 76

.code
ConvertRGBAToBW_MMX_ PROC
    push    rbx
    push    rsi
    push    rdi

    mov     r10d, r8d            ; nWidth
    imul    r10d, 4              ; nWidth4 = nWidth * 4
    mov     r11d, r9d            ; nHeight

    pxor    mm7, mm7             ; mm7 = 0 (zero register)

    xor     r12d, r12d           ; j = 0 (row)
row_loop:
    xor     r13d, r13d           ; i = 0 (col)
col_loop:
    mov     rax, r12
    imul    rax, r10
    add     rax, r13

    movq    mm0, qword ptr [rcx + rax]    ; mm0 = 2픽셀(RGBA RGBA)

    ; B 추출 및 가중치 곱
    movq    mm1, mm0
    pand    mm1, [maskB]
    pmullw  mm1, [weightB]

    ; G 추출 및 가중치 곱
    movq    mm2, mm0
    pand    mm2, [maskG]
    psrlw   mm2, 8
    pmullw  mm2, [weightG]

    ; R 추출 및 가중치 곱
    movq    mm3, mm0
    pand    mm3, [maskR]
    psrlw   mm3, 16
    pmullw  mm3, [weightR]

    ; B+G+R 합산
    paddw   mm1, mm2
    paddw   mm1, mm3

    ; 8비트로 정규화 (>>8)
    psrlw   mm1, 8

    ; A 채널 추출
    movq    mm2, mm0
    pand    mm2, [maskA]

    ; 흑백값을 R,G,B에 복사
    movq    mm3, mm1
    psllw   mm3, 8
    por     mm1, mm3
    movq    mm3, mm1
    psllw   mm3, 8
    por     mm1, mm3

    ; A와 합치기
    por     mm1, mm2

    ; 결과 저장
    movq    qword ptr [rdx + rax], mm1

    add     r13d, 8
    cmp     r13d, r10d
    jl      col_loop

    inc     r12d
    cmp     r12d, r11d
    jl      row_loop

    emms
    pop     rdi
    pop     rsi
    pop     rbx
    ret
ConvertRGBAToBW_MMX_ ENDP
end