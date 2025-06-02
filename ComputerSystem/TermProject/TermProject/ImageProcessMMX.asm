;=======================================================================
; extern "C" void ConvertRGBAToBW_MMX(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
; nWidth는 반드시 2의 배수(2픽셀 단위)라고 가정
;=======================================================================

.data
    maskR   dq 0000000FF000000FFh
    maskG   dq 00000FF000000FF00h
    maskB   dq 000FF000000FF0000h
    maskA   dq 0FF000000FF000000h ; 여기서 맨 앞에 0을 추가!
    weightR dq 0004C004C004C004Ch  ; 76
    weightG dq 00096009600960096h  ; 150
    weightB dq 0001D001D001D001Dh  ; 29

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
HeightLoop:
    xor     r13d, r13d           ; i = 0 (col)
WidthLoop:
    mov     rax, r12
    imul    rax, r10
    add     rax, r13

    movq    mm0, qword ptr [rcx + rax]    ; mm0 = 2픽셀(RGBA RGBA)

    ; R 추출 및 가중치 곱
    movq    mm1, mm0
    pand    mm1, [maskR]
    pmullw  mm1, [weightR]
    

    ; G 추출 및 가중치 곱
    movq    mm2, mm0
    pand    mm2, [maskG]
    psrlw   mm2, 8
    pmullw  mm2, [weightG]

    ; B 추출 및 가중치 곱
    movq    mm3, mm0
    pand    mm3, [maskB]
    psrld   mm3, 16
    pmullw  mm3, [weightB]

    ; B+G+R 합산
    paddd   mm1, mm2
    paddd   mm1, mm3

    ; 8비트로 정규화 (>>8)
    psrld   mm1, 8

    ; A 채널 추출
    movq    mm2, mm0
    pand    mm2, [maskA]

    ; 흑백값을 R,G,B에 복사
    movq    mm3, mm1
    psllw   mm3, 8
    por     mm1, mm3
    movq    mm3, mm1
    pslld   mm3, 8
    por     mm1, mm3

    ; A와 합치기
    por     mm1, mm2

    ; 결과 저장
    movq    qword ptr [rdx + rax], mm1

    add     r13d, 8
    cmp     r13d, r10d
    jl      WidthLoop

    inc     r12d
    cmp     r12d, r11d
    jl      HeightLoop

    emms
    pop     rdi
    pop     rsi
    pop     rbx
    ret
ConvertRGBAToBW_MMX_ ENDP
end