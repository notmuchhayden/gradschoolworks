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

    movdqu  xmm0, xmmword ptr [rsi + rax] ; 4�ȼ�(RGBA*4) �ε�

    ; R ���� �� ��ȯ
    movdqa  xmm1, xmm0                        ; xmm0(�Է�)���� �����Ͽ� xmm1�� ����
    pand    xmm1, xmmword ptr g_maskR         ; R ä�θ� ����� ������ ��Ʈ 0���� ����ŷ
    psrld   xmm1, 16                          ; R ä���� ���� ����Ʈ�� �̵� (�� �ȼ��� 16��Ʈ ������ ����Ʈ)
    packusdw xmm1, xmm1                       ; 32��Ʈ ���� 16��Ʈ�� ����(����)
    punpcklbw xmm1, xmmword ptr [zero16]      ; 8��Ʈ ���� 16��Ʈ�� zero-extend
    cvtdq2ps xmm1, xmm1                       ; 16��Ʈ ������ float�� ��ȯ
    mulps   xmm1, xmm4                        ; R ����ġ(0.299f) ����

    ; G ���� �� ��ȯ
    movdqa  xmm2, xmm0                        ; xmm0(�Է�)���� �����Ͽ� xmm2�� ����
    pand    xmm2, xmmword ptr g_maskG         ; G ä�θ� ����� ������ ��Ʈ 0���� ����ŷ
    psrld   xmm2, 8                           ; G ä���� ���� ����Ʈ�� �̵� (�� �ȼ��� 8��Ʈ ������ ����Ʈ)
    packusdw xmm2, xmm2                       ; 32��Ʈ ���� 16��Ʈ�� ����(����)
    punpcklbw xmm2, xmmword ptr [zero16]      ; 8��Ʈ ���� 16��Ʈ�� zero-extend
    cvtdq2ps xmm2, xmm2                       ; 16��Ʈ ������ float�� ��ȯ
    mulps   xmm2, xmm5                        ; G ����ġ(0.587f) ����

    ; B ���� �� ��ȯ
    movdqa  xmm3, xmm0                        ; xmm0(�Է�)���� �����Ͽ� xmm3�� ����
    pand    xmm3, xmmword ptr g_maskB         ; B ä�θ� ����� ������ ��Ʈ 0���� ����ŷ
    packusdw xmm3, xmm3                       ; 32��Ʈ ���� 16��Ʈ�� ����(����)
    punpcklbw xmm3, xmmword ptr [zero16]      ; 8��Ʈ ���� 16��Ʈ�� zero-extend
    cvtdq2ps xmm3, xmm3                       ; 16��Ʈ ������ float�� ��ȯ
    mulps   xmm3, xmm6                        ; B ����ġ(0.114f) ����

    ; R+G+B �ջ�
    addps   xmm1, xmm2
    addps   xmm1, xmm3

    ; float �� int ��ȯ (�� �ȼ��� gray)
    cvtps2dq xmm1, xmm1

    ; �� gray���� 32��Ʈ �ȼ��� R,G,B�� ����
    ; ��: gray | gray | gray | gray (�� 32��Ʈ)
    ; A ä���� �������� �����ؼ� OR

    ; gray���� 8��Ʈ�� �� �ȼ��� B,G,R�� ����
    ; (SSE�� �Ϸ��� shuffle/pack �� ���, �ƴϸ� C++���� ó��)

    ; ����: gray���� 32��Ʈ�� Ȯ��
    movdqa  xmm2, xmm1
    pslld   xmm2, 8
    por     xmm1, xmm2
    movdqa  xmm2, xmm1
    pslld   xmm2, 8
    por     xmm1, xmm2

    ; A ä�� ���� �� OR
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