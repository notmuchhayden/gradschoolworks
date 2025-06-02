;=======================================================================
; extern "C" void Sharpen_MMX(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
;=======================================================================

.data
    NineVec dq 0009000900090009h    ; 4ä��(���� 16��Ʈ ����) ��� 9�� ���ϱ� ���� ����ũ

.code
Sharpen_MMX_ PROC
    ; Windows x64 ȣ�� �Ծ�
    ; rcx: bufIn (�Է� ����)
    ; rdx: bufOut (��� ����)
    ; r8d: nWidth (�̹��� �ʺ�)
    ; r9d: nHeight (�̹��� ����)

    push    rsi                     ; rsi �������� ����
    push    rdi                     ; rdi �������� ����
    push    r12                     ; r12 �������� ����
    push    r13                     ; r13 �������� ����
    push    r14                     ; r14 �������� ����
    push    r15                     ; r15 �������� ����
    push    rbx                     ; rbx �������� ���� (mm7 ��� ����� �� ������, MMX �������� ����� �Ϲ���)

    pxor    mm7, mm7                ; mm7�� 0���� �ʱ�ȭ (����ŷ�� ���)

    mov     rsi, rcx                ; rsi = bufIn (�Է� ���� �ּ�)
    mov     rdi, rdx                ; rdi = bufOut (��� ���� �ּ�)
    mov     r12d, r8d               ; r12d = nWidth
    imul    r12d, 4                 ; r12d = nImageWidth4 (�ȼ��� 4����Ʈ)
    mov     r13d, r9d               ; r13d = nHeight

    mov     r14d, 1                 ; j = 1 (�� ��° ����� ����)
HeightLoop:
    cmp     r14d, r13d              ; j >= nHeight -1 ? (�����δ� nHeight���� ���� X)
    jge     end_func                ; ������ ����� ó�������� ���� (j�� nHeight-1�� ��������)
    mov     r15d, 4                 ; i = 4 (�� ��° �ȼ����� ����)
WidthLoop:
    cmp     r15d, r12d              ; i >= nImageWidth4 - 4 ? (�����δ� nImageWidth4-4���� ���� X)
    jge     next_row                ; �� �� ������ ���� ������ (i�� nImageWidth4-4�� ��������)

    ; center = i + j * nImageWidth4
    mov     rax, r14                ; rax = j
    imul    rax, r12                ; rax = j * nImageWidth4
    add     rax, r15                ; rax = j * nImageWidth4 + i (���� �ȼ� ���� �ε���)

    pxor    mm0, mm0                ; mm0 = 0, �ֺ��� ��(16��Ʈ ���� ����) �ʱ�ȭ

    ; �ֺ� 8�ȼ� �ջ� (�� �ȼ��� 4����Ʈ, �ε� �� 16��Ʈ ����� ����ŷ)
    ; top left
    mov     rcx, rax
    sub     rcx, r12                ; �� �� ����
    sub     rcx, 4                  ; �� �ȼ� ����
    movd    mm1, dword ptr [rsi + rcx] ; 4����Ʈ(1�ȼ�) �ε�
    punpcklbw mm1, mm7              ; 8��Ʈ ä�� -> 16��Ʈ ����� Ȯ��
    paddw   mm0, mm1                ; mm0 += mm1 (���� ���� ����)

    ; top
    mov     rcx, rax
    sub     rcx, r12                ; �� �� ����
    movd    mm1, dword ptr [rsi + rcx]
    punpcklbw mm1, mm7
    paddw   mm0, mm1

    ; top right
    mov     rcx, rax
    sub     rcx, r12
    add     rcx, 4                  ; �� �ȼ� ������
    movd    mm1, dword ptr [rsi + rcx]
    punpcklbw mm1, mm7
    paddw   mm0, mm1

    ; left
    mov     rcx, rax
    sub     rcx, 4                  ; �� �ȼ� ����
    movd    mm1, dword ptr [rsi + rcx]
    punpcklbw mm1, mm7
    paddw   mm0, mm1

    ; right
    mov     rcx, rax
    add     rcx, 4                  ; �� �ȼ� ������
    movd    mm1, dword ptr [rsi + rcx]
    punpcklbw mm1, mm7
    paddw   mm0, mm1

    ; bottom left
    mov     rcx, rax
    add     rcx, r12                ; �� �� �Ʒ���
    sub     rcx, 4                  ; �� �ȼ� ����
    movd    mm1, dword ptr [rsi + rcx]
    punpcklbw mm1, mm7
    paddw   mm0, mm1

    ; bottom
    mov     rcx, rax
    add     rcx, r12                ; �� �� �Ʒ���
    movd    mm1, dword ptr [rsi + rcx]
    punpcklbw mm1, mm7
    paddw   mm0, mm1

    ; bottom right
    mov     rcx, rax
    add     rcx, r12
    add     rcx, 4                  ; �� �ȼ� ������
    movd    mm1, dword ptr [rsi + rcx]
    punpcklbw mm1, mm7
    paddw   mm0, mm1

    ; �߾Ӱ� * 9
    movd    mm2, dword ptr [rsi + rax]   ; �߾� �ȼ� 4ä�� �� �ε�
    punpcklbw mm2, mm7                   ; 16��Ʈ ����� ����ŷ
    movq    mm3, qword ptr [NineVec]     ; 9�� ä���� ����ũ �ε� (0x0009 0009 0009 0009)
    pmullw  mm2, mm3                     ; mm2 = center_words * 9 (���� ���� ����)

    ; �߾�*9 - �ֺ���
    psubw   mm2, mm0                     ; mm2 = (center_words*9) - �ֺ���_words (���� ���� ����)

    ; 0~255 ���� ���� (saturate)
    packuswb mm2, mm7                    ; mm2�� ���� 4�� ���带 ��ȣ ���� 8��Ʈ ����Ʈ�� ��ȭ ��ȯ�Ͽ� mm2�� ���� 4����Ʈ�� ����

    ; ��� ���� (4ä��)
    movd    dword ptr [rdi + rax], mm2   ; ����� ��� ���ۿ� ���� (mm2�� ���� 32��Ʈ)

    add     r15d, 4                      ; ���� �ȼ�(4����Ʈ �̵�)
    jmp     WidthLoop

next_row:
    inc     r14d                         ; ���� ��
    jmp     HeightLoop

end_func:
    emms                                 ; MMX ���� �ʱ�ȭ
    pop     rbx
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    ret
Sharpen_MMX_ ENDP
end