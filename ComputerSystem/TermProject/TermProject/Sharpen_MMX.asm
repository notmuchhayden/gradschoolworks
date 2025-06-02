;=======================================================================
; extern "C" void Sharpen_MMX(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
;=======================================================================

.data
    NineVec dq 0009000900090009h    ; 4ä�� ��� 9�� ���ϱ� ���� ����ũ

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

    mov     rsi, rcx                ; rsi = bufIn (�Է� ���� �ּ�)
    mov     rdi, rdx                ; rdi = bufOut (��� ���� �ּ�)
    mov     r12d, r8d               ; r12d = nWidth
    imul    r12d, 4                 ; r12d = nImageWidth4 (�ȼ��� 4����Ʈ)
    mov     r13d, r9d               ; r13d = nHeight

    mov     r14d, 1                 ; j = 1 (�� ��° ����� ����)
HeightLoop:
    cmp     r14d, r13d              ; j >= nHeight ?
    jge     end_func                ; ������ ����� ó�������� ����
    mov     r15d, 4                 ; i = 4 (�� ��° �ȼ����� ����)
WidthLoop:
    cmp     r15d, r12d              ; i >= nImageWidth4 ?
    jge     next_row                ; �� �� ������ ���� ������

    ; center = i + j * nImageWidth4
    mov     rax, r14                ; rax = j
    imul    rax, r12                ; rax = j * nImageWidth4
    add     rax, r15                ; rax = j * nImageWidth4 + i (���� �ȼ� ���� �ε���)

    pxor    mm0, mm0                ; mm0 = 0, �ֺ��� ��(����) �ʱ�ȭ

    ; �ֺ� 8�ȼ� �ջ�
    ; top left
    mov     rcx, rax
    sub     rcx, r12                ; �� �� ����
    sub     rcx, 4                  ; �� �ȼ� ����
    movq    mm1, qword ptr [rsi + rcx] ; 4ä�� �� �ε�
    paddw   mm0, mm1                ; mm0 += mm1

    ; top
    mov     rcx, rax
    sub     rcx, r12                ; �� �� ����
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; top right
    mov     rcx, rax
    sub     rcx, r12
    add     rcx, 4                  ; �� �ȼ� ������
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; left
    mov     rcx, rax
    sub     rcx, 4                  ; �� �ȼ� ����
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; right
    mov     rcx, rax
    add     rcx, 4                  ; �� �ȼ� ������
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; bottom left
    mov     rcx, rax
    add     rcx, r12                ; �� �� �Ʒ���
    sub     rcx, 4                  ; �� �ȼ� ����
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; bottom
    mov     rcx, rax
    add     rcx, r12                ; �� �� �Ʒ���
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; bottom right
    mov     rcx, rax
    add     rcx, r12
    add     rcx, 4                  ; �� �ȼ� ������
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; �߾Ӱ� * 9
    movq    mm2, qword ptr [rsi + rax]   ; �߾� �ȼ� 4ä�� �� �ε�
    movq    mm3, qword ptr [NineVec]     ; 9�� ä���� ����ũ �ε�
    pmullw  mm2, mm3                     ; mm2 = center * 9

    ; �߾�*9 - �ֺ���
    psubw   mm2, mm0                     ; mm2 = (center*9) - �ֺ���

    ; 0~255 ���� ���� (saturate)
    packuswb mm2, mm2                    ; 8��Ʈ�� ����(0~255�� Ŭ����)

    ; ��� ���� (4ä��)
    movd    dword ptr [rdi + rax], mm2   ; ����� ��� ���ۿ� ����

    add     r15d, 4                      ; ���� �ȼ�(4����Ʈ �̵�)
    jmp     WidthLoop

next_row:
    inc     r14d                         ; ���� ��
    jmp     HeightLoop

end_func:
    emms                                 ; MMX ���� �ʱ�ȭ
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    ret
Sharpen_MMX_ ENDP
end