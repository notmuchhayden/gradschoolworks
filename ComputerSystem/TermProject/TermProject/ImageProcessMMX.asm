;=======================================================================
; extern "C" void ConvertRGBAToBW_MMX(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
;=======================================================================

.data
    maskB   dq 000000FF000000FFh
    maskG   dq 0000FF000000FF00h
    maskR   dq 00FF000000FF0000h
    maskA   dq 0FF000000FF000000h   ; �� ���⼭ 0�� �߰�!
    weightB dq 001D001D001D001Dh  ; 29
    weightG dq 0096009600960096h  ; 150
    weightR dq 004C004C004C004Ch  ; 76

.code
ConvertRGBAToBW_MMX_ PROC
    push    rbx
    mov     r10d, r8d            ; nWidth
    imul    r10d, 4              ; nWidth4 = nWidth * 4
    mov     r11d, r9d            ; nHeight

    xor     r12d, r12d           ; j = 0

row_loop:
    xor     r13d, r13d           ; i = 0

col_loop:
    mov     rax, r12
    imul    rax, r10            ; j * nWidth4
    add     rax, r13            ; i + j * nWidth4

    ; 2�ȼ��� ó��
    movq    mm0, qword ptr [rcx + rax]    ; mm0 = 2�ȼ�(RGBA RGBA)
    movq    mm1, mm0

    ; ù ��° �ȼ�
    movd    eax, mm0             ; eax = [B0 G0 R0 A0]
    mov     ebx, eax
    and     ebx, 0FFh            ; B
    imul    ebx, 29
    shr     eax, 8
    mov     ecx, eax
    and     ecx, 0FFh            ; G
    imul    ecx, 150
    shr     eax, 8
    mov     edx, eax
    and     edx, 0FFh            ; R
    imul    edx, 77
    add     ebx, ecx
    add     ebx, edx
    shr     ebx, 8

    ; �� ��° �ȼ�
    psrlq   mm1, 32              ; mm1 = 2��° �ȼ�
    movd    eax, mm1
    mov     esi, eax
    and     esi, 0FFh            ; B
    imul    esi, 29
    shr     eax, 8
    mov     edi, eax
    and     edi, 0FFh            ; G
    imul    edi, 150
    shr     eax, 8
    mov     ebp, eax
    and     ebp, 0FFh            ; R
    imul    ebp, 77
    add     esi, edi
    add     esi, ebp
    shr     esi, 8

    ; ��� ���� (��鰪�� R,G,B��, A�� ���� ����)
    ; ù ��° �ȼ�
    movd    eax, mm0
    and     eax, 0FF000000h      ; A
    or      eax, ebx
    shl     ebx, 8
    or      eax, ebx
    shl     ebx, 8
    or      eax, ebx
    mov     ebx, eax             ; ebx = ù ��° �ȼ� ���

    ; �� ��° �ȼ�
    movd    eax, mm1
    and     eax, 0FF000000h      ; A
    or      eax, esi
    shl     esi, 8
    or      eax, esi
    shl     esi, 8
    or      eax, esi
    mov     ecx, eax             ; ecx = �� ��° �ȼ� ���

    ; 8����Ʈ�� ����
    mov     dword ptr [rdx + rax], ebx
    mov     dword ptr [rdx + rax + 4], ecx

    add     r13d, 8
    cmp     r13d, r10d
    jl      col_loop

    inc     r12d
    cmp     r12d, r11d
    jl      row_loop

    emms
    pop     rbx
    ret
ConvertRGBAToBW_MMX_ ENDP
end