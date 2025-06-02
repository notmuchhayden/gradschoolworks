;=======================================================================
; extern "C" void Sharpen_MMX(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
;=======================================================================

.data
    NineVec dq 0009000900090009h    ; 4채널 모두 9로 곱하기 위한 마스크

.code
Sharpen_MMX_ PROC
    ; Windows x64 호출 규약
    ; rcx: bufIn (입력 버퍼)
    ; rdx: bufOut (출력 버퍼)
    ; r8d: nWidth (이미지 너비)
    ; r9d: nHeight (이미지 높이)

    push    rsi                     ; rsi 레지스터 보존
    push    rdi                     ; rdi 레지스터 보존
    push    r12                     ; r12 레지스터 보존
    push    r13                     ; r13 레지스터 보존
    push    r14                     ; r14 레지스터 보존
    push    r15                     ; r15 레지스터 보존

    mov     rsi, rcx                ; rsi = bufIn (입력 버퍼 주소)
    mov     rdi, rdx                ; rdi = bufOut (출력 버퍼 주소)
    mov     r12d, r8d               ; r12d = nWidth
    imul    r12d, 4                 ; r12d = nImageWidth4 (픽셀당 4바이트)
    mov     r13d, r9d               ; r13d = nHeight

    mov     r14d, 1                 ; j = 1 (두 번째 행부터 시작)
HeightLoop:
    cmp     r14d, r13d              ; j >= nHeight ?
    jge     end_func                ; 마지막 행까지 처리했으면 종료
    mov     r15d, 4                 ; i = 4 (두 번째 픽셀부터 시작)
WidthLoop:
    cmp     r15d, r12d              ; i >= nImageWidth4 ?
    jge     next_row                ; 한 행 끝나면 다음 행으로

    ; center = i + j * nImageWidth4
    mov     rax, r14                ; rax = j
    imul    rax, r12                ; rax = j * nImageWidth4
    add     rax, r15                ; rax = j * nImageWidth4 + i (현재 픽셀 시작 인덱스)

    pxor    mm0, mm0                ; mm0 = 0, 주변값 합(누적) 초기화

    ; 주변 8픽셀 합산
    ; top left
    mov     rcx, rax
    sub     rcx, r12                ; 한 행 위로
    sub     rcx, 4                  ; 한 픽셀 왼쪽
    movq    mm1, qword ptr [rsi + rcx] ; 4채널 값 로드
    paddw   mm0, mm1                ; mm0 += mm1

    ; top
    mov     rcx, rax
    sub     rcx, r12                ; 한 행 위로
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; top right
    mov     rcx, rax
    sub     rcx, r12
    add     rcx, 4                  ; 한 픽셀 오른쪽
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; left
    mov     rcx, rax
    sub     rcx, 4                  ; 한 픽셀 왼쪽
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; right
    mov     rcx, rax
    add     rcx, 4                  ; 한 픽셀 오른쪽
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; bottom left
    mov     rcx, rax
    add     rcx, r12                ; 한 행 아래로
    sub     rcx, 4                  ; 한 픽셀 왼쪽
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; bottom
    mov     rcx, rax
    add     rcx, r12                ; 한 행 아래로
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; bottom right
    mov     rcx, rax
    add     rcx, r12
    add     rcx, 4                  ; 한 픽셀 오른쪽
    movq    mm1, qword ptr [rsi + rcx]
    paddw   mm0, mm1

    ; 중앙값 * 9
    movq    mm2, qword ptr [rsi + rax]   ; 중앙 픽셀 4채널 값 로드
    movq    mm3, qword ptr [NineVec]     ; 9로 채워진 마스크 로드
    pmullw  mm2, mm3                     ; mm2 = center * 9

    ; 중앙*9 - 주변합
    psubw   mm2, mm0                     ; mm2 = (center*9) - 주변합

    ; 0~255 범위 제한 (saturate)
    packuswb mm2, mm2                    ; 8비트로 포장(0~255로 클램프)

    ; 결과 저장 (4채널)
    movd    dword ptr [rdi + rax], mm2   ; 결과를 출력 버퍼에 저장

    add     r15d, 4                      ; 다음 픽셀(4바이트 이동)
    jmp     WidthLoop

next_row:
    inc     r14d                         ; 다음 행
    jmp     HeightLoop

end_func:
    emms                                 ; MMX 상태 초기화
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    ret
Sharpen_MMX_ ENDP
end