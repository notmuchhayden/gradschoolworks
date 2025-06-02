// TermProject.cpp : 애플리케이션에 대한 진입점을 정의합니다.
//

#include "framework.h"
#include "TermProject.h"
#include "ImageProcess.h" // 상단에 추가
#include <commdlg.h> // 파일 열기 대화상자 사용
#include <algorithm>
#include <chrono> // 상단에 추가
#include <fstream> // BMP 파일 직접 읽기
#include <vector>

#define MAX_LOADSTRING 100
#define IDC_BTN_BW_MMX 2001 // 버튼 ID 추가
#define IDC_BTN_BW_SSE 2002 // SSE 버튼 ID 추가

// 전역 변수:
HINSTANCE hInst;                                // 현재 인스턴스입니다.
WCHAR szTitle[MAX_LOADSTRING];                  // 제목 표시줄 텍스트입니다.
WCHAR szWindowClass[MAX_LOADSTRING];            // 기본 창 클래스 이름입니다.
HBITMAP hBitmap = NULL;                         // 전역 비트맵 핸들
HBITMAP hBitmapBW = NULL;                       // 흑백 비트맵 핸들 추가
HBITMAP hBitmapBW_MMX = NULL;                   // MMX 변환 비트맵 핸들 추가
HBITMAP hBitmapBW_SSE = NULL;                   // SSE 변환 비트맵 핸들 추가
WCHAR szBmpFile[MAX_PATH] = L"";                // 선택된 파일 경로
HWND hBtnBW = NULL;                             // 흑백변환 버튼 핸들
HWND hBtnBW_MMX = NULL;                         // MMX 흑백변환 버튼 핸들 추가
HWND hBtnBW_SSE = NULL;                         // SSE 버튼 핸들 추가
WCHAR g_bwTimeStr[128] = L"";                   // 흑백 변환 시간 문자열

int xScrollPos = 0, yScrollPos = 0;             // 스크롤 위치
int xMaxScroll = 0, yMaxScroll = 0;             // 스크롤 최대값

// 이 코드 모듈에 포함된 함수의 선언을 전달합니다:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

// 비트맵 해제 함수
void ReleaseBitmap() {
    if (hBitmap) {
        DeleteObject(hBitmap);
        hBitmap = NULL;
    }
    if (hBitmapBW) {
        DeleteObject(hBitmapBW);
        hBitmapBW = NULL;
    }
    if (hBitmapBW_MMX) {
        DeleteObject(hBitmapBW_MMX);
        hBitmapBW_MMX = NULL;
    }
    if (hBitmapBW_SSE) {
        DeleteObject(hBitmapBW_SSE);
        hBitmapBW_SSE = NULL;
    }
    ShowWindow(hBtnBW, SW_HIDE); // 버튼 상태 초기화
    ShowWindow(hBtnBW_MMX, SW_HIDE); // MMX 버튼 상태 초기화
    ShowWindow(hBtnBW_SSE, SW_HIDE); // SSE 버튼 상태 초기화
}

// 컬러 비트맵을 흑백 비트맵으로 변환하는 함수 (32비트 기준)
HBITMAP ConvertColorToBW(HBITMAP hSrcBmp) {
    if (!hSrcBmp) return NULL;

    BITMAP bmp;
    GetObject(hSrcBmp, sizeof(BITMAP), &bmp);

    int width = bmp.bmWidth;
    int height = bmp.bmHeight;
    int bytesPerPixel = 4; // 32비트
    int stride = ((width * bytesPerPixel + 3) & ~3);

    // 입력 버퍼 준비
    unsigned char* bufIn = new unsigned char[stride * height];
    unsigned char* bufOut = new unsigned char[stride * height];

    // 원본 비트맵에서 픽셀 데이터 추출
    BITMAPINFO bmi = { 0 };
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; // 상단이 원점
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32; // 32비트로 변경
    bmi.bmiHeader.biCompression = BI_RGB;

    HDC hdc = GetDC(NULL);
    GetDIBits(hdc, hSrcBmp, 0, height, bufIn, &bmi, DIB_RGB_COLORS);
    ReleaseDC(NULL, hdc);

    // 시간 측정 및 1000번 반복
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        ConvertRGBAToBW(bufIn, bufOut, width, height); // RGBA용 함수여야 함
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // 시간 문자열 저장
    swprintf(g_bwTimeStr, 128, L"흑백 변환 1000회 실행 시간: %.2f ms", elapsed);

    // 결과로 DIBSection 생성
    void* pBits = nullptr;
    HBITMAP hDstBmp = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, &pBits, NULL, 0);
    if (hDstBmp && pBits) {
        memcpy(pBits, bufOut, stride * height);
    }

    delete[] bufIn;
    delete[] bufOut;

    return hDstBmp;
}

// MMX 버전도 동일하게 32비트로 변경
HBITMAP ConvertColorToBW_MMX(HBITMAP hSrcBmp) {
    if (!hSrcBmp) return NULL;

    BITMAP bmp;
    GetObject(hSrcBmp, sizeof(BITMAP), &bmp);

    int width = bmp.bmWidth;
    int height = bmp.bmHeight;
    int bytesPerPixel = 4; // 32비트
    int stride = ((width * bytesPerPixel + 3) & ~3);

    // 입력 버퍼 준비
    unsigned char* bufIn = new unsigned char[stride * height];
    unsigned char* bufOut = new unsigned char[stride * height];

    // 원본 비트맵에서 픽셀 데이터 추출
    BITMAPINFO bmi = { 0 };
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; // 상단이 원점
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32; // 32비트로 변경
    bmi.bmiHeader.biCompression = BI_RGB;

    HDC hdc = GetDC(NULL);
    GetDIBits(hdc, hSrcBmp, 0, height, bufIn, &bmi, DIB_RGB_COLORS);
    ReleaseDC(NULL, hdc);

    // 시간 측정 및 1000번 반복
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        ConvertRGBAToBW_MMX_(bufIn, bufOut, width, height); // RGBA용 함수여야 함
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // 시간 문자열 저장
    swprintf(g_bwTimeStr, 128, L"흑백 변환 1000회 실행 시간: %.2f ms", elapsed);

    // 결과로 DIBSection 생성
    void* pBits = nullptr;
    HBITMAP hDstBmp = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, &pBits, NULL, 0);
    if (hDstBmp && pBits) {
        memcpy(pBits, bufOut, stride * height);
    }

    delete[] bufIn;
    delete[] bufOut;

    return hDstBmp;
}

// MMX 버전도 동일하게 32비트로 변경
HBITMAP ConvertColorToBW_SSE(HBITMAP hSrcBmp) {
    if (!hSrcBmp) return NULL;

    BITMAP bmp;
    GetObject(hSrcBmp, sizeof(BITMAP), &bmp);

    int width = bmp.bmWidth;
    int height = bmp.bmHeight;
    int bytesPerPixel = 4; // 32비트
    int stride = ((width * bytesPerPixel + 3) & ~3);

    // 입력 버퍼 준비
    unsigned char* bufIn = new unsigned char[stride * height];
    unsigned char* bufOut = new unsigned char[stride * height];

    // 원본 비트맵에서 픽셀 데이터 추출
    BITMAPINFO bmi = { 0 };
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; // 상단이 원점
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32; // 32비트로 변경
    bmi.bmiHeader.biCompression = BI_RGB;

    HDC hdc = GetDC(NULL);
    GetDIBits(hdc, hSrcBmp, 0, height, bufIn, &bmi, DIB_RGB_COLORS);
    ReleaseDC(NULL, hdc);

    // 시간 측정 및 1000번 반복
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        ConvertRGBAToBW_SSE_(bufIn, bufOut, width, height); // RGBA용 함수여야 함
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // 시간 문자열 저장
    swprintf(g_bwTimeStr, 128, L"흑백 변환 1000회 실행 시간: %.2f ms", elapsed);

    // 결과로 DIBSection 생성
    void* pBits = nullptr;
    HBITMAP hDstBmp = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, &pBits, NULL, 0);
    if (hDstBmp && pBits) {
        memcpy(pBits, bufOut, stride * height);
    }

    delete[] bufIn;
    delete[] bufOut;

    return hDstBmp;
}

// 32비트 BMP 파일을 직접 읽어서 HBITMAP으로 반환
HBITMAP LoadBitmap32FromFile(const wchar_t* filename, int* outWidth = nullptr, int* outHeight = nullptr)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) return NULL;

    BITMAPFILEHEADER header;
    BITMAPINFOHEADER info;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    file.read(reinterpret_cast<char*>(&info), sizeof(info));

    if (header.bfType != 0x4D42 || info.biBitCount != 32 /*|| info.biCompression != 0*/) {
        // BMP magic number, 32bpp, BI_RGB(0)만 지원
        return NULL;
    }

    int width = info.biWidth;
    int height = abs(info.biHeight);
    int stride = ((width * 4 + 3) & ~3);

    // 파일에서 픽셀 데이터 읽기
    std::vector<unsigned char> buf(stride * height);
    file.seekg(header.bfOffBits, std::ios::beg);
    file.read(reinterpret_cast<char*>(buf.data()), stride * height);

    // BMP는 하단이 원점이므로, biHeight < 0 이 아니면 상하 반전 필요
    bool flip = (info.biHeight > 0);

    // DIBSection 생성
    BITMAPINFO bmi = { 0 };
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; // 항상 상단이 원점
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* pBits = nullptr;
    HBITMAP hBmp = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, &pBits, NULL, 0);
    if (!hBmp || !pBits) return NULL;

    if (flip) {
        // 하단이 원점인 BMP는 상하 반전해서 복사
        for (int y = 0; y < height; ++y) {
            memcpy((BYTE*)pBits + y * stride, buf.data() + (height - 1 - y) * stride, stride);
        }
    } else {
        memcpy(pBits, buf.data(), stride * height);
    }

    if (outWidth) *outWidth = width;
    if (outHeight) *outHeight = height;
    return hBmp;
}

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: 여기에 코드를 입력합니다.

    // 전역 문자열을 초기화합니다.
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_TERMPROJECT, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // 애플리케이션 초기화를 수행합니다:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_TERMPROJECT));

    MSG msg;

    // 기본 메시지 루프입니다:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}



//
//  함수: MyRegisterClass()
//
//  용도: 창 클래스를 등록합니다.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_TERMPROJECT));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_TERMPROJECT);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   함수: InitInstance(HINSTANCE, int)
//
//   용도: 인스턴스 핸들을 저장하고 주 창을 만듭니다.
//
//   주석:
//
//        이 함수를 통해 인스턴스 핸들을 전역 변수에 저장하고
//        주 프로그램 창을 만든 다음 표시합니다.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // 인스턴스 핸들을 전역 변수에 저장합니다.

   HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//
//  함수: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  용도: 주 창의 메시지를 처리합니다.
//
//  WM_COMMAND  - 애플리케이션 메뉴를 처리합니다.
//  WM_PAINT    - 주 창을 그립니다.
//  WM_DESTROY  - 종료 메시지를 게시하고 반환합니다.
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    static int imgWidth = 0, imgHeight = 0;

    switch (message)
    {
    case WM_CREATE:
        hBtnBW = CreateWindowW(L"BUTTON", L"흑백변환",
            WS_CHILD | WS_VISIBLE,
            10, 10, 100, 30, hWnd, (HMENU)IDC_BTN_BW, hInst, NULL);
        hBtnBW_MMX = CreateWindowW(L"BUTTON", L"MMX 흑백변환",
            WS_CHILD | WS_VISIBLE,
            120, 10, 120, 30, hWnd, (HMENU)IDC_BTN_BW_MMX, hInst, NULL);
        hBtnBW_SSE = CreateWindowW(L"BUTTON", L"SSE 흑백변환",
            WS_CHILD | WS_VISIBLE,
            250, 10, 120, 30, hWnd, (HMENU)IDC_BTN_BW_SSE, hInst, NULL);
        ShowWindow(hBtnBW, SW_HIDE);
        ShowWindow(hBtnBW_MMX, SW_HIDE);
        ShowWindow(hBtnBW_SSE, SW_HIDE);
        break;

    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            switch (wmId)
            {
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_OPEN:
                {
                    OPENFILENAME ofn = { 0 };
                    WCHAR szFile[MAX_PATH] = L"";
                    ofn.lStructSize = sizeof(ofn);
                    ofn.hwndOwner = hWnd;
                    ofn.lpstrFilter = L"Bitmap Files (*.bmp)\0*.bmp\0All Files (*.*)\0*.*\0";
                    ofn.lpstrFile = szFile;
                    ofn.nMaxFile = MAX_PATH;
                    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
                    ofn.lpstrTitle = L"비트맵 파일 열기";

                    if (GetOpenFileName(&ofn)) {
                        ReleaseBitmap();
                        hBitmap = LoadBitmap32FromFile(szFile, &imgWidth, &imgHeight);
                        if (hBitmap) {
                            wcscpy_s(szBmpFile, szFile);
                            hBitmapBW = NULL;
                            hBitmapBW_MMX = NULL;
                            hBitmapBW_SSE = NULL;
                            RECT rc;
                            GetClientRect(hWnd, &rc);

                            // 버튼 위치 계산 (가로로)
                            int btnY = imgHeight + 20 - yScrollPos;
                            SetWindowPos(hBtnBW, NULL, 10, btnY, 100, 30, SWP_SHOWWINDOW);
                            SetWindowPos(hBtnBW_MMX, NULL, 120, btnY, 120, 30, SWP_SHOWWINDOW);
                            SetWindowPos(hBtnBW_SSE, NULL, 250, btnY, 120, 30, SWP_SHOWWINDOW);
                            ShowWindow(hBtnBW, SW_SHOW);
                            ShowWindow(hBtnBW_MMX, SW_SHOW);
                            ShowWindow(hBtnBW_SSE, SW_SHOW);

                            // 스크롤바 설정 (가로: 원본+변환*3, 세로: 이미지+버튼)
                            int totalWidth = imgWidth * (1 + (hBitmapBW ? 1 : 0) + (hBitmapBW_MMX ? 1 : 0) + (hBitmapBW_SSE ? 1 : 0));
                            int totalHeight = imgHeight + 10 + 30;
                            SCROLLINFO si = { sizeof(SCROLLINFO), SIF_RANGE | SIF_PAGE };
                            si.nMin = 0;
                            si.nMax = max(totalWidth, rc.right - rc.left) - 1;
                            si.nPage = rc.right - rc.left;
                            SetScrollInfo(hWnd, SB_HORZ, &si, TRUE);
                            xMaxScroll = max(0, totalWidth - (rc.right - rc.left));
                            si.nMax = max(totalHeight, rc.bottom - rc.top) - 1;
                            si.nPage = rc.bottom - rc.top;
                            SetScrollInfo(hWnd, SB_VERT, &si, TRUE);
                            yMaxScroll = max(0, totalHeight - (rc.bottom - rc.top));
                            xScrollPos = yScrollPos = 0;
                            InvalidateRect(hWnd, NULL, TRUE);
                        } else {
                            MessageBox(hWnd, L"비트맵 파일을 불러올 수 없습니다.", L"오류", MB_ICONERROR);
                        }
                    }
                }
                break;
            case IDC_BTN_BW:
                if (hBitmap && !hBitmapBW) {
                    hBitmapBW = ConvertColorToBW(hBitmap);
                    InvalidateRect(hWnd, NULL, TRUE);
                }
                break;
            case IDC_BTN_BW_MMX:
                if (hBitmap && !hBitmapBW_MMX) {
                    hBitmapBW_MMX = ConvertColorToBW_MMX(hBitmap);
                    InvalidateRect(hWnd, NULL, TRUE);
                }
                break;
            case IDC_BTN_BW_SSE:
                if (hBitmap && !hBitmapBW_SSE) {
                    hBitmapBW_SSE = ConvertColorToBW_SSE(hBitmap);
                    InvalidateRect(hWnd, NULL, TRUE);
                }
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;

    case WM_SIZE:
        {
            RECT rc;
            GetClientRect(hWnd, &rc);
            int nImages = 1 + (hBitmapBW ? 1 : 0) + (hBitmapBW_MMX ? 1 : 0) + (hBitmapBW_SSE ? 1 : 0);
            int totalWidth = imgWidth * nImages;
            int totalHeight = imgHeight + 10 + 30;
            SCROLLINFO si = { sizeof(SCROLLINFO), SIF_RANGE | SIF_PAGE };
            si.nMin = 0;
            si.nMax = max(totalWidth, rc.right - rc.left) - 1;
            si.nPage = rc.right - rc.left;
            SetScrollInfo(hWnd, SB_HORZ, &si, TRUE);
            xMaxScroll = max(0, totalWidth - (rc.right - rc.left));
            si.nMax = max(totalHeight, rc.bottom - rc.top) - 1;
            si.nPage = rc.bottom - rc.top;
            SetScrollInfo(hWnd, SB_VERT, &si, TRUE);
            yMaxScroll = max(0, totalHeight - (rc.bottom - rc.top));
            // 버튼 위치 조정 (가로로)
            int btnY = imgHeight + 20 - yScrollPos;
            SetWindowPos(hBtnBW, NULL, 10, btnY, 100, 30, SWP_NOZORDER);
            SetWindowPos(hBtnBW_MMX, NULL, 120, btnY, 120, 30, SWP_NOZORDER);
            SetWindowPos(hBtnBW_SSE, NULL, 250, btnY, 120, 30, SWP_NOZORDER);
        }
        break;

    case WM_HSCROLL:
        {
            int nScrollCode = LOWORD(wParam);
            int nPos = HIWORD(wParam);
            SCROLLINFO si = { sizeof(SCROLLINFO), SIF_ALL };
            GetScrollInfo(hWnd, SB_HORZ, &si);
            int prevPos = xScrollPos;
            switch (nScrollCode) {
            case SB_LINELEFT:   xScrollPos = max(xScrollPos - 10, 0); break;
            case SB_LINERIGHT:  xScrollPos = min(xScrollPos + 10, xMaxScroll); break;
            case SB_PAGELEFT:   xScrollPos = max(xScrollPos - (int)si.nPage, 0); break;
            case SB_PAGERIGHT:  xScrollPos = min(xScrollPos + (int)si.nPage, xMaxScroll); break;
            case SB_THUMBTRACK: xScrollPos = nPos; break;
            }
            if (xScrollPos != prevPos) {
                SetScrollPos(hWnd, SB_HORZ, xScrollPos, TRUE);
                InvalidateRect(hWnd, NULL, TRUE);
            }
        }
        break;

    case WM_VSCROLL:
        {
            int nScrollCode = LOWORD(wParam);
            int nPos = HIWORD(wParam);
            SCROLLINFO si = { sizeof(SCROLLINFO), SIF_ALL };
            GetScrollInfo(hWnd, SB_VERT, &si);
            int prevPos = yScrollPos;
            switch (nScrollCode) {
            case SB_LINEUP:     yScrollPos = max(yScrollPos - 10, 0); break;
            case SB_LINEDOWN:   yScrollPos = min(yScrollPos + 10, yMaxScroll); break;
            case SB_PAGEUP:     yScrollPos = max(yScrollPos - (int)si.nPage, 0); break;
            case SB_PAGEDOWN:   yScrollPos = min(yScrollPos + (int)si.nPage, yMaxScroll); break;
            case SB_THUMBTRACK: yScrollPos = nPos; break;
            }
            if (yScrollPos != prevPos) {
                SetScrollPos(hWnd, SB_VERT, yScrollPos, TRUE);
                int btnY = imgHeight + 20 - yScrollPos;
                SetWindowPos(hBtnBW, NULL, 10, btnY, 100, 30, SWP_NOZORDER);
                SetWindowPos(hBtnBW_MMX, NULL, 120, btnY, 120, 30, SWP_NOZORDER);
                SetWindowPos(hBtnBW_SSE, NULL, 250, btnY, 120, 30, SWP_NOZORDER);
                InvalidateRect(hWnd, NULL, TRUE);
            }
        }
        break;

    case WM_MOUSEWHEEL:
        {
            // 휠 한 칸당 120, 기본적으로 3줄씩 스크롤
            int zDelta = GET_WHEEL_DELTA_WPARAM(wParam);
            int lines = 3; // 한 번에 스크롤할 줄 수
            int scrollAmount = lines * 10; // 기존 한 번에 10픽셀씩 스크롤

            if (zDelta > 0)
                yScrollPos = max(yScrollPos - scrollAmount, 0);
            else if (zDelta < 0)
                yScrollPos = min(yScrollPos + scrollAmount, yMaxScroll);

            SetScrollPos(hWnd, SB_VERT, yScrollPos, TRUE);

            // 버튼 위치도 갱신
            int btnY = imgHeight + 20 - yScrollPos;
            SetWindowPos(hBtnBW, NULL, 10, btnY, 100, 30, SWP_NOZORDER);
            SetWindowPos(hBtnBW_MMX, NULL, 120, btnY, 120, 30, SWP_NOZORDER);
            SetWindowPos(hBtnBW_SSE, NULL, 250, btnY, 120, 30, SWP_NOZORDER);

            InvalidateRect(hWnd, NULL, TRUE);
        }
        break;

    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            int yOffset = -yScrollPos;
            int xOffset = -xScrollPos;
            int curX = xOffset;
            if (hBitmap) {
                HDC hMemDC = CreateCompatibleDC(hdc);
                HBITMAP hOldBmp = (HBITMAP)SelectObject(hMemDC, hBitmap);

                BITMAP bmp;
                GetObject(hBitmap, sizeof(BITMAP), &bmp);
                // 원본 비트맵을 (curX, yOffset)에 출력
                BitBlt(hdc, curX, yOffset, bmp.bmWidth, bmp.bmHeight, hMemDC, 0, 0, SRCCOPY);
                curX += bmp.bmWidth;

                SelectObject(hMemDC, hOldBmp);
                DeleteDC(hMemDC);

                // 변환 이미지들 오른쪽에 순서대로 출력
                if (hBitmapBW) {
                    HDC hMemDCBW = CreateCompatibleDC(hdc);
                    HBITMAP hOldBmpBW = (HBITMAP)SelectObject(hMemDCBW, hBitmapBW);
                    BITMAP bmpBW;
                    GetObject(hBitmapBW, sizeof(BITMAP), &bmpBW);
                    BitBlt(hdc, curX, yOffset, bmpBW.bmWidth, bmpBW.bmHeight, hMemDCBW, 0, 0, SRCCOPY);
                    // 시간 문자열 출력 (이미지 하단)
                    if (g_bwTimeStr[0]) {
                        int textY = yOffset + bmpBW.bmHeight + 10;
                        SetBkMode(hdc, TRANSPARENT);
                        SetTextColor(hdc, RGB(0,0,0));
                        TextOutW(hdc, curX + 10, textY, g_bwTimeStr, (int)wcslen(g_bwTimeStr));
                    }
                    curX += bmpBW.bmWidth;
                    SelectObject(hMemDCBW, hOldBmpBW);
                    DeleteDC(hMemDCBW);
                }
                if (hBitmapBW_MMX) {
                    HDC hMemDCBW = CreateCompatibleDC(hdc);
                    HBITMAP hOldBmpBW = (HBITMAP)SelectObject(hMemDCBW, hBitmapBW_MMX);
                    BITMAP bmpBW;
                    GetObject(hBitmapBW_MMX, sizeof(BITMAP), &bmpBW);
                    BitBlt(hdc, curX, yOffset, bmpBW.bmWidth, bmpBW.bmHeight, hMemDCBW, 0, 0, SRCCOPY);
                    curX += bmpBW.bmWidth;
                    SelectObject(hMemDCBW, hOldBmpBW);
                    DeleteDC(hMemDCBW);
                }
                if (hBitmapBW_SSE) {
                    HDC hMemDCBW = CreateCompatibleDC(hdc);
                    HBITMAP hOldBmpBW = (HBITMAP)SelectObject(hMemDCBW, hBitmapBW_SSE);
                    BITMAP bmpBW;
                    GetObject(hBitmapBW_SSE, sizeof(BITMAP), &bmpBW);
                    BitBlt(hdc, curX, yOffset, bmpBW.bmWidth, bmpBW.bmHeight, hMemDCBW, 0, 0, SRCCOPY);
                    curX += bmpBW.bmWidth;
                    SelectObject(hMemDCBW, hOldBmpBW);
                    DeleteDC(hMemDCBW);
                }
            }
            EndPaint(hWnd, &ps);
        }
        break;

    case WM_DESTROY:
        ReleaseBitmap();
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// 정보 대화 상자의 메시지 처리기입니다.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
