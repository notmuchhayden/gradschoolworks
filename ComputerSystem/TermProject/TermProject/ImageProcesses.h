#pragma once

// 흑백 변환
// C 언어로 컬러 이미지를 흑백으로 변환 함수 선언
void ConvertRGBAToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
// MMX 명령어를 사용하여 컬러 이미지를 흑백으로 변환 함수 선언
extern "C" void ConvertRGBAToBW_MMX_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
// SSE 명령어를 사용하여 컬러 이미지를 흑백으로 변환 함수 선언
extern "C" void ConvertRGBAToBW_SSE_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);

// Sharpen 필터 적용
void Sharpen_C(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
// MMX 명령어를 사용하여 Sharpen 필터 적용 함수 선언
extern "C" void Sharpen_MMX_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
// SSE 명령어를 사용하여 Sharpen 필터 적용 함수 선언
extern "C" void Sharpen_SSE_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);