#pragma once

// C 언어로 컬러 이미지를 흑백으로 변환 함수 선언
void ConvertRGBToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
void ConvertRGBAToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);

// MMX 명령어를 사용하여 컬러 이미지를 흑백으로 변환 함수 선언
extern "C" void ConvertRGBAToBW_MMX_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
extern "C" void ConvertRGBAToBW_SSE_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
