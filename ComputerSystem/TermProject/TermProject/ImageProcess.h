#pragma once

// C ���� �÷� �̹����� ������� ��ȯ �Լ� ����
void ConvertRGBToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
void ConvertRGBAToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);

// MMX ��ɾ ����Ͽ� �÷� �̹����� ������� ��ȯ �Լ� ����
extern "C" void ConvertRGBAToBW_MMX_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
extern "C" void ConvertRGBAToBW_SSE_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
