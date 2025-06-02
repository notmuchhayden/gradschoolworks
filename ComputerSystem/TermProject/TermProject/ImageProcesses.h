#pragma once

// ��� ��ȯ
// C ���� �÷� �̹����� ������� ��ȯ �Լ� ����
void ConvertRGBAToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
// MMX ��ɾ ����Ͽ� �÷� �̹����� ������� ��ȯ �Լ� ����
extern "C" void ConvertRGBAToBW_MMX_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
// SSE ��ɾ ����Ͽ� �÷� �̹����� ������� ��ȯ �Լ� ����
extern "C" void ConvertRGBAToBW_SSE_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);

// Sharpen ���� ����
void Sharpen_C(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
// MMX ��ɾ ����Ͽ� Sharpen ���� ���� �Լ� ����
extern "C" void Sharpen_MMX_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
// SSE ��ɾ ����Ͽ� Sharpen ���� ���� �Լ� ����
extern "C" void Sharpen_SSE_(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);