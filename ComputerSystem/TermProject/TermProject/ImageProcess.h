#pragma once

// C ���� �÷� �̹����� ������� ��ȯ �Լ� ����
void ConvertRGBToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
void ConvertRGBAToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);

// MMX ��ɾ ����Ͽ� �÷� �̹����� ������� ��ȯ �Լ� ����
void ConvertRGBToBW_MMX(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);
void ConvertRGBAToBW_MMX(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight);