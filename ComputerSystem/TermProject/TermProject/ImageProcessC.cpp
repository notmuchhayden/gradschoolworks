#include "ImageProcess.h"

// Convert RGB image to grayscale (black and white)
void ConvertRGBToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight)
{
	int j = 0, i = 0, nValue = 0;
	int nWidth3 = nWidth * 3;
	for (j = 0; j < nHeight; j++) {
		for (i = 0; i < nWidth3; i += 3) {
			nValue =	bufIn[i + j * nWidth3 + 0] * 0.299f + 
						bufIn[i + j * nWidth3 + 1] * 0.587f + 
						bufIn[i + j * nWidth3 + 2] * 0.114f;

			bufOut[i + j * nWidth3 + 0] = nValue;
			bufOut[i + j * nWidth3 + 1] = nValue;
			bufOut[i + j * nWidth3 + 2] = nValue;
		}
	}
}

// Convert an RGBA image to a black and white image
void ConvertRGBAToBW(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight)
{
	int j = 0, i = 0, nValue = 0;
	int nWidth4 = nWidth * 4;
	for (j = 0; j < nHeight; j++) {
		for (i = 0; i < nWidth4; i += 4) {
			nValue = bufIn[i + j * nWidth4 + 0] * 0.299 +
				bufIn[i + j * nWidth4 + 1] * 0.587 +
				bufIn[i + j * nWidth4 + 2] * 0.114;

			bufOut[i + j * nWidth4 + 0] = nValue;
			bufOut[i + j * nWidth4 + 1] = nValue;
			bufOut[i + j * nWidth4 + 2] = nValue;
			bufOut[i + j * nWidth4 + 3] = bufIn[i + j * nWidth4 + 3]; // Alpha channel remains unchanged
		}
	}
}
