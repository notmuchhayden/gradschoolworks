#include "ImageProcesses.h"


void Sharpen_C(unsigned char* bufIn, unsigned char* bufOut, int nWidth, int nHeight) {
	int j = 0, i = 0;

	short kernel[9] = { -1, -1, -1,
						-1,  9, -1,
						-1, -1, -1 };
	int nImageWidth4 = nWidth * 4;
	short total = 0;
	short value[9];

	for (j = 1; j < nHeight - 1; j++) {
		for (i = 4; i < nImageWidth4; i += 4) {
			total = 0;

			// red sharpen filter
			value[0] = bufIn[i + 0 + j * nImageWidth4 - nImageWidth4 - 4]; // top left
			total += value[0] * kernel[0];
			value[1] = bufIn[i + 0 + j * nImageWidth4 - nImageWidth4]; // top
			total += value[1] * kernel[1];
			value[2] = bufIn[i + 0 + j * nImageWidth4 - nImageWidth4 + 4]; // top right
			total += value[2] * kernel[2];
			value[3] = bufIn[i + 0 + j * nImageWidth4 - 4]; // left
			total += value[3] * kernel[3];
			value[4] = bufIn[i + 0 + j * nImageWidth4]; // center
			total += value[4] * kernel[4];
			value[5] = bufIn[i + 0 + j * nImageWidth4 + 4]; // right
			total += value[5] * kernel[5];
			value[6] = bufIn[i + 0 + j * nImageWidth4 + nImageWidth4 - 4]; // bottom left
			total += value[6] * kernel[6];
			value[7] = bufIn[i + 0 + j * nImageWidth4 + nImageWidth4]; // bottom
			total += value[7] * kernel[7];
			value[8] = bufIn[i + 0 + j * nImageWidth4 + nImageWidth4 + 4]; // bottom right
			total += value[8] * kernel[8];
			if (total < 0)
				total = 0;
			else if (total > 255)
				total = 255;
			bufOut[i + 0 + j * nImageWidth4] = (unsigned char)total; // red

			// green sharpen filter
			total = 0;
			value[0] = bufIn[i + 1 + j * nImageWidth4 - nImageWidth4 - 4]; // top left
			total += value[0] * kernel[0];
			value[1] = bufIn[i + 1 + j * nImageWidth4 - nImageWidth4]; // top
			total += value[1] * kernel[1];
			value[2] = bufIn[i + 1 + j * nImageWidth4 - nImageWidth4 + 4]; // top right
			total += value[2] * kernel[2];
			value[3] = bufIn[i + 1 + j * nImageWidth4 - 4]; // left
			total += value[3] * kernel[3];
			value[4] = bufIn[i + 1 + j * nImageWidth4]; // center
			total += value[4] * kernel[4];
			value[5] = bufIn[i + 1 + j * nImageWidth4 + 4]; // right
			total += value[5] * kernel[5];
			value[6] = bufIn[i + 1 + j * nImageWidth4 + nImageWidth4 - 4]; // bottom left
			total += value[6] * kernel[6];
			value[7] = bufIn[i + 1 + j * nImageWidth4 + nImageWidth4]; // bottom
			total += value[7] * kernel[7];
			value[8] = bufIn[i + 1 + j * nImageWidth4 + nImageWidth4 + 4]; // bottom right
			total += value[8] * kernel[8];
			if (total < 0)
				total = 0;
			else if (total > 255)
				total = 255;
			bufOut[i + 1 + j * nImageWidth4] = (unsigned char)total; // green

			// blue sharpen filter
			total = 0;
			value[0] = bufIn[i + 2 + j * nImageWidth4 - nImageWidth4 - 4]; // top left
			total += value[0] * kernel[0];
			value[1] = bufIn[i + 2 + j * nImageWidth4 - nImageWidth4]; // top
			total += value[1] * kernel[1];
			value[2] = bufIn[i + 2 + j * nImageWidth4 - nImageWidth4 + 4]; // top right
			total += value[2] * kernel[2];
			value[3] = bufIn[i + 2 + j * nImageWidth4 - 4]; // left
			total += value[3] * kernel[3];
			value[4] = bufIn[i + 2 + j * nImageWidth4]; // center
			total += value[4] * kernel[4];
			value[5] = bufIn[i + 2 + j * nImageWidth4 + 4]; // right
			total += value[5] * kernel[5];
			value[6] = bufIn[i + 2 + j * nImageWidth4 + nImageWidth4 - 4]; // bottom left
			total += value[6] * kernel[6];
			value[7] = bufIn[i + 2 + j * nImageWidth4 + nImageWidth4]; // bottom
			total += value[7] * kernel[7];
			value[8] = bufIn[i + 2 + j * nImageWidth4 + nImageWidth4 + 4]; // bottom right
			total += value[8] * kernel[8];
			if (total < 0)
				total = 0;
			else if (total > 255)
				total = 255;
			bufOut[i + 2 + j * nImageWidth4] = (unsigned char)total; // blue
		}
	}
}