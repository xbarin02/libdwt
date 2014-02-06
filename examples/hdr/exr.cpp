#include "exr.h"
#include "libdwt.h"
#include "inline.h"

#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfNamespace.h>

using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;
using namespace std;

int dwt_util_load_from_exr_s(
	const char *filename,	///< input file name, e.g. "input.pgm"
	const char *channel,	///< "R", "G", or "B"
	void **pptr,		///< place the pointer to beginning of image data at this address
	int *pstride_x,		///< place the difference between rows (in bytes) at this address
	int *pstride_y,		///< place the difference between columns (in bytes) at this address
	int *psize_x,		///< place the width of the image (in elements) at this address
	int *psize_y		///< place the height of the image (in elements) at this address
)
{
	int w = 1, h = 1;

	Array2D<float> ch(h, w);

	InputFile file(filename);

	Box2i dw = file.header().dataWindow();
	w  = dw.max.x - dw.min.x + 1;
	h = dw.max.y - dw.min.y + 1;

	*psize_x = w;
	*psize_y = h;
	*pstride_y = sizeof(float);
	*pstride_x = dwt_util_get_opt_stride(*pstride_y * *psize_x);

	dwt_util_alloc_image(pptr, *pstride_x, *pstride_y, *psize_x, *psize_y);

	ch.resizeErase(h, w);

	FrameBuffer frameBuffer;

	frameBuffer.insert(channel,
		Slice(FLOAT,
			(char *) (&ch[0][0] -
				dw.min.x -
				dw.min.y * w),
			sizeof (ch[0][0]) * 1,
			sizeof (ch[0][0]) * w,
			1, 1,
			0.0));

	file.setFrameBuffer(frameBuffer);
	file.readPixels(dw.min.y, dw.max.y);

	for(int y=0; y<h; y++)
		for(int x=0; x<w; x++)
		{
			float px = ch[y][x];

			float *ppx = addr2_s(*pptr, y, x, *pstride_x, *pstride_y);

			*ppx = px;
		}

	return 0;
}
