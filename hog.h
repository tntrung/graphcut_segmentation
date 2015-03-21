#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

void hogCalcCellHistogram( CvSize szCell, int nBins, int is_signed );
void hogBlockNormalization( CvSize szBlock, double epsilon );
void hogCalcGradient( IplImage *img );
void hog(CvMat *&vect, IplImage* src, CvSize szCell, int nBins, CvSize szBlock, float epsilon);









