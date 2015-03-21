#include<stdio.h>
#include<stdlib.h>
#include"hog.h"
#include"cv.h"


CvMat *magnitude = 0, *orientation = 0;
CvHistogram *hist = 0;
CvMat *descriptor;


void hogBlockNormalization ( CvSize szBlock, double epsilon )
{
	assert ( hist );

	CvMatND *bins = (CvMatND*)(hist->bins);

	int nBins = bins->dim[0].size;
	int nCells_x = bins->dim[1].size;
	int nCells_y = bins->dim[2].size;

	//CvHistogram *norm_hist = 0;
	//cvCopyHist ( hist, &norm_hist );

	int entries = ( nCells_x - szBlock.width + 1 )*
					( nCells_y - szBlock.height +1 )*
					szBlock.height * szBlock.width * nBins;

	descriptor = cvCreateMat ( 1, entries, CV_32FC1 );
	cvZero(descriptor);

	int count = 0;

//	assert ( norm_hist );


	for ( int i = 0 ; i < nCells_y - szBlock.height + 1 ; ++i )
	{
		for ( int j = 0 ; j < nCells_x -szBlock.width + 1; ++j )
		{
			float sum = 0;

			for ( int k = i ; k < szBlock.height ; ++k )
			{
				for ( int l = j ; l < szBlock.width ; ++l )
				{
					for ( int z = 0 ; z < nBins ; ++z )
					{
						sum += cvGetReal3D( hist->bins, z, l, k )*
								cvGetReal3D( hist->bins, z, l, k );
					}
				}
			}

			sum += epsilon*epsilon;
			sum = cvSqrt (sum);

			for ( int k = i ; k < szBlock.height ; ++k )
			{
				for ( int l = j ; l < szBlock.width ; ++l )
				{
					for ( int z = 0 ; z < nBins ; ++z )
					{
						float v = cvGetReal3D( hist->bins, z, l, k )/sum;
						CV_MAT_ELEM ( *descriptor, float, 0, count++ ) = v;
					}
				}
			}
		}
	}
}

//compute HoG and return a feature vector
//of an image
void hog(CvMat *&vect, IplImage* src, CvSize szCell, int nBins, CvSize szBlock, float epsilon){

	assert ( src );

	hogCalcGradient( src );

	assert ( magnitude && orientation );

	hogCalcCellHistogram( szCell, nBins, 0.0 );

	hogBlockNormalization ( szBlock, epsilon );

	cvReleaseMat ( &magnitude );
	cvReleaseMat ( &orientation );
	cvReleaseHist ( &hist );

	vect = descriptor;
}

void hogCalcGradient ( IplImage *img )
{

	assert ( img != 0 && img->nChannels == 1 );

	IplImage *deriv_x = cvCreateImage( cvGetSize(img), IPL_DEPTH_16S, 1);
	IplImage *deriv_y = cvCreateImage( cvGetSize(img), IPL_DEPTH_16S, 1);

	cvSobel ( img, deriv_x, 1, 0, 1 );

	cvSobel ( img, deriv_y, 0, 1, 1 );

	CvSize size = cvGetSize( img );

	magnitude = cvCreateMat ( size.height, size.width, CV_32FC1 );
	orientation = cvCreateMat ( size.height, size.width, CV_32FC1 );

	for ( int i = 0 ; i < size.height ; ++i )
	{
		for ( int j = 0 ; j < size.width ; ++j )
		{
			float dx = CV_IMAGE_ELEM ( deriv_x, short, i, j );
			float dy = CV_IMAGE_ELEM ( deriv_y, short, i, j );

			CV_MAT_ELEM(*magnitude,float,i,j) = cvSqrt ( dx*dx + dy*dy );
			CV_MAT_ELEM(*orientation,float,i,j) = cvFastArctan( dy, dx );
		}
	}

	cvReleaseImage( &deriv_x );
	cvReleaseImage( &deriv_y );
}

void hogCalcCellHistogram ( CvSize szCell, int nBins, int is_signed = 0 )
{
	assert ( magnitude && orientation && nBins > 0 && szCell.height > 0 && szCell.width > 0);

	CvSize szIm = cvGetSize( magnitude );

	int border_x = szIm.width % szCell.width;
	int border_y = szIm.height % szCell.height;
	int nCells_x = (szIm.width - border_x) / szCell.width;
	int nCells_y = (szIm.height - border_y) / szCell.height;

	CvRect rect = cvRect(0, 0, szIm.width-border_x, szIm.height-border_y);


	CvMat *roi = cvCreateMatHeader(szIm.height-border_y,
									szIm.width-border_x,
									magnitude->type);

	roi = cvGetSubRect( magnitude, roi, rect );
	assert (roi);


	/************************************************************************/
	/*init histogram                                                        */

	int dim = 3; //nCells_x * nCells_y;
	int *size = (int*)cvAlloc(dim*sizeof(int));

	int i,j;

	size[0] = nBins;
	size[1] = nCells_x;
	size[2] = nCells_y;

	int totalBins = size[0]*size[1]*size[2];

	float** range = (float**)cvAlloc( nBins * sizeof(float*) );

	for ( i = 0 ; i < nBins ; ++i )
	{
		range[i] = (float*)cvAlloc(2*sizeof(float));
		range[i][0] = 0.0;
		range[i][1] = (1 + is_signed)*180.0;
	}

	hist = cvCreateHist ( dim, size, CV_HIST_ARRAY, range );
	cvClearHist (hist);
	/************************************************************************/


	/************************************************************************/
	/* calc hist cell                                                       */
	int angular = 180*(1+is_signed)/nBins;
	int cell_count = 0;

	for ( i = 0 ; i < nCells_x ; ++i )
	{
		for ( j = 0 ; j < nCells_y ; ++j )
		{
			int pos_x = szCell.width*i;
			for ( int u = pos_x; u < pos_x + szCell.width ; u++ )
			{
				int pos_y = szCell.height*j;
				for ( int v = pos_y ; v < pos_y + szCell.height ; v++ )
				{
					float theta = CV_MAT_ELEM ( *orientation, float, v, u );
					float mag = CV_MAT_ELEM ( *magnitude, float, v, u );

					theta = theta > 180 ? theta - 180 : theta;
					int bin_index = floor(theta / angular);
					float residue = theta - 1.5*bin_index*angular;

					float value1 = 0, value2 = 0;
					int prev_bin, forw_bin;

					if ( residue > 0 )
					{
						prev_bin = bin_index;
						forw_bin = bin_index + 1;
					}else if ( residue < 0 )
					{
						prev_bin = bin_index - 1;
						forw_bin = bin_index;
					}else
						prev_bin = forw_bin = bin_index;

					prev_bin = prev_bin < 0 ? nBins-1 : prev_bin;
					forw_bin = forw_bin == nBins ? 0 : forw_bin;

					value1 = cvGetReal3D( hist->bins, prev_bin, i, j );
					float weight1 = value1 + mag * ( fabs(residue)/float(angular) );
					cvSetReal3D( hist->bins, prev_bin, i, j, weight1 );

					if ( prev_bin == forw_bin )
					{
						value2 = cvGetReal3D( hist->bins, forw_bin, i, j );
						float weight2 = value2 + mag * ( 1-fabs(residue)/float(angular) );
						cvSetReal3D( hist->bins, forw_bin, i, j, weight2 );
					}
				}
			}
		}

	}

}


/*int main(int argc, char* argv[]){

	printf("starting...");

	if ( argc == 10 ) {

		IplImage* img = cvLoadImage(argv[1],0);
		int height = img->height;
		int width  = img->width;

		FILE* pKey = fopen(argv[2],"r");
		FILE* pOut = fopen(argv[4],"w");

		int n, size;
		float x,y;
		int win_size = atoi(argv[3]);

		int nBin = atoi(argv[7]);
		int cell_x_size  = atoi(argv[5]);
		int cell_y_size  = atoi(argv[6]);
		int block_x_size = atoi(argv[8]);
		int block_y_size = atoi(argv[9]);

		fscanf(pKey,"%d %d",&n,&size);

		CvRect rect;

		CvMat* descriptor;
		CvMat* roi = cvCreateMatHeader(win_size, win_size,IPL_DEPTH_8U);
		IplImage* subimage = cvCreateImageHeader(cvSize(win_size,win_size),IPL_DEPTH_8U,1);
		fscanf(pKey,"%d",&n);

		int inc = 0;

		for(int i = 0 ; i < n ; i++){

			fscanf(pKey,"%f",&x);
			fscanf(pKey,"%f",&y);

			rect.x = x - win_size/2;
			rect.y = y - win_size/2;
			rect.height = win_size;
			rect.width  = win_size;

			if (rect.x > 0 && rect.x + win_size < width &&
				rect.y > 0 && rect.y + win_size < height) {

				inc++;

				cvGetSubRect(img,roi,rect);

				cvGetImage(roi,subimage);

				hog(descriptor, subimage, cvSize(cell_x_size,cell_y_size),nBin, cvSize(block_x_size, block_y_size), 0.005 );

				int desc_size = descriptor->width;

				float tmp;
				for(int i = 0 ; i < desc_size ; i++){
					tmp = CV_MAT_ELEM(*descriptor,float,0,i);
					fprintf(pOut,"%f ",tmp);
				}

				fprintf(pOut,"\n");
			}

		}

		fclose(pKey);
		fclose(pOut);

		cvReleaseMat(&roi);
		//cvReleaseImage(&subimage);
		cvReleaseImage(&img);

	}
	else{
		printf("args error! - args should be: <*.exe> <image> <keypoint> <win_size> <output> <cell_x_size> <cell_y_size> <nBins> <block_x_size> <block_y_size>");
	}
}*/
