/*
 * fusion_framework.h
 *
 *  Created on: Feb 18, 2010
 *      Author: tranngoctrung
 */

#ifndef FUSION_FRAMEWORK_H_
#define FUSION_FRAMEWORK_H_


#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include "./max_flow/graph.h"
#include "feature_model.h"
#include "hog.h"


#define CONFIG_FILE "./config/config.xml"

#define COLOR "COLOR"
#define HOG   "HOG"

#define MAX_IMG_SIZE 1000

using namespace std;

typedef Graph<float,float,float> GraphType;

class Fusion_Framework {

public:

	GraphType *g;
	int w_img, h_img;

	int n_features;

	float K;
	float lamda;

	ForegroundPoints foreground;
	BackgroundRect   background;

	vector<Feature_Model*>* models;

	bool** foreground_flag;

public:

	Fusion_Framework ( int m,                /*estimated # of nodes*/
					   int n,                /*estimated # of edges*/
					   int w,                /*width of image*/
					   int h,                /*height of image*/
					   BackgroundRect *bkg,  /*background rectangle of hint area*/
					   ForegroundPoints *pts /*points of hint on foreground */
					   ) {

		models = new vector<Feature_Model*>();

		/*
		 *  initialize graph with m nodes and n edges
		 */
		g = new GraphType(m, n);
		w_img = w;
		h_img = h;

		init_params_load ( (char*)CONFIG_FILE );

		this->background = *bkg;
		this->foreground = *pts;

		foreground_flag = new bool*[h_img];

		for ( int i = 0 ; i < h_img ; i++ ) {
			foreground_flag[i] = new bool[w_img];
			for(int j = 0 ; j < w_img ; j++) {
				foreground_flag[i][j] = false;
			}
		}

	}

	void init_params_load( char* path ) {

		CvFileStorage* fs= cvOpenFileStorage( CONFIG_FILE , 0, CV_STORAGE_READ );

		char tmp[] = "_";
		char i2a_tmp[10];
		char* str = new char[30];

		K = -1;

		this->n_features = cvReadIntByName( fs, 0, "N", 1 /* default value */ );
		this->lamda      = cvReadRealByName( fs, 0, "LAMDA", 2 /* default value */ );

		for ( int i = 1 ; i <= n_features ; i++ ) {

			Feature_Model* fm = new Feature_Model();

			strcpy(str,"FEATURE");
			strcat(str,tmp);
			sprintf(i2a_tmp,"%d",i);
			strcat(str, i2a_tmp);

			CvFileNode* feature_map = cvGetFileNodeByName( fs, 0, "FEATURES" );
			feature_map = cvGetFileNodeByName( fs, feature_map, str );

			strcpy(str,"CLUSTER");
			strcat(str,tmp);
			sprintf(i2a_tmp,"%d",i);
			strcat(str, i2a_tmp);
			fm->n_cluster = cvReadIntByName( fs, feature_map, str, -1 );

			printf("%d\n",fm->n_cluster);

			strcpy(str,"DIM");
			strcat(str,tmp);
			sprintf(i2a_tmp,"%d",i);
			strcat(str, i2a_tmp);
			fm->dim = cvReadIntByName( fs, feature_map, str, -1);

			printf("%d\n",fm->dim);

			strcpy(str,"SIGMA");
			strcat(str,tmp);
			sprintf(i2a_tmp,"%d",i);
			strcat(str, i2a_tmp);
			fm->sigma = cvReadRealByName( fs, feature_map , str, -1 );

			printf("%s = %d\n", str , fm->sigma);

			strcpy(str,"LAMDA");
			strcat(str,tmp);
			sprintf(i2a_tmp,"%d",i);
			strcat(str, i2a_tmp);
			fm->lamda = cvReadIntByName( fs, feature_map , str, -1 );

			printf("%d\n",fm->lamda);

			strcpy(str,"W_PROB");
			strcat(str,tmp);
			sprintf(i2a_tmp,"%d",i);
			strcat(str, i2a_tmp);
			fm->w_prob = cvReadIntByName( fs, feature_map , str, -1 );

			printf("%d\n",fm->w_prob);

			strcpy(str,"W_EXP");
			strcat(str,tmp);
			sprintf(i2a_tmp,"%d",i);
			strcat(str, i2a_tmp);
			fm->w_exp = cvReadIntByName( fs, feature_map , str, -1 );

			printf("%d\n",fm->w_exp);

			strcpy(str,"TYPE");
			strcat(str,tmp);
			sprintf(i2a_tmp,"%d",i);
			strcat(str, i2a_tmp);
			strcpy(fm->name, cvReadStringByName( fs, feature_map, str ));

			printf("%s\n",fm->name);

			strcpy(str,"SUB_WINDOW_SIZE");
			strcat(str,tmp);
			sprintf(i2a_tmp,"%d",i);
			strcat(str, i2a_tmp);
			fm->sub_window_size = cvReadIntByName( fs, feature_map, str, -1 );

			printf("%d\n",fm->sub_window_size);

			fm->initImportantVariables( w_img, h_img );

			models->push_back( fm );

		}

		cvReleaseFileStorage( &fs);

	}

	IplImage* Sub_Image(IplImage *image, CvRect roi)
	{
		IplImage *result;

		cvSetImageROI(image,roi);

		result = cvCreateImage( cvSize(roi.width, roi.height), image->depth, image->nChannels );

		cvCopy(image,result);

		cvResetImageROI(image); // release image ROI

		return result;
	}

	void feature_extract ( IplImage* img, Feature_Model* fm, float** &data_points ) {

		int count  = 0;
		int width  = img->width;
		int height = img->height;
		int step       = img->widthStep;
		int nchannels  = img->nChannels;
		int dim    = fm->dim;

		if ( strcmp(fm->name,COLOR) == 0 ) {

			/* setup the pointer to access image data */
			uchar *data = ( uchar* )img->imageData;

			for ( int i = 0 ; i < height ; i++ ) {
				for ( int j = 0 ; j < width ; j++ ) {
					for ( int k = 0 ; k < dim ; k++ ) {
						data_points[count][k] = 1.0*data[i*step + j*nchannels + k];
					}
					count++;
				}
			}
		}

		else if ( strcmp(fm->name,HOG) == 0 ) {

			/*
			 * Load configuration variables of HOG feature.
			 */

			CvFileStorage* fs= cvOpenFileStorage( CONFIG_FILE , 0, CV_STORAGE_READ );

			CvFileNode* feature_map = cvGetFileNodeByName( fs, 0, "FEATURES" );
			feature_map = cvGetFileNodeByName( fs, feature_map, "FEATURE_2" );

			int nCell    = cvReadIntByName ( fs, feature_map, "CELL" );
			CvSize cvCell   = cvSize(nCell,nCell);
			int nBins    = cvReadIntByName ( fs, feature_map, "BIN" );
			int nBlock  = cvReadIntByName ( fs, feature_map, "BLOCK" );
			CvSize cvBlock   = cvSize(nBlock,nBlock);
			int epsilon  = cvReadRealByName( fs, feature_map, "EPSILON" );

			cvReleaseFileStorage( &fs );

			/*
			 * Load end.
			 */

			IplImage * gray = cvCreateImage(cvSize(w_img,h_img),8,1);
			cvCvtColor(img,gray,CV_RGB2GRAY);

			CvRect    sub_rect;
			IplImage* sub_img;
			CvMat*    hog_vec;

			int sub_window_size = fm->sub_window_size;

			if ( fm->sub_window_size == -1 ) {

				printf("sub-window size should be edited in configuration file!\n");
				return;

			}

			sub_rect.height = sub_rect.width = 2*fm->sub_window_size + 1;

			count = 0;

			for ( int i = 0 ; i < h_img ; i++ ) {
				for ( int j = 0 ; j < w_img ; j++ ) {

					sub_rect.x  = j - sub_window_size;
					sub_rect.y  = i - sub_window_size;

					if ( sub_rect.x < 0 ) {
						sub_rect.x = sub_rect.x + sub_window_size;
					}

					if ( sub_rect.y < 0 ) {
						sub_rect.y = sub_rect.y + sub_window_size;
					}

					if ( sub_rect.height + sub_rect.x >= w_img ) {
						sub_rect.x = sub_rect.x - sub_window_size;
					}

					if ( sub_rect.width + sub_rect.y >= h_img ) {
						sub_rect.y = sub_rect.y - sub_window_size;
					}

					sub_img = Sub_Image( gray, sub_rect );

					hog( hog_vec, sub_img, cvCell, nBins, cvBlock, epsilon );

					for (int k = 0 ; k < dim ; k++) {
						data_points[count][k] = CV_MAT_ELEM(*hog_vec,float,0,k);
					}

					cvReleaseImage(&sub_img);
					count++;

				}
			}

			cvReleaseImage(&gray);
			cvReleaseMat(&hog_vec);

		}

		else {

			printf("This feature is not supported\n!");
			return;

		}

	}

	Feature_Model* quantized_feature ( IplImage* img, int i ) {

		Feature_Model* fm = models->at(i);

		int num_points =  img->height*img->width;

		int* cluster_cnt = new int[fm->n_cluster];
		int  kcluster = fm->n_cluster;
		int  dim      = fm->dim;

		int count     = 0;

		float** data_points = new float*[num_points];
		for( int i = 0 ; i < num_points ; i++ ) {
			data_points[i] = new float[dim];
		}

		int size = foreground.pts.size();
		for ( int t = 0 ; t < size ; t++ ) {
			foreground_flag[foreground.pts.at(t).x][foreground.pts.at(t).y] = true;
		}

		feature_extract( img, fm, data_points );

		cvZero(fm->center);

		for (int i = 0 ; i < kcluster ; i++){

			cluster_cnt[i] = 0;

		}

		cvKMeans( fm->n_cluster,
				  data_points,
				  num_points,
				  fm->dim,
				  cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 1.0),
				  fm->cluster_idx);

		count = 0;
		for ( int i = 0 ; i < h_img ; i++){
			for ( int j = 0 ; j < w_img ; j++ ) {
					cluster_cnt[fm->cluster_idx[count]] = cluster_cnt[fm->cluster_idx[count]] + 1;
					for (int k = 0 ; k < dim ; k++) {
						CV_MAT_ELEM(*(fm->center),float,fm->cluster_idx[count],k) += data_points[count][k];
					}
					count++;
			}
		}

		for (int i = 0 ; i < kcluster ; i++ ) {
			for (int k = 0 ; k < dim ; k++) {
				CV_MAT_ELEM(*(fm->center),float,i,k) /= cluster_cnt[i];
			}
		}

		count = 0;
		int which_c;

		for ( int i = 0 ; i < h_img ; i++ ){
			for ( int j = 0 ; j < w_img ; j++ ) {

				which_c = fm->cluster_idx[count];

				if ( is_background(j,i) ){

					fm->bkg_hist.hist[which_c] = fm->bkg_hist.hist[which_c] + 1;
					fm->bkg_hist.total++;

				}
				else if ( is_foreground(i,j) ) {

					fm->obj_hist.hist[which_c] = fm->obj_hist.hist[which_c] + 1;
					fm->obj_hist.total++;

				}

				count++;

			}
		}

		for ( int i = 0 ; i < kcluster ; i++ ) {

			fm->bkg_hist.hist[i] = fm->bkg_hist.hist[i]*1.0/(fm->bkg_hist.total*1.0);
			fm->obj_hist.hist[i] = fm->obj_hist.hist[i]*1.0/(fm->obj_hist.total*1.0);

		}

		delete data_points;
		delete cluster_cnt;

		return fm;

	}

	void buildFramework ( IplImage* img ) {

		int n_node = img->width * img->height;
		int w = img->width;
		int h = img->height;

		for( int i = 0 ; i  < n_node ; i++ ) {
			g -> add_node();
		}

		int node = 0;

		for( int i = 0 ; i < h ; i++ ) {
			for( int j = 0 ; j < w ; j++ ) {

				g -> add_tweights(
								   node,
								   source_tweight(img,i,j),
								   sink_tweight(img,i,j)
								 );

				node++;

			}
		}

		//******* ver 1.0 *******************//
		//relationship between neighbor pixels.

		node = 0;

		for( int i = 0 ; i < h ; i++ ) {
			for( int j = 0 ; j < w ; j++ ) {

				if ( i-1>=0 && j-1>=0 )
					g -> add_edge( node,
							       pixel_2_node(i-1,j-1),
							       weight(img,node,pixel_2_node(i-1,j-1)),
							       weight(img,pixel_2_node(i-1,j-1),node)
							     );
				if ( i-1>=0 )
					g -> add_edge( node,
								   pixel_2_node(i-1,j),
								   weight(img,node,pixel_2_node(i-1,j)),
								   weight(img,pixel_2_node(i-1,j),node)
								 );
				if ( i-1>=0 && j+1<w )
					g -> add_edge( node,
								   pixel_2_node(i-1,j+1),
								   weight(img,node,pixel_2_node(i-1,j+1)),
								   weight(img,pixel_2_node(i-1,j+1),node)
								 );
				if ( j-1>=0 )
					g -> add_edge( node,
								   pixel_2_node(i  ,j-1),
								   weight(img,node,pixel_2_node(i  ,j-1)),
								   weight(img,pixel_2_node(i  ,j-1),node)
								 );
				if ( j+1<w )
					g -> add_edge( node,
								   pixel_2_node(i  ,j+1),
								   weight(img,node,pixel_2_node(i  ,j+1)),
								   weight(img,pixel_2_node(i  ,j+1),node)
								 );
				if ( i+1<h && j-1>=0 )
					g -> add_edge( node,
							       pixel_2_node(i+1,j-1),
							       weight(img,node,pixel_2_node(i+1,j-1)),
							       weight(img,pixel_2_node(i+1,j-1),node)
							     );
				if ( i+1<h ){
					g -> add_edge( node,
							       pixel_2_node(i+1,j  ),
							       weight(img,node,pixel_2_node(i+1,j  )),
							       weight(img,pixel_2_node(i+1,j  ),node)
							     );
				}
				if ( i+1<h && j+1<w )
					g -> add_edge( node,
								   pixel_2_node(i+1,j+1),
								   weight(img,node,pixel_2_node(i+1,j+1)),
								   weight(img,pixel_2_node(i+1,j+1),node)
								 );
				node++;
			}
		}
        //******* ver 1.0 *******************//

	}

	float source_tweight(IplImage* img, int i, int j){

		if ( is_background(j,i) ) {
			return 0;
		}
		else if ( is_foreground(i,j) ) {
			return K_factor(img);
		}
		else{
			return lamda * Pr_bkg(img,i,j);
		}
		return 0;
	}

	float sink_tweight(IplImage* img, int i, int j){

		if ( is_background(j,i) ) {
			return K_factor(img);
		}
		else if ( is_foreground(i,j) ) {
			return 0;
		}
		else{
			return lamda * Pr_obj(img,i,j);
		}
		return 1;
	}

	float weight(IplImage* img, int node1, int node2){
		int x1,y1;
		int x2,y2;
		node_2_pixel(node1,x1,y1);
		node_2_pixel(node2,x2,y2);
		return B_pq(img, x1,y1,x2,y2);
	}

	int pixel_2_node(int i, int j){
		return i * w_img + j;
	}

	void node_2_pixel(int node, int &i, int &j){

		i = node/w_img;
		j = node%w_img;

	}

	static int num_nodes(IplImage* img){
		return img->width * img->height;
	}

	static int num_edges(IplImage* img){
		return img->width * img->height * 8;
	}

	int is_background(int i, int j){
		if ( !((i - background.p1.x)*(i - background.p2.x) < 0 &&
			 (j - background.p1.y)*(j - background.p2.y) < 0) ) {
			 return 1;
		}
		return 0;
	}

	bool is_foreground(int i, int j){
		return foreground_flag[i][j];
	}

	float B_pq(IplImage* img, int i1, int j1, int i2, int j2){

		float  a,b;
		float  f_res = 0;
		float* res = new float[n_features];

		for ( int i = 0 ; i < n_features ; i++ ) {

			res[i] = 0;

			int dim = models->at(i)->dim;

			int c1 = models->at(i)->cluster_idx[pixel_2_node(i1,j1)];
			int c2 = models->at(i)->cluster_idx[pixel_2_node(i2,j2)];

			for ( int j = 0 ; j < dim ; j++ ) {

	            a = CV_MAT_ELEM(*(models->at(i)->center),float,c1,j);
				b = CV_MAT_ELEM(*(models->at(i)->center),float,c2,j);

				res[i] = res[i] + ( a - b ) * ( a - b );

			}

			//f_res += models->at(i)->w_exp * exp((-res[i]*1.0) / (2*models->at(i)->sigma*models->at(i)->sigma));
			f_res += models->at(i)->w_exp * exp((-res[i]*1.0) / (2*200*200));

		}

		return  f_res;

	}


	float K_factor ( IplImage* img ){

		if ( K == -1 ) {
			float K_buf = 0;
			int m,n;
			for ( int i = 0 ; i < h_img ; i++ ) {
				for ( int j = 0 ; j < w_img ; j++ ) {
					if ( !is_background(j,i) && !is_foreground(i,j) ) {
						K_buf = 0;
						for ( int x = -1 ; x <= 1 ; x++ ){
							for ( int y = -1 ; y <= 1 ; y++ ){
								m = i + x;
								n = j + y;
								if ( m < h_img && n < w_img ) {
									if ( x != 0 || y != 0 ) {
										K_buf += B_pq(img,i,j,m,n);
									}
								}
							}
						}
						if ( K < K_buf ) K = K_buf ;
					}
				}
			}
		}
		K = K + 1;
		return K;
	}


	float Pr_bkg(IplImage* img,int i, int j){

		float res = 0 ;

		for ( int i = 0 ; i < n_features ; i++ ) {

			res += models->at(i)->w_prob*-log(models->at(i)->bkg_hist.hist[models->at(i)->cluster_idx[pixel_2_node(i,j)]]);
		}

		//printf ("Pr_bkg = %f\n" ,res);

		return res;

	}

	float Pr_obj(IplImage* img,int i, int j){

		float res = 0 ;

		for ( int i = 0 ; i < n_features ; i++ ) {

			res += models->at(i)->w_prob*-log(models->at(i)->obj_hist.hist[models->at(i)->cluster_idx[pixel_2_node(i,j)]]);
		}

		//printf ("Pr_obj = %f\n" ,res);

		return res;
	}

	void segment_image(IplImage* img){

		int nchannels = img->nChannels;
		int step      = img->widthStep;

		IplImage* seg_bkg = cvCreateImage(cvSize(w_img,h_img),8,nchannels);
		IplImage* seg_obj = cvCreateImage(cvSize(w_img,h_img),8,nchannels);

		/* setup the pointer to access image data */
		uchar *data = ( uchar* )img->imageData;
		uchar *data_bkg = ( uchar* )seg_bkg->imageData;
		uchar *data_obj = ( uchar* )seg_obj->imageData;

		float flow = g->maxflow();
		printf("Flow = %f\n", flow);

		int node = 0;

		bool label[MAX_IMG_SIZE][MAX_IMG_SIZE];

		for(int i = 0 ; i < h_img ; i++ ) {
			for(int j = 0 ; j < w_img ; j++ ) {

				if ( g->what_segment(node) == GraphType::SOURCE  ){

					data_obj[i*step + j*nchannels + 0] = data[i*step + j*nchannels + 0];
					data_obj[i*step + j*nchannels + 1] = data[i*step + j*nchannels + 1];
					data_obj[i*step + j*nchannels + 2] = data[i*step + j*nchannels + 2];

					data_bkg[i*step + j*nchannels + 0] = 255;
					data_bkg[i*step + j*nchannels + 1] = 255;
					data_bkg[i*step + j*nchannels + 2] = 255;

					label[i][j] = true;

				}
				else{

					data_bkg[i*step + j*nchannels + 0] = data[i*step + j*nchannels + 0];
					data_bkg[i*step + j*nchannels + 1] = data[i*step + j*nchannels + 1];
					data_bkg[i*step + j*nchannels + 2] = data[i*step + j*nchannels + 2];

					data_obj[i*step + j*nchannels + 0] = 255;
					data_obj[i*step + j*nchannels + 1] = 255;
					data_obj[i*step + j*nchannels + 2] = 255;

					label[i][j] = false;

				}
				node++;
			}
		}


		int x,y;
		// Correctly label the boundary pixels ... Label as foreground if any of the neighbors is foreground
		for(int i=0;i<h_img;i++)
		for(int j=0;j<w_img;j++)
		{
			bool thisLabel = label[i][j];
			if (thisLabel==false)
			{
				int total = 0;
				int lab   = 0;
				for (int m = -1 ; m <= 1; m++ ) {
					for (int n = -1 ; n <= 1; n++ ) {
						x = i + m ;
						y = j + n ;
						total = total + 1;
						if ( !(x < 0 || y < 0 || x >= h_img || y >= w_img) ) {
							int node = pixel_2_node(x,y);
							if ( g->what_segment(node) == GraphType::SOURCE ) {
								lab = lab + 1;
							}
						}
					}
				}
				if ( (lab*1.0/total)>0.5 ) {
					data_obj[i*step + j*nchannels + 0] = data[i*step + j*nchannels + 0];
					data_obj[i*step + j*nchannels + 1] = data[i*step + j*nchannels + 1];
					data_obj[i*step + j*nchannels + 2] = data[i*step + j*nchannels + 2];

					data_bkg[i*step + j*nchannels + 0] = 255;
					data_bkg[i*step + j*nchannels + 1] = 255;
					data_bkg[i*step + j*nchannels + 2] = 255;
				}
			}
		}

		cvNamedWindow("BKG_IMG");
		cvNamedWindow("OBJ_IMG");

		cvShowImage("BKG_IMG",seg_bkg);
		cvShowImage("OBJ_IMG",seg_obj);

		cvWaitKey(0);

		cvDestroyWindow("BKG_IMG");
		cvDestroyWindow("OBJ_IMG");

		cvReleaseImage(&seg_bkg);
		cvReleaseImage(&seg_obj);
	}

	~Fusion_Framework(){

		delete &background;
		delete &foreground;

		for ( int i = 0 ; i < MAX_IMG_SIZE ; i++ ) {
			delete foreground_flag[i];
		}
		delete []foreground_flag;

		for ( int i = 0 ; i < n_features ; i++ ) {

			models->at(i)->~Feature_Model();

		}

		delete models;
		delete g;

	}

};


#endif /* FUSION_FRAMEWORK_H_ */
