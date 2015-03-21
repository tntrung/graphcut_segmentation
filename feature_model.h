/*
 * feature_model.h
 *
 *  Created on: Feb 18, 2010
 *      Author: tranngoctrung
 */

#ifndef FEATURE_MODEL_H_
#define FEATURE_MODEL_H_

#define MAX_HISTOGRAM_SIZE 100

typedef struct SimpHist {

	float hist[MAX_HISTOGRAM_SIZE];
	int   total;

	~SimpHist() {

		delete hist;

	}

};

class Feature_Model {

public:

	char   name[10];
	int    sub_window_size;

	CvMat* center;
	int    n_cluster;
	int    dim;

	float  sigma;
	float  lamda;

	int	   w_prob;
	int    w_exp;

	int*   cluster_idx;

	SimpHist bkg_hist;
	SimpHist obj_hist;

	Feature_Model ( ) {
	}

	void initImportantVariables( int w_img, int h_img ) {

		printf("%d,%d\n",n_cluster,dim);

		center = cvCreateMat( n_cluster, dim, CV_32FC1 );

		bkg_hist.total = 0;
		obj_hist.total = 0;

		for ( int i = 0 ; i < n_cluster ; i++ ) {

			bkg_hist.hist[i] = 0;
			obj_hist.hist[i] = 0;

		}

		cluster_idx = new int[w_img*h_img];

	}

	~Feature_Model(){

		delete name;

		cvReleaseMat(&center);

		delete &bkg_hist;
		delete &obj_hist;

	}

};


#endif /* FEATURE_MODEL_H_ */
