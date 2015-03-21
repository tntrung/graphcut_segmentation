/*
 * Fusion-Cut (version 1.0) program.
 * this program is developed for interactive image segmentation application.
 * An implementation of "Feature Extraction using Graph-Cut on Fusion of Features".
 * @author: Tran Ngoc Trung
 * @email:  tntrung@fit.hcmus.edu.vn
 *
 * DESCRIPTION:
 *
 * This program uses min-cut/max-flow library of Yuri Boykov and Vladimir Kolmogorov.
 * "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision.", PAMI 2004.
 *
 * Program uses graph-cut on fusion of multiple feature to improve performance in
 * interactive segmentation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <highgui.h>
#include "util.h"
#include "fusion_framework.h"

#define	ARG_MSG		(char*)"./msg/arg_message.txt"

#define FLAG		int

#define LBUTTON_OFF	0
#define LBUTTON_ON 	1

#define HINT_BACKGROUND 0
#define HINT_FOREGROUND 1

#define BACKGROUND_LINE_STYLE cvScalar(0, 0, 255, 0), 2, 8, 0
#define FOREGROUND_LINE_STYLE 0.5, cvScalar(0, 255, 0, 0), 2, 8, 0

IplImage* original_img;
IplImage* hint_img;

ForegroundPoints foreground;
BackgroundRect   background;

CvPoint lbutton_xy;
FLAG    lbutton_flag;
FLAG    hint_flag;


void params_init(){

	lbutton_xy 	 = cvPoint(0,0);

	lbutton_flag = LBUTTON_OFF;

	hint_flag 	 = HINT_BACKGROUND;

}


void argument_disp(char* msg){

	if ( msg == NULL ){ printf("unknown error: plz, check error_message_file!") ; return; }

	FILE* fp = fopen(msg,"r");

	char c;

	while(!feof(fp)){
		c = fgetc(fp);
		printf("%c",c);
	}

	fclose(fp);

}

void mouseHandler(int event, int x, int y, int flags, void *param)

{

    switch(event) {

        /* left button down */

        case CV_EVENT_LBUTTONDOWN:

        	lbutton_xy.x = x;
        	lbutton_xy.y = y;
        	lbutton_flag = LBUTTON_ON;
            break;

        /* left button up */

        case CV_EVENT_LBUTTONUP:

        	lbutton_flag = LBUTTON_OFF;
			break;

        /* mouse move */

        case CV_EVENT_MOUSEMOVE:

        	if ( lbutton_flag == LBUTTON_ON ) {

        		if ( hint_flag == HINT_BACKGROUND ) {


        			background.p1.x = lbutton_xy.x;
        			background.p1.y = lbutton_xy.y;

        			background.p2.x = x;
        			background.p2.y = y;

        		}

        		else {

        			foreground.pts.push_back(cvPoint(y,x));
        			foreground.pts.push_back(cvPoint(y,x-1));
        			foreground.pts.push_back(cvPoint(y,x+1));
        			foreground.pts.push_back(cvPoint(y-1,x));
        			foreground.pts.push_back(cvPoint(y+1,x));

        		}

        	}

        	hint_img = cvCloneImage(original_img);

			if (background.p1.x > 0 && background.p1.y > 0) {
				cvRectangle(hint_img,
							background.p1,
							background.p2,
							BACKGROUND_LINE_STYLE);
			}

			for ( unsigned int i = 0 ; i < foreground.pts.size() ; i++ ) {
				cvCircle(hint_img,
						 cvPoint(foreground.pts.at(i).y,foreground.pts.at(i).x),
						 FOREGROUND_LINE_STYLE);
			}

			cvShowImage("image", hint_img);

        	break;

        /* right button down */

		case CV_EVENT_RBUTTONDOWN:

			hint_flag = 1 - hint_flag;

        	if ( hint_flag == HINT_BACKGROUND ){

        		fprintf(stdout, "hints on background...\n");

        	}
        	else{

        		fprintf(stdout, "hints on foreground...\n");

        	}

			break;
    }

}


void mouse_callback_register(char *info){

	original_img = cvLoadImage( info , 1 );

	cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);

	cvSetMouseCallback( "Image", mouseHandler, NULL );

	cvShowImage("Image", original_img);

	cvWaitKey(0);

}

void hint_and_save (char* info) {

	mouse_callback_register( info );

	char* hint_info = new char[strlen(info)+5];

	strcpy( hint_info, info );

	strcat( hint_info , ".hint" );

	FILE* fp = fopen ( hint_info , "w" );

	//background hints
	fprintf( fp, "%d %d %d %d\n", background.p1.x, background.p1.y, background.p2.x, background.p2.y);

	//foreground hints
	int n_tmp = foreground.pts.size();

	fprintf( fp, "%d\n", n_tmp );

	for ( int i = 0 ; i < n_tmp ; i++) {
		fprintf( fp, "%d %d\n", foreground.pts[i].x, foreground.pts[i].y);
	}

	fclose(fp);

}

void hint_and_segmentation( char* info ) {

	mouse_callback_register( info );

	printf( "[starting to segment image: %s...]\n" , info );

	Fusion_Framework* framework = new Fusion_Framework (
														Fusion_Framework::num_nodes(original_img),
														Fusion_Framework::num_edges(original_img),
														original_img->width,
														original_img->height,
														&background,
														&foreground
													  );


	for ( int i = 0 ; i < framework->n_features ; i++ ) {

		printf( "setting up %dth model...\n" , i + 1);

		framework->quantized_feature( original_img, i );

		printf("finished!\n");

	}

	printf("************************************************************\n\n");

	printf("building graph...\n");

	framework->buildFramework( original_img );

	printf("finished!\n");

	printf("************************************************************\n\n");

	printf("segmenting image...\n");

	framework->segment_image( original_img );

	printf("finished!\n");


}

int main(int argc, char* argv[]){

	/*
	 * command argument: <binary> <type>
	 *
	 *  <type>: 0 (display GUI for annotation)
	 *          1 (console test)
	 *          2 (display GUI for segmentation)
	 */

	if ( argc == 1 ) {

		argument_disp(ARG_MSG);
	}

	else {

		params_init();

		if( strcmp( argv[1], "0" ) == 0 ){

			if ( argv[2] == NULL ) { argument_disp(ARG_MSG); return 1;}

			hint_and_save( argv[2] );

		}

		else if ( strcmp( argv[1], "1" ) == 0 ){

			if ( argv[2] == NULL ) { argument_disp(ARG_MSG); return 1;}

			hint_and_segmentation( argv[2] );
		}

		else if ( strcmp( argv[1] , "2" ) == 0 ){

			if ( argv[2] == NULL ) { argument_disp(ARG_MSG); return 1;}

			//dataset_evalutation( argv[2] );

		}

		else {

			argument_disp(ARG_MSG);
		}

	}

	return 0;

}
