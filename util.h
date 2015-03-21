/*
 * util.h
 *
 *  Created on: Feb 4, 2010
 *      Author: tranngoctrung
 */

#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#ifndef UTIL_H_
#define UTIL_H_

class ForegroundPoints {

public:

	vector<CvPoint> pts;

	ForegroundPoints(){}

	~ForegroundPoints(){
		pts.~vector();
	}

};

class BackgroundRect {

public:
	CvPoint p1;
	CvPoint p2;

	BackgroundRect(){

		p1 = cvPoint(0,0);

		p2 = cvPoint(0,0);

	}

	~BackgroundRect(){
	}

};

/*
char* itos_less(int i , int num_of_char) {

	int q = i;
	int idx = num_of_char, c = 0;
	char* res = new char[num_of_char];

	if (i == 0) {
		res[0]='0';
		res[1] = NULL;
	}

	else {

		while ( q > 0 && c < num_of_char ) {
			c++;
			res[--idx] = q % 10;
			q = q / 10;
			printf("%d - %d\n",res[idx],q);
		}

		for (int j = (num_of_char - c) ; j > 0 ; j-- ) {
			res[j-1] = res[j];
		}

		res[c] = NULL;

	}

	return res;

}
*/

#endif /* UTIL_H_ */
