#ifndef __COMMON_FUNCTIONS_H__
#define __COMMON_FUNCTIONS_H__

#include <eigen3/Eigen/Dense>


class CommonFunctions{
public:
	// sigmod function, depend on <cmath> library
	static double sigmod(double x);
	static double crossEntropyLoss(Eigen::VectorXi y,Eigen::VectorXd h);

};



#endif