#include <cmath>
#include "common_functions.h"


double CommonFunctions::sigmod(double x){
	return 1.0/(1.0+exp(-x));
}


double CommonFunctions::crossEntropyLoss(Eigen::VectorXi y,Eigen::VectorXd h){
	Eigen::VectorXd y_d = y.cast<double>();
	int n = y_d.size();
	double loss;
	for(int i=0;i<n;i++){
		loss -= (y_d(i)*log2(h(i))+(1-y_d(i))*log2(1-h(i)));
	}
	return loss/n;
}