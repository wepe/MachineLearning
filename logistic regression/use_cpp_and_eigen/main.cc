#include <iostream>
#include <eigen3/Eigen/Dense>
#include "lr.h"

using namespace std;
using namespace Eigen;

int main(){

	//data prepare,10 samples
	MatrixXd X(10,2);
	X<<1.0,0.8,2.0,1.7,3.0,2.5,4.0,3.6,5.0,4.9,
	   1.0,1.2,2.0,2.5,3.0,3.4,4.0,4.5,5.0,6.0;
	VectorXi y(10);
	y<<0,0,0,0,0,1,1,1,1,1;

	//train and save the weights
	LR clf1 = LR(200,0.01,0.05,0.01);  //max_iter=200,alpha=0.01(learning rate),l2_lambda=0.05,tolerance=0.01
	clf1.fit(X,y);
	cout<<"weights:\n"<<clf1.getW()<<endl; 
	clf1.saveWeights("test.weights");

	//load the weights and predict
	LR clf2 = LR();
	clf2.loadWeights("test.weights");
	cout<<"Predict:\n"<<clf2.predict(X)<<endl;

	return 0;
}