#ifndef __LR_H__
#define __LR_H__

#include <eigen3/Eigen/Dense>
#include <string>

class LR{
public:
	LR(int max_i=100,double alp=0.01,double l2_lambda=0.05,double tolerance=0.01); //the params name can't be the same as the class member?
	~LR();
	void fit(Eigen::MatrixXd X,Eigen::VectorXi y);
	Eigen::VectorXd getW();
	Eigen::VectorXd predict_prob(Eigen::MatrixXd X);
	Eigen::VectorXi predict(Eigen::MatrixXd X);
	void saveWeights(std::string filename);
	void loadWeights(std::string filename);
private:
	Eigen::VectorXd W;
	int max_iter;
	double lambda;  //l2 regulization
	double tol;  // error tolence
	double alpha;
};



#endif