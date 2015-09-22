/*
Created on 2015/09/14
Author: wepon, http://2hwp.com
Reference: http://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html
*/

#include <iostream>
#include <eigen3/Eigen/Dense>
using namespace Eigen;
int main()
{

  /* Matrix */
  Matrix2d m;  //2*2,double
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << "m:\n" << m << std::endl;


  //MatrixXd m1 =  MatrixXd::Random(3,3);  //Dynamic,double
  //MatrixXd m1 =  MatrixXd::Zero(3,3);
  //MatrixXd m1 =  MatrixXd::Ones(3,3);
  MatrixXd m1 =  MatrixXd::Identity(3,3);
  std::cout << "m1:\n" << m1 << std::endl;

  

  MatrixXd m2(2,2);  //Dynamic,double
  m2<<1,2,3,4;
  std::cout << "m2:\n" << m2.size() << std::endl;  //size: 4

  std::cout << "m2*m2:\n" << m2*m2 << std::endl;   //cross product

  


  int rows=5;  
  int cols=5;  
  MatrixXf m3(rows,cols);  
  m3<<( Matrix3f()<<1,2,3,4,5,6,7,8,9 ).finished(),
  MatrixXf::Zero(3,cols-3),MatrixXf::Zero(rows-3,3),
  MatrixXf::Identity(rows-3,cols-3);  
  std::cout << "m3=\n" << m3 << std::endl;  
  std::cout << "m3.rows: " << m3.rows() << std::endl;  
  std::cout << "m3.cols: " << m3.cols() << std::endl;

  std::cout << "m3.transpose():\n" << m3.transpose() << std::endl;
  std::cout << "m3.adjoint():\n" << m3.adjoint() << std::endl; 

  Matrix2d m4 =  Matrix2d::Constant(3.0);
  std::cout << "m4:\n" << m4 << std::endl;


  


  /* Vector */
  Vector2f v;  //2,float
  //Vector2d v;
  //VectorXd v(2);
  v(0) = 4.4;
  v(1) = v(0) - 1;
  std::cout << "v:\n" << v << std::endl;

  Vector2i vv;
  vv<<1,2;
  //std::cout<< "v-vv:\n"<<v-vv<<std::endl; // error

  Vector2f v1;  //2,float
  v1 << 4.0,8.0;
  std::cout << "v1:\n" << v1 << std::endl;

  std::cout << "v.*v1:\n" << v.dot(v1) << std::endl;   //dot product

  std::cout << "v1.norm():\n" << v1.norm() << std::endl;
  std::cout << "v1.squaredNorm():\n" << v1.squaredNorm() << std::endl;

  Matrix2f m5;
  m5<<1.0,2.0,3.0,4.0;

  MatrixXf m6(2,3);
  m6<<1.0,2.0,3.0,4.0,5.0,6.0;
  std::cout<< "m6:\n"<<m6<<std::endl;
  std::cout<< "m6:\n"<<m6.row(0)<<std::endl;  //1 2 3 


  


  /* Array */
  Array4i v2;
  v2<<1,2,3,4;
  std::cout << "v2:\n" << v2 << std::endl;

  MatrixXd m8(2,3);
  std::cout<< m8.rows() << " " << m8.cols() << std::endl; 



  MatrixXd X(10,2);
  X<<1.0,0.8,2.0,1.7,3.0,2.5,4.0,3.6,5.0,4.9,
     1.0,1.2,2.0,2.5,3.0,3.4,4.0,4.5,5.0,6.0;
  MatrixXd X_new(X.rows(),X.cols()+1);
  X_new<<X,MatrixXd::Ones(X.rows(),1);
  std::cout << "X:\n" << X_new <<  std::endl << X_new.rows() << " " << X_new.cols() <<  std::endl;


}
