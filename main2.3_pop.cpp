// Algorithm :Linear Regression
// Data : pop.csv

#include <iostream>
#include <vector>
#include "rapidcsv.h"
#include "algorithms/GradientDescent.h"

using namespace std;


int main(int argc, char const *argv[])
{
	// data attribute : id,speed,dist
	rapidcsv::Document doc("data/pop.csv");

  	vector<float> x1 = doc.GetColumn<float>("X1");
  	vector<float> x2 = doc.GetColumn<float>("X2");
  	vector<float> x3 = doc.GetColumn<float>("X3");
  	vector<float> x4 = doc.GetColumn<float>("X4");
  	vector<float> x5 = doc.GetColumn<float>("X5");

  	int m= x1.size();

	mat data;
	
	mat X(m, 4);  
	for(int i=0;i<m;i++)
		{
			X(i,0)=x1[i]; 
			X(i,1)=x2[i]; 
			X(i,2)=x3[i]; 
			X(i,3)=x4[i]; 
		}   
   
    mat y(m, 1);
	for(int i=0;i<m;i++)y[i]=x5[i];   
   	
	mat theta = arma::zeros<vec>(5);
	vec X_One(m);
	X_One.ones();
	X.insert_cols(0, X_One);
	

	gradientDescent(X, y, theta,LeastSquaesCost,LeastSquaesGradient, "pop") ;

	theta.print("Theta found by gradient descent:");


	
	return 0;
}
