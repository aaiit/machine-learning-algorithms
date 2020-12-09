// Algorithm : Linear Regression 
// Data : cars.csv

#include <iostream>
#include <vector>
#include "rapidcsv/rapidcsv.h"
#include "algorithms/GradientDescent.h"

using namespace std;



int main(int argc, char const *argv[])
{
	rapidcsv::Document doc("data/cars.csv");

  	vector<float> c1 = doc.GetColumn<float>("speed");
  	vector<float> c2 = doc.GetColumn<float>("dist");

  	int m= c1.size();
	int n=1;
	
	mat X(m, 2);  
	for(int i=0;i<m;i++)
	{
		X(i,0)=1;
		X(i,1)=c1[i];  
	}  
   
    mat y(m, 1);

	for(int i=0;i<m;i++)y[i]=c2[i]; 

	mat theta = arma::zeros<vec>(n+1);


	gradientDescent(X, y, theta,LeastSquaesCost,LeastSquaesGradient, "regression_gradient-descent_cars") ;

	theta.print("Theta found by gradient descent:");


	return 0;
}