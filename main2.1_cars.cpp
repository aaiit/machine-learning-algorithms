// Algorithm : Linear Regression 
// Data : cars.csv

#include <iostream>
#include <vector>
#include "rapidcsv.h"
#include "algorithms/GradientDescent.h"

using namespace std;



int main(int argc, char const *argv[])
{
	rapidcsv::Document doc("data/cars.csv");

  	vector<float> cspeed = doc.GetColumn<float>("speed");
  	vector<float> cdist = doc.GetColumn<float>("dist");

  	int m= cspeed.size();
	int n=1;
	
	mat X(m, 2);  
	for(int i=0;i<m;i++)
	{
		X(i,0)=1;
		X(i,1)=cspeed[i];  
	}  
   
    mat y(m, 1);

	for(int i=0;i<m;i++)y[i]=cdist[i]; 

	mat theta = arma::zeros<vec>(n+1);


	gradientDescent(X, y, theta,LeastSquaesCost,LeastSquaesGradient, "cars") ;

	theta.print("Theta found by gradient descent:");


	return 0;
}
