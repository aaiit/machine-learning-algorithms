#include <iostream>
#include <vector>
#include "rapidcsv.h"
#include "GradientDescent.h"

using namespace std;

mat LeastSquaesCost(const mat& X, const mat& y, const mat& theta)
{
	mat J;
	int m;
	m = y.n_rows;
	J = sum((pow(((X*theta)-y), 2))/m) ;
	return J;
}

mat computeGradient(const mat& X, const mat& y, const mat& theta)
{
	int m= X.n_rows;
	return 2*arma::trans(X)*(X*theta-y)/m;
}

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

	gradientDescent(X, y, theta,LeastSquaesCost,computeGradient,"cars");

	printf("Theta found by gradient descent: \n") ;
	printf("%f %f \n", theta(0), theta(1)) ;
	
	return 0;
}
