// Algorithm : momentum  for linear regression 
// Data : cars.csv

#include <iostream>
#include <vector>
#include "rapidcsv/rapidcsv.h"
#include "algorithms/Momentum.h"

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


	momentum(X, y, theta,LeastSquaesCost,LeastSquaesGradient, "momentum-gradient-descent_cars" ,5,1000) ;

	theta.print("Theta found by momentum:");


	return 0;
}


