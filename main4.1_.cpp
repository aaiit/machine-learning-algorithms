// Algorithm :   Logistic Regression without Nonlinear Transformation 
// Data : microchips.csv

#include <iostream>
#include <vector>
#include "rapidcsv/rapidcsv.h"
#include "algorithms/GradientDescent.h"

using namespace std;



int main(int argc, char const *argv[])
{
	rapidcsv::Document doc("data/microchips.csv");

  	vector<float> x1 = doc.GetColumn<float>("x");
  	vector<float> x2 = doc.GetColumn<float>("y");
  	vector<float> x3 = doc.GetColumn<float>("z");

  	int m= x1.size();
  	int n= 2;
	mat data;
	
	mat X(m, n+1);  
	for(int i=0;i<m;i++)
		{
			X(i,0)=1;
			X(i,1)=x1[i]; 
			X(i,2)=x2[i]; 
		}   
   
    mat y(m, 1);
	for(int i=0;i<m;i++)y[i]=x3[i];   

	mat theta(n+1,1);
	theta.zeros();


	gradientDescent(X, y, theta,logisticCost, logisticGradient, "microchips0") ;
	theta.print("Theta found by logistic gradient descent") ;
	
	return 0;
}
