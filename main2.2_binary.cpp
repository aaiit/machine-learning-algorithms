// Algoruthm : Logistic Regression
// Data : binary.csv
#include <iostream>
#include <vector>
#include "rapidcsv/rapidcsv.h"
#include "algorithms/GradientDescent.h"

using namespace std;


int main(int argc, char const *argv[])
{
	rapidcsv::Document doc("data/binary.csv");

  	vector<float> x1 = doc.GetColumn<float>("admit");
  	vector<float> x2 = doc.GetColumn<float>("gre");
  	vector<float> x3 = doc.GetColumn<float>("gpa");
  	vector<float> x4 = doc.GetColumn<float>("rank");

  	int m= x1.size();
  	int n= 3;
	mat data;
	
	mat X(m, n+1);  
	for(int i=0;i<m;i++)
		{
			X(i,0)=1;
			X(i,1)=x1[i]; 
			X(i,2)=x2[i]; 
			X(i,3)=x3[i]; 
		}   
   
    mat y(m, 1);
	for(int i=0;i<m;i++)y[i]=x4[i];   
   	


	mat theta(n+1,1);

	// Fist case :theta initialized by zeros
	theta.zeros();
	
	// Second case :theta initialized by ones
	// theta.ones();}

	// Third case :theta initialized by Gaussian/normal distribution with μ = 0 and σ = 1
	//theta.randn();



	string s = "zeros";
	gradientDescent(X, y, theta,logisticCost, logisticGradient, "binary_"+s) ;
	theta.print("Theta found by logistic gradient descent") ;
	


	return 0;
}
