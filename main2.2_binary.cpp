// Algoruthm : Logistic Regression
// Data : binary.csv
#include <iostream>
#include <vector>
#include "rapidcsv.h"
#include "algorithms/GradientDescent.h"

using namespace std;

mat sigmoid(mat z)
{
	return 1/(1+exp(-z));
}

mat logisticCost(const mat& X, const mat& y, const mat& theta)
{
	int m;
	m = y.n_rows;

	mat J(1,1);
    mat z=X*theta;
    for(int i=0;i<m;i++){
    J[0] =J[0]+ log(1 + exp(-y[i] * sum(z[i]))); 
    } 
    return J / m;
}

mat logisticGradient(const mat& X, const mat& y, const mat& theta)
{
	int m= X.n_rows;
	int n= X.n_cols;
	mat gradient(n,1);

	for(int i=0;i<n;i++) 
		{
			gradient[i]= sum(dot(sigmoid(X*theta)-y,X.col(i)));
		}

	return gradient;
}



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
   	
   	cout<<"theta : zeros  or ones or normal";
   	string s;
   	cin>>s;

	mat theta(n+1,1);

   	if(s == "zeros" )
	{
		// Fist case :theta initialized by zeros
		theta.zeros();
	}
   	else if ( s == "ones" )
	{// Second case :theta initialized by ones
		theta.ones();}
   	else if(s =="normal")
   	{
			// Third case :theta initialized by Gaussian/normal distribution with μ = 0 and σ = 1
			theta.randn();
   	}
   	else theta.zeros();




	gradientDescent(X, y, theta,logisticCost, logisticGradient, "binary_"+s) ;
	theta.print("Theta found by logistic gradient descent") ;
	


	return 0;
}
