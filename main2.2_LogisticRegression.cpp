// Logistic Regression

#include <iostream>
#include <vector>
#include "rapidcsv.h"
#include "GradientDescent.h"

using namespace std;

mat sigmoid(mat z)
{
	return 1/(1+exp(-z));
}

mat computeCost(const mat& X, const mat& y, const mat& theta)
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

mat computeGradient(const mat& X, const mat& y, const mat& theta)
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
   	
	mat theta = arma::zeros<vec>(n+1);		
	
	gradientDescent(X, y, theta,computeCost, computeGradient, "binary") ;
	theta.print("Theta found by gradient descent") ;
	
	return 0;
}
