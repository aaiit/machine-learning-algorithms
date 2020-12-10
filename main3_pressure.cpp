// Algorith : Polynomial Regression
// Data : pressure.csv

#include <iostream>
#include <vector>
#include "rapidcsv/rapidcsv.h"
#include "algorithms/GradientDescent.h"

using namespace std;

mat phi(vec x,int np)
{
	mat X(x.n_rows , np+1);
	X.col(0).ones();


	for(int i=1;i<=np;i++)
	{
		mat xx=X.col(i-1);
		for(int j=0;j<x.n_rows;j++)X.col(i)[j]=x[j]*X.col(i-1)[j];
	}
	return X;
}

int main(int argc, char const *argv[])
{
	int np ;
	cout<<"NP : "<<endl;
	cin>>np;

	rapidcsv::Document doc("data/pressure.csv");

  	vector<float> c1 = doc.GetColumn<float>("temperature");
  	vector<float> c2 = doc.GetColumn<float>("pressure");

  	int m= c1.size();
	
	mat X(m, np),x(m,1);  
	for(int i=0;i<m;i++)
	{
		x[i]=c1[i];  
	} 

	X=phi(x,np);

    mat y(m, 1);

	for(int i=0;i<m;i++)y[i]=c2[i]; 

	mat theta = arma::zeros<vec>(np+1);


	gradientDescent(X, y, theta,LeastSquaesCost,LeastSquaesGradient, "pressure_p"+to_string(np),1,"armijo") ;

	theta.print("Theta found by gradient descent:"); 


	return 0;
}
