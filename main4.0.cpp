#include <iostream>
#include <vector>
#include "includes/rapidcsv.h"

#include "algorithms/BFGS.h"

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


	mat X, y,theta;

	csv_to_xy("datasets/pressure.csv", {"temperature" }, "pressure", X, y);

	
	bfgs(X, y, theta,LeastSquaesCost,LeastSquaesGradient,LeastSquaesHessian) ;

	theta.print("Theta :"); 


	return 0;
}