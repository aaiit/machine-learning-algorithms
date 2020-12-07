#include <armadillo>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace arma;

mat sigmoid(mat z)
{
	return 1/(1+exp(-z));
}

mat predict(mat X, mat theta) // p1 p2 p3 ...
{
	return sigmoid(X*theta);
}

mat computeCost(const mat& X, const mat& y, const mat& theta)
{
	mat J;
	int m;
	m = y.n_rows;
	J = arma::sum(log(1+exp(-y*w*x)))/m; 
	return J;
}

void gradientDescent(const mat&    X,
                     const mat&    y,
                           double  alpha,
                           int     num_iters,
                           mat&    theta)
{
	mat gradient;
	int iter;
	int m;
	m = y.n_rows;

	
	for (iter = 0; iter < num_iters; iter++) // todo while   Norme( gradient) < delta
	{
		gradient = trans(sigmoid(X*theta))*(1-sigmoid(X*theta)) ;
		theta = theta-alpha*gradient ; // todo search by armijo
		// J = computeCost(X, y, theta)(0) ;

		mat J = computeCost(X, y, theta);
		J.print("J:");
	
	}
}

