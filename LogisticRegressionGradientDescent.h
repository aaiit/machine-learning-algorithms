#include <armadillo>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace arma;

mat sigmoid(mat z)
{
	return 1/(1+exp(-z));
} 

mat predict(mat features, mat theta)
{
	return sigmoid(features*theta);
}

mat computeCost(const mat& X, const mat& y, const mat& theta)
{
	mat J;
	int m;
	m = y.n_rows;
	mat h=predict(X,theta);
	J = arma::sum(-y*log(h)-(1-y)*log(1-h))/m ;
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

	//vec J_history = arma::zeros<vec>(num_iters) ;
	for (iter = 0; iter < num_iters; iter++)
	{
		gradient = trans(sigmoid(X*theta))*(1-sigmoid(X*theta)) ;
		theta = theta-alpha*gradient ;
		//J_history(iter) = computeCost(X, y, theta)(0) ;

		mat J = computeCost(X, y, theta);
		J.print("J:");
	
	}
	//J_history.print("J_history");
}

