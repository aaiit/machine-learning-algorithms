#include <armadillo>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace arma;

mat computeCost(const mat& X, const mat& y, const mat& theta)
{
	mat J;
	int m;
	m = y.n_rows;
	J = arma::sum((pow(((X*theta)-y), 2))/m) ;
	return J;
}

void gradientDescent(const mat&    X,
                     const mat&    y,
                           double  alpha,
                           int     num_iters,
                           mat&    theta)
{
	mat delta;
	int iter;
	int m;
	m = y.n_rows;

	//vec J_history = arma::zeros<vec>(num_iters) ;
	for (iter = 0; iter < num_iters; iter++)
	{
		delta = 2*arma::trans(X)*(X*theta-y)/m ;
		theta = theta-alpha*delta ;
		//J_history(iter) = computeCost(X, y, theta)(0) ;

		mat J = computeCost(X, y, theta);
		J.print("J:");
	
	}
	//J_history.print("J_history");
}

