#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wformat-security"


#include <armadillo>
#include <bits/stdc++.h>
#include "../includes/conio.h"


using namespace std;
using namespace arma;



double LeastSquaesCost(const mat& X, const mat& y, const mat& parameters)
{
    vec tmp(X*parameters-y);
    tmp = dot(tmp,tmp);
	double s= sum(tmp);
	return s/y.n_rows;
}

mat LeastSquaesGradient(const mat& X, const mat& y, const mat& parameters)
{
	int m= X.n_rows;
	return 2*arma::trans(X)*(X*parameters-y)/m;
}

mat sigmoid(mat z)
{
	return 1/(1+exp(-z));
}
double sigmoid(double z)
{
	return 1/(1+exp(-z));
}

double logisticCost(const mat& X, const mat& y, const mat& parameters)
{
	double loss=0;
	int DATA_SIZE = X.n_rows;
    for(int i=0;i<DATA_SIZE;i++)
    {
        double temp = as_scalar(X.row(i)*parameters);
        loss = loss + -1*log(1/(1+exp(-1*temp*y(i))));
        // loss -= y(i)*log(sigmoid(temp)) +(1-y(i))*log(1-sigmoid(temp)); 
    }
    loss = loss/DATA_SIZE;
    return loss;
}

mat logisticGradient(const mat& X, const mat& y, const mat& parameters)
{

    mat gradient(parameters.n_elem,1);
    int DATA_SIZE = X.n_rows;
    for(int i=0;i<DATA_SIZE;i++)
    {
        double w_x = as_scalar(X.row(i)*parameters);

	    double temp0 = (1/(1+std::exp(y[i]*w_x)))*(-y[i]);
        gradient += trans(X.row(i)*temp0);
    }

    gradient = gradient/DATA_SIZE;

    return gradient;

    // int m= X.n_rows;
    // return 2*arma::trans(X)*(sigmoid(X*parameters)-y)/m;
}


double armijo(const mat& X, const mat& y, const mat& parameters, mat gradient,double computeCost(const mat& X, const mat& y, const mat& parameters)
	) {
	// Armijo Hyperparameters
	
     double eps = .001;
     double eta = 10;

    double alpha = 10.e-15;
    mat yy=-trans(gradient)*gradient;
    double phiPrZero =yy[0]; 
    double phiZero = computeCost(X,y,parameters);
    while (computeCost(X,y,parameters-alpha*gradient) <= alpha * phiPrZero * eps + phiZero)
        alpha *= eta;

    return alpha / eta;
}

