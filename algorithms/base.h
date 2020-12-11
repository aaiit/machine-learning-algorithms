#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;



double LeastSquaesCost(const mat& X, const mat& y, const mat& parameters)
{
	double s= accu((pow(((X*parameters)-y), 2)));
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
    	vec e = parameters;
    	vec v = X.row(i)*e;

        double temp = as_scalar(v);
        loss = loss + -1*log(1/(1+exp(-1*temp*y(i))));
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

	    double temp0 = (1/(1+std::exp(y[i]*w_x)))*(-1*y[i]);
        gradient += trans(X.row(i)*temp0);
    }

    gradient = gradient/DATA_SIZE;

    return gradient;
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

