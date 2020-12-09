#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;



mat LeastSquaesCost(const mat& X, const mat& y, const mat& theta)
{
	mat J;
	int m;
	m = y.n_rows;
	J = sum((pow(((X*theta)-y), 2))/m) ;
	return J;
}

mat LeastSquaesGradient(const mat& X, const mat& y, const mat& theta)
{
	int m= X.n_rows;
	return 2*arma::trans(X)*(X*theta-y)/m;
}

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


double armijo(const mat& X, const mat& y, const mat& theta, mat gradient,mat computeCost(const mat& X, const mat& y, const mat& theta)
	) {
	// Armijo Hyperparameters
	
     double eps = .001;
     double eta = 10;

    double alpha = 10.e-15;
    mat yy=-trans(gradient)*gradient;
    double phiPrZero =yy[0]; 
    double phiZero = computeCost(X,y,theta)[0];
    while (computeCost(X,y,theta-alpha*gradient)[0] <= alpha * phiPrZero * eps + phiZero)
        alpha *= eta;

    return alpha / eta;
}

// double doldstein(const mat& X, const mat& y, const mat& theta, mat gradient,mat computeCost(const mat& X, const mat& y, const mat& theta)) 
// {
// 	double alpha = 1;

// 	double phi_alpha = computeCost(X,y,theta + alpha*gradient);

// 	double d_phi_alpha = 1e5*(computeCost(X,y,theta + (alpha + 1e-5)*gradient) - computeCost(X,y,theta + alpha*gradient) );


// }


// double wolfe(const mat& X, const mat& y, const mat& theta, mat gradient,mat computeCost(const mat& X, const mat& y, const mat& theta)) 
// {
// 	double alpha = 10.e-15 , eps = 0.25;
// 	double eta = 10;

// 	mat d_phi_alpha_0 = 1e5*(computeCost(X,y,theta + ( 1e-5)*gradient) - computeCost(X,y,theta + alpha*gradient) );

// 	mat d_phi_alpha = 1e5*(computeCost(X,y,theta + (alpha + 1e-5)*gradient) - computeCost(X,y,theta + alpha*gradient) );

// 	while(d_phi_alpha[0] >=eps * d_phi_alpha_0[0]) alpha*=eta;
// 	return alpha/eta;

// }