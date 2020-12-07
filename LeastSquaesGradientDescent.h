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
	J = sum((pow(((X*theta)-y), 2))/m) ;
	return J;
}

double armijo(const mat& X, const mat& y, const mat& theta, mat gradient) {
	// Armijo Hyperparameters
	
    static double eps = .001;
    static double eta = 10;

    double alpha = 10.e-15;
    mat yy=-trans(gradient)*gradient;
    double phiPrZero =yy[0]; 
    double phiZero = computeCost(X,y,theta)[0];
    while (computeCost(X,y,theta-alpha*gradient)[0] <= alpha * phiPrZero * eps + phiZero)
        alpha *= eta;

    return alpha / eta;
}

void gradientDescent(const mat&    X,
                     const mat&    y,
                     mat&    theta,
                     string file_name)
{
	int iter;
	int m,n;
	m = y.n_rows;
	n = theta.n_elem;
	mat gradient(n,1),old_gradient(n,1),dg;
	gradient.ones();

	vector<double> J_history;
	do
	{
		old_gradient=gradient;
		gradient = 2*arma::trans(X)*(X*theta-y)/m ;
		double  alpha = armijo(X,y,theta,gradient);
		theta = theta-alpha*gradient ;
		mat J = computeCost(X, y, theta);
		J.print("J: ");
		J_history.push_back(J[0]);

		dg=sum(gradient-old_gradient);
	
	}while(abs(dg[0])>0.0000001);

	ofstream output_file("loss_history:"+file_name);
    for (const auto &e : J_history) output_file << e << " ";
}

