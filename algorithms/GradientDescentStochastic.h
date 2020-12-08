#include <armadillo>
#include <iostream>
#include <stdio.h>

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


void stochasticgradientDescent(const mat&    X,
                     const mat&    Y,
                     mat&    theta,
                     mat  computeCost(const mat& X, const mat& y, const mat& theta),
                     mat  computeGradient(const mat& X, const mat& y, const mat& theta),
                     string file_name ,
                     int batch_size,
                     int iterations)
{
	int it=0;
	int m,n;
	m = Y.n_rows;
	n = theta.n_elem;
	mat gradient(n,1),old_gradient(n,1),dg;
	gradient.ones();

	vector<double> J_history;
	
	while(iterations--)
	{
		old_gradient=gradient;

		//  Select Batch
		int p=it%m,q=(it+batch_size)%m ; 
		if(p>q){it++;continue;}

		mat x=X.rows(p, q) , y= Y.rows(p ,q);

		gradient = computeGradient(x,y,theta) ;
		double  alpha = armijo(x,y,theta,gradient,computeCost);
		theta = theta-alpha*gradient ;
		mat J = computeCost(x, y, theta);
		J.print("J: ");
		J_history.push_back(J[0]);
	
	}
	ofstream output_file("costs/"+file_name);
    for (const auto &e : J_history) output_file << e << " ";

	ofstream Woutput_file("W/"+file_name);
	for (const auto &e : theta) Woutput_file << e << " ";
}

