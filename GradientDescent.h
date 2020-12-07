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
	J = arma::sum((pow(((X*theta)-y), 2))/(2*m)) ;
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
	int m ;
	m = y.n_rows;

	//vec J_history = arma::zeros<vec>(num_iters) ;
	for (iter = 0; iter < num_iters; iter++)
	{
		delta = arma::trans(X)*(X*theta-y)/m ;
		theta = theta-alpha*delta ;
		//J_history(iter) = computeCost(X, y, theta)(0) ;

		mat J = computeCost(X, y, theta);
		J.print("J:");
	
	}
	//J_history.print("J_history");
}

// int main()
// {
// 	mat data;
	
// 	mat X(100, 1);      
//     for(int i=0;i<100;i++)
//     {
//         X[i]=i;
//     }

//     mat y(100, 1);
//     for(int i=0;i<100;i++)
//     {
//         y[i]= 40*(i)+50;
//     }

	
// 	int m = X.n_elem;
	
// 	mat theta = arma::zeros<vec>(2);
// 	vec X_One(m);
// 	X_One.ones();
// 	X.insert_cols(0, X_One);
	
// 	// X.print("X:");
// 	// y.print("y:");

// 	int iterations = 150000 ;
// 	double alpha = 0.00001 ;
	
	
// 	gradientDescent(X, y, alpha, iterations, theta) ;
// 	printf("Theta found by gradient descent: \n") ;
// 	printf("%f %f \n", theta(0), theta(1)) ;
	
// 	return 0;
// }
