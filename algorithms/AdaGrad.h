#include "base.h"

void adagrad(const mat&    X,
                     const mat&    Y,
                     mat&    theta,
                     double  computeCost(const mat& X, const mat& y, const mat& parameters), // LeastSquaesCost or logisticCost
                     mat  computeGradient(const mat& X, const mat& y, const mat& parameters), // LeastSquaesGradient or logisticGradient
                     string costs_file = "costs.out",
                     string parameters_file = "parameters.out",
                     int batch_size,
                     int iterations,
                     double alpha =.001,
                     double ep = 1e-7 // the fuzz factor
                  )
{
	
	// double alpha = .001 ;
	// double ep = 1e-7; 


	int it=0;
	int m,n;
	m = Y.n_rows;
	n = theta.n_elem;
	mat V(n,1),dg,gradient;

	V.zeros(); 

	vector<double> J_history;
	
	while(iterations--)
	{
		//  Select Batch
		int p=it%m,q=(it+batch_size)%m ; 
		if(p>q){it++;continue;}

		mat x=X.rows(p, q) , y= Y.rows(p ,q);

		gradient = computeGradient(x,y,theta) ;
		V = V + dot(gradient,gradient); // Vomentum


		double  alpha = armijo(x,y,theta,V,computeCost);

		theta = theta-alpha*dot(pow(V + ep,-2) , gradient) ; // Adagrad updating W

		mat J = computeCost(x, y, theta);
		J.print("J: ");
		J_history.push_back(J[0]);
	
	}
	ofstream output_file(costs_file);
    for (const auto &e : J_history) output_file << e << " ";

	ofstream Woutput_file(parameters_file);
	for (const auto &e : theta) Woutput_file << e << " ";
}

