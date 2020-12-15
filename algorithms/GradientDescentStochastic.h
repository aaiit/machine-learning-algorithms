#include "base.h"

void stochasticgradientdescent(const mat&    X,
                               const mat&    Y,
                               mat&    parameters,
                               double  computeCost(const mat& X, const mat& y, const mat& parameters), // LeastSquaesCost or logisticCost
                               mat  computeGradient(const mat& X, const mat& y, const mat& parameters), // LeastSquaesGradient or logisticGradient
                               string costs_file = "costs",
                               string parameters_file = "parameters",
                               int batch_size,
                               int iterations)
{
	int it = 0;
	int m, n;
	m = Y.n_rows;
	n = parameters.n_elem;
	mat gradient(n, 1), dg;
	gradient.ones();

	vector<double> J_history;

	while (iterations--)
	{
		//  Select Batch
		int p = it % m, q = (it + batch_size) % m ;
		if (p > q) {it++; continue;}

		mat x = X.rows(p, q) , y = Y.rows(p , q);

		gradient = computeGradient(x, y, parameters) ;
		double  alpha = armijo(x, y, parameters, gradient, computeCost);
		parameters = parameters - alpha * gradient ;
		mat J = computeCost(x, y, parameters);
		J.print("J: ");
		J_history.push_back(J[0]);

	}


	ofstream output_file(costs_file);
	for (const auto &e : J_history) output_file << e << " ";

	ofstream Woutput_file(parameters_file);
	for (const auto &e : parameters) Woutput_file << e << " ";
}

