#include <armadillo>
#include <iostream>
#include "base.h"

using namespace std;
using namespace arma;


double Ls(mat X, mat y, mat theta)
{
	double loss = 0;
	mat t = X * theta;
	int n = X.n_rows;
	for (int i = 0; i < n; i++) {
		if (y[i] * t[i] < 0)
			loss++;
	}
	return loss / n;
}


void perceptron(const mat X, const mat y, mat& theta, int Tmax = 100, string costs_file = "costs.out",
                string parameters_file = "parameters.out")
{
	int t = 1 , n = X.n_cols;

	theta.reshape(n, 1);
	theta.zeros();

	double l;

	double e = Ls(X, y, theta) ;
	vector<double> J_history = {e} ;

	for (int t = 0; t <= Tmax ; t++)
	{

		for (int i = 1; i <= n; i++)
		{
			mat p = theta.t() * X(i);
			if (p[0]*y[i] < 0)
			{
				theta = theta + y[i] * X(i);
				t++;
			}
		}
		l = Ls(X, y, theta);
		J_history.push_back(l);

	}

	ofstream f;
	f = ofstream(costs_file);
	for (const auto &e : J_history) f << e << " ";


	f = ofstream(parameters_file);
	for (const auto &e : theta) f << e << " ";

}