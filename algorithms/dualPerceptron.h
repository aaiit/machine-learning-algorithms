#include <armadillo>
#include <iostream>
#include "base.h"

using namespace std;
using namespace arma;


double Predict(vec x,mat X, mat y,mat parameters)
{
	double s =0;
	for(int i=0;i<X.n_cols;i++)
	{
		s+= parameters[i]*y[i]*as_scalar(trans(X(i))*x);
	}
	return s>0?1:-1;
}

void dualperceptron(const mat X, const mat y, mat& theta, int Tmax = 100, string costs_file = "costs.out",
                    string parameters_file = "parameters.out",)
{
	int t = 1 , n = X.n_cols;

	theta.reshape(n, 1);
	theta.zeros();


	vector<double> J_history = {e[0]} ;


	while (Tmax--)
	{
		int t = Tmax % (n);

		if ( Predict(X(t),X,y,parameters) != y[t]) parameters[t]++;

	}

	ofstream f;
	f = ofstream(costs_file);
	for (const auto &e : J_history) f << e << " ";


	f = ofstream(parameters_file);
	for (const auto &e : theta) f << e << " ";

}