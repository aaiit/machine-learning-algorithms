#include <armadillo>
#include <bits/stdc++.h>
#include <conio.h>
using namespace std;

double LeastSquaesCost(const mat& X, const mat& y, const mat& parameters)
{
    vec tmp(X * parameters - y);
    tmp = dot(tmp, tmp);
    double s = sum(tmp);
    return s / y.n_rows;
}

mat LeastSquaesGradient(const mat& X, const mat& y, const mat& parameters)
{
    int m = X.n_rows;
    return 2 * arma::trans(X) * (X * parameters - y) / m;
}

mat sigmoid(mat z)
{
    return 1 / (1 + exp(-z));
}
double sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

double logisticCost(const mat& X, const mat& y, const mat& parameters)
{
    double loss = 0;
    int m = X.n_rows;
    for (int i = 0; i < m; i++)
    {
        double z = as_scalar(X.row(i) * parameters);

        loss -= y[i] * log(sigmoid(-z)) + (1 - y[i]) * log(1 - sigmoid(-z));
    }

    return loss / m;
}

mat logisticGradient(const mat& X, const mat& y, const mat& parameters)
{

    int m = X.n_rows;
    return 2 * arma::trans(X) * (sigmoid(X * parameters) - y) / m;

}


double armijo(const mat& X, const mat& y, const mat& parameters, mat gradient, double computeCost(const mat& X, const mat& y, const mat& parameters)
             ) {
    // Armijo Hyperparameters

    double eps = .001;
    double eta = 10;

    double alpha = 10.e-15;
    mat yy = -trans(gradient) * gradient;
    double phiPrZero = yy[0];
    double phiZero = computeCost(X, y, parameters);
    while (computeCost(X, y, parameters - alpha * gradient) <= alpha * phiPrZero * eps + phiZero)
        alpha *= eta;

    return alpha / eta;
}


void gradientDescent(const mat&    X,
                     const mat&    y,
                     mat&    parameters,
                     double  computeCost(const mat& X, const mat& y, const mat& parameters),
                     mat  computeGradient(const mat& X, const mat& y, const mat& parameters),
                     string costs_file = "costs",
                     string parameters_file = "parameters",
                     string step = "armijo", //  double as string like "0.01" by default armijo rule
                     double tol = 1e-10)
{
	mat _parameters;

	int it = 0;
	int m, n;
	m = y.n_rows;
	n = X.n_cols;

	parameters.reshape(n, 1);
	parameters.randu();

	mat gradient(n, 1), error;
	gradient.ones();

	vector<double> training_error_history , testing_error_history;
	double alpha = atof(step.c_str());
	do
	{
		it++; _parameters = parameters;

		gradient = computeGradient(X, y, parameters) ;

		if (step == "armijo") alpha =  armijo(X, y, parameters, gradient, computeCost);

		parameters = parameters - alpha * gradient ;

		error = computeCost(X, y, parameters);

		if (it % 1000 == 0)
		{	
			// clrscr();
			error.print("training error (" + to_string(it) + "): ");
			cout << "alpha : " << alpha << endl;
			cout << "escape : Stop a while loop" << endl;
			if (step != "armijo")cout << "+ : increase alpha \n- : decrease alpha" << endl;
		}


		training_error_history.push_back(error[0]);

		if ( kbhit() ) {

			// Stores the pressed key in ch
			char ch = getch();
			// Terminates the loop
			// when escape is pressed
			if (int(ch) == 27)break;
			if (ch == '+' and step!="armijo")alpha *= 1.2;
			if (ch == '-' and step!="armijo")alpha /= 1.2;

		}

	} while (norm(parameters - _parameters) > tol );

	ofstream coutput_file(costs_file);
	for (const auto &e : training_error_history) coutput_file << e << " ";
	
	ofstream Woutput_file(parameters_file);
	for (const auto &e : parameters) Woutput_file << e << " ";


}



