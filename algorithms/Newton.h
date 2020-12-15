#include "base.h"

void newton(const mat&    X,
            const mat&    y,
            mat&    parameters,
            double  computeCost(const mat& X, const mat& y, const mat& parameters),
            mat  computeGradient(const mat& X, const mat& y, const mat& parameters),
            mat  computeHessian(const mat& X, const mat& y, const mat& parameters),
            string costs_file = "costs",
            string parameters_file = "parameters",
            double tol = 1e-10)
{

	int it = 0;
	int m, n;
	m = y.n_rows;
	n = X.n_cols;

	parameters.reshape(n, 1);
	parameters.randu();

	mat gradient(n, 1), error;
	gradient.ones();

	vector<double> training_error_history ;
	do
	{
		it++;

		gradient = computeGradient(X, y, parameters) ;



		parameters = parameters -  inv(computeHessian(X, y, parameters)) * gradient;

		error = computeCost(X, y, parameters);


		error.print("training error (" + to_string(it) + "): ");


		training_error_history.push_back(error[0]);


		if ( kbhit() ) {

			// Stores the pressed key in ch
			char ch = getch();
			// Terminates the loop
			// when escape is pressed
			if (int(ch) == 27)
				break;


		}

	} while (norm(gradient) > tol );

	ofstream coutput_file(costs_file);
	for (const auto &e : training_error_history) coutput_file << e << " ";


	ofstream Woutput_file(parameters_file);
	for (const auto &e : parameters) Woutput_file << e << " ";


}



