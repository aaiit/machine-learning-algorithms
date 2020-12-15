#include "base.h"

void gradientDescent(const mat&    X,
                     const mat&    y,
                     mat&    parameters,
                     double  computeCost(const mat& X, const mat& y, const mat& parameters), // LeastSquaesCost or logisticCost
                     mat  computeGradient(const mat& X, const mat& y, const mat& parameters), // LeastSquaesGradient or logisticGradient
                     string costs_file = "costs",
                     string parameters_file = "parameters",
                     string step = "armijo", //  double as string like "0.01" by default armijo rule
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

	vector<double> training_error_history , testing_error_history;
	double alpha = atof(step.c_str());
	do
	{
		it++;

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

	} while (norm(gradient) > tol );

	ofstream coutput_file(costs_file);
	for (const auto &e : training_error_history) coutput_file << e << " ";
	
	ofstream Woutput_file(parameters_file);
	for (const auto &e : parameters) Woutput_file << e << " ";


}
