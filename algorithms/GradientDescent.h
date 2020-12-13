#include "base.h"

void gradientDescent(const mat&    X,
                     const mat&    y,
                     mat&    parameters,
                     double  computeCost(const mat& X, const mat& y, const mat& parameters),
                     mat  computeGradient(const mat& X, const mat& y, const mat& parameters),
                     string file_name = "temp",
                     string step = "armijo",
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

	int training_size = m * .8;
	mat training_X = X;//.rows(0,training_size);
	mat training_y = y;//.rows(0,training_size);
	mat testing_X = X.rows(training_size, m - 1);
	mat testing_y = y.rows(training_size, m - 1);

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

		error = computeCost(testing_X, testing_y, parameters);

		testing_error_history.push_back(error[0]);



		if ( kbhit() ) {

			// Stores the pressed key in ch
			char ch = getch();
			// Terminates the loop
			// when escape is pressed
			if (int(ch) == 27)break;
			if (ch == '+')alpha *= 1.2;
			if (ch == '-')alpha /= 1.2;

		}

	} while (norm(parameters - _parameters) > tol );

	ofstream coutput_file("costs/training-error_" + file_name);
	for (const auto &e : training_error_history) coutput_file << e << " ";

	// coutput_file = ofstream("costs/testing-error_" + file_name);
	// for (const auto &e : testing_error_history) coutput_file << e << " ";

	ofstream Woutput_file("W/" + file_name);
	for (const auto &e : parameters) Woutput_file << e << " ";


}



