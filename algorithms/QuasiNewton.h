#include "base.h"

void quasinewton(const mat&    X,
                 const mat&    y,
                 mat&    parameters,
                 double  computeCost(const mat& X, const mat& y, const mat& parameters), // LeastSquaesCost or logisticCost
                 vec  computeGradient(const mat& X, const mat& y, const mat& parameters), // LeastSquaesGradient or logisticGradient
                 mat  computeHessian(const mat& X, const mat& y, const mat& parameters), //  logisticHessian or
                 string method = "BFGS", // BFGS or  BROYDEN  or DFP or SR1
                 string costs_file = "costs.out",
                 string parameters_file = "parameters.out",
                 string step = "armijo", //  double as string like "0.01" by default armijo rule
                 double tol = 1e-10)
{
	int it = 0;
	int m, n;
	m = y.n_rows;
	n = X.n_cols;

	parameters.reshape(n, 1);
	parameters.randu();

	mat gradient(n, 1), error, I = eye(n, n), _parameters, _gradient , s , dg;
	gradient.ones();

	vector<double> costs ;


	mat H = computeHessian(X, y, parameters);
	double ep = 1e-15;


	while (det(H) == 0)
	{
		H = ep * I + H;
		ep *= 1.2;
	}

	double alpha = atof(step.c_str());

	do
	{
		it++;

		_parameters = parameters;
		_gradient = gradient;

		gradient = computeGradient(X, y, parameters) ;
		if (step == "armijo") alpha =  armijo(X, y, parameters, gradient, computeCost);
		parameters = parameters -  alpha * H * gradient;

		// Compute Hessian
		s = parameters - _parameters;
		dg = gradient - _gradient;
		if (method == "BFGS")
		{
			double phi = 1 / as_scalar(trans(dg) * s);
			H = (I - phi * s * trans(dg)) * H * (I -  phi * s * trans(dg)) + phi * s * trans(s);
		}
		if (method == "BROYDEN")
		{
			H = H + (s - H * dg) * trans(s) * H * inv(trans(s) * H * dg);
		}
		if (method == "BFP")
		{
			H = H + s * trans(s) * inv(trans(s) * dg) - H * dg * trans(dg) * H * inv(trans(dg) * H * dg);
		}
		if (method == "SR1")
		{
			H = H + (s - H * dg) * trans(s - H * dg) * inv(trans(s - H * dg) * dg);
		}

		H.print("H" + to_string(it) );

		error = computeCost(X, y, parameters);
		error.print("Error" + to_string(it) );
		costs.push_back(error[0]);


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
	for (const auto &e : costs) coutput_file << e << " ";


	ofstream Woutput_file(parameters_file);
	for (const auto &e : parameters) Woutput_file << e << " ";


}



