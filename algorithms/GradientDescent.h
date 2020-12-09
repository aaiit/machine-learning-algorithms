#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;

mat LeastSquaesCost(const mat& X, const mat& y, const mat& theta)
{
	mat J;
	int m;
	m = y.n_rows;
	J = sum((pow(((X*theta)-y), 2))/m) ;
	return J;
}

mat LeastSquaesGradient(const mat& X, const mat& y, const mat& theta)
{
	int m= X.n_rows;
	return 2*arma::trans(X)*(X*theta-y)/m;
}

mat sigmoid(mat z)
{
	return 1/(1+exp(-z));
}

mat logisticCost(const mat& X, const mat& y, const mat& theta)
{
	int m;
	m = y.n_rows;

	mat J(1,1);
    mat z=X*theta;
    for(int i=0;i<m;i++){
    J[0] =J[0]+ log(1 + exp(-y[i] * sum(z[i]))); 
    } 
    return J / m;
}

mat logisticGradient(const mat& X, const mat& y, const mat& theta)
{
	int m= X.n_rows;
	int n= X.n_cols;
	mat gradient(n,1);

	for(int i=0;i<n;i++) 
		{
			gradient[i]= sum(dot(sigmoid(X*theta)-y,X.col(i)));
		}

	return gradient;
}


double armijo(const mat& X, const mat& y, const mat& theta, mat gradient,mat computeCost(const mat& X, const mat& y, const mat& theta)
	) {
	// Armijo Hyperparameters
	
    static double eps = .001;
    static double eta = 10;

    double alpha = 10.e-15;
    mat yy=-trans(gradient)*gradient;
    double phiPrZero =yy[0]; 
    double phiZero = computeCost(X,y,theta)[0];
    while (computeCost(X,y,theta-alpha*gradient)[0] <= alpha * phiPrZero * eps + phiZero)
        alpha *= eta;

    return alpha / eta;
}

void gradientDescent(const mat&    X,
                     const mat&    y,
                     mat&    theta,
                     mat  computeCost(const mat& X, const mat& y, const mat& theta),
                     mat  computeGradient(const mat& X, const mat& y, const mat& theta),
                     string file_name)
{
	int it=0;
	int m,n;
	m = y.n_rows;
	n = theta.n_elem;
	mat gradient(n,1),old_gradient(n,1),dg,error;
	gradient.ones();

	int training_size = m*.8;
	mat training_X = X.rows(0,training_size);
	mat training_y = y.rows(0,training_size);
	mat testing_X = X.rows(training_size,m-1);
	mat testing_y = y.rows(training_size,m-1);

	vector<double> training_error_history , testing_error_history;
	do
	{
		old_gradient=gradient;
		gradient = computeGradient(X,y,theta) ;
		double  alpha = armijo(X,y,theta,gradient,computeCost);
		theta = theta-alpha*gradient ;

		error = computeCost(training_X, training_y, theta);
		error.print("training error: ");
		training_error_history.push_back(error[0]);

		error = computeCost(testing_X, testing_y, theta);
		error.print("testing error: ");
		training_error_history.push_back(error[0]);

		it++;
		if(it%1000==0)
		{	
			ofstream coutput_file("costs/"+file_name);
		    for (const auto &e : training_error_history) coutput_file << e << " ";
		}

		dg=sum(gradient-old_gradient);
	
	}while(abs(dg[0])>0.0000001);

	ofstream coutput_file("costs/training_error_"+file_name);
	for (const auto &e : training_error_history) coutput_file << e << " ";
	
	coutput_file=ofstream("costs/testing_error_"+file_name);
	for (const auto &e : training_error_history) coutput_file << e << " ";
		

	ofstream Woutput_file("W/"+file_name);
	for (const auto &e : theta) Woutput_file << e << " ";
			
}

