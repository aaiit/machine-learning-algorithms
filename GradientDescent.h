#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;



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
	mat gradient(n,1),old_gradient(n,1),dg;
	gradient.ones();

	vector<double> J_history;
	do
	{
		old_gradient=gradient;
		gradient = computeGradient(X,y,theta) ;
		double  alpha = armijo(X,y,theta,gradient,computeCost);
		theta = theta-alpha*gradient ;
		mat J = computeCost(X, y, theta);
		J.print("J: ");
		J_history.push_back(J[0]);


		it++;
		if(it%1000==0)
		{	
			ofstream coutput_file("costs/"+file_name);
		    for (const auto &e : J_history) coutput_file << e << " ";
		}

		dg=sum(gradient-old_gradient);
	
	}while(abs(dg[0])>0.0000001);

	ofstream coutput_file("costs/"+file_name);
	for (const auto &e : J_history) coutput_file << e << " ";
		

	ofstream Woutput_file("W/"+file_name);
	for (const auto &e : theta) Woutput_file << e << " ";
			
}

