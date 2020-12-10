#include "base.h"

void newton(const mat&    X,
                     const mat&    y,
                     mat&    theta,
                     mat  computeCost(const mat& X, const mat& y, const mat& theta),
                     mat  computeGradient(const mat& X, const mat& y, const mat& theta),
                     mat  computeH(const mat& X, const mat& y, const mat& theta),
                     string file_name)
{
	int it=0;
	int m,n;
	m = y.n_rows;
	n = theta.n_elem;
	mat gradient(n,1),D(n,1),dg,error;
	gradient.ones();


	vector<double> training_error_history , testing_error_history;
	do
	{
		old_gradient=gradient;
		gradient = computeGradient(X,y,theta) ;

		if (det(D)>0)
		{
			D= inv(computeH(X,y,theta))*gradient;
		}
		else
		{
			break; // H + eps*I
		}
		double  alpha = armijo(X,y,theta,gradient,computeCost); // wolfe or armijo

		theta = theta-alpha*gradient ;

		error = computeCost(X, y, theta);
		error.print("training error: ");
		training_error_history.push_back(error[0]);


		it++;
		if(it%1000==0)
		{	
			ofstream coutput_file("costs/newton_"+file_name);
			for (const auto &e : training_error_history) coutput_file << e << " ";
	
			
			ofstream Woutput_file("W/"+file_name);
			for (const auto &e : theta) Woutput_file << e << " ";
		}

		dg=sum(gradient-old_gradient);
	
	}while(abs(dg[0])>0.0000001);


			
}

