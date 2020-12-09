#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;



mat Ls(const mat& X, const mat& y, const mat& theta)
{
	mat J;
	int m;
	m = y.n_rows;
	J = sum((pow(((X*theta)-y), 2))/m) ;
	return J;
}
void adaline(const mat X,const mat y, mat& theta,const string file_name,const int Tmax)
{
	theta.randu();
	int t=1 , n = X.n_rows;
	mat l ;

	vector<double> J_history;
	
	for(int t=0;t<Tmax;t++)
	{
		for(int i=0;i<n;i++)
		{
			 
			mat e = sum(y[i]-theta.t()*X(i))/n;
			e.print("J :");
			if(e[0] !=0)
				theta=theta +2*e[0]*X(i);

		}
		J_history.push_back(Ls(X,y,theta)[0]);
	}

	ofstream coutput_file("costs/"+file_name);
	for (const auto &e : J_history) coutput_file << e << " ";
		

	ofstream Woutput_file("W/"+file_name);
	for (const auto &e : theta) Woutput_file << e << " ";

}