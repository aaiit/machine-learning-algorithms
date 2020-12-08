#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;


mat Ls(int i,const mat& X, const mat& y, const mat& theta)
{
	int n=X.n_rows;
	return sum(y[i]-theta.t()*X(i))/n;
}

void adaline(const mat X,const mat y,mat& theta,string file_name)
{
	theta.randu();
	int t=1 , n = X.n_rows;
	mat l ;

	mat e = Ls(0,X,y,theta) ;
	vector<double> J_history ={e[0]} ;

	int Tmax = 100;
	
	for(int t=0;t<Tmax;t++)
	{
		for(int i=0;i<n;i++)
		{
			 
			mat e = Ls(i,X,y,theta) ;
			e.print("J :");
			J_history.push_back(e[0]);
			if(e[0] !=0)
				theta=theta +2*e[0]*X(i);

		}
	}

	ofstream coutput_file("costs/"+file_name);
	for (const auto &e : J_history) coutput_file << e << " ";
		

	ofstream Woutput_file("W/"+file_name);
	for (const auto &e : theta) Woutput_file << e << " ";

}