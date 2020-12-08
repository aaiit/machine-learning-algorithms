#include <armadillo>
#include <iostream>

#define pp(x) cout<<#x<<" : "<<x.n_rows<<"#"<<x.n_cols<<endl;

using namespace std;
using namespace arma;


mat Ls(mat X,mat y,mat theta)
{
	mat loss= zeros<vec>(1);
    mat t = X * theta;
    int n= X.n_rows;
    for(int i=0;i< n;i++) {
        if (y[i] * t[i] < 0)
            loss++;
    }
    return loss / n;
}

void pocket(const mat X,const mat y,mat& theta,string file_name)
{
	theta.randu();
	int t=1 , n = X.n_rows;
	mat l1,l2 ;

	mat e = Ls(X,y,theta) ;
	vector<double> J_history ={e[0]} ;

	int Tmax = 100;

	for(int t=0;t<=Tmax;t++)
	{
		mat _theta = theta;

		int iw = Tmax%(theta.n_elem);

		for(int i=0;i<n;i++)
		{
			// pp(_theta(iw))pp(X(i,iw))

			double p = _theta(iw)*X(i,iw);
			if(p*y[i] < 0)
			{
				_theta(iw) = _theta(iw) +y[i]*X(i);
				t++;
			}

		}

		l1 = Ls(X,y,theta);
		l2 = Ls(X,y,_theta);

		if(l2[0] < l1[0]) theta= _theta;

		l1.print("Pocket -> Ls "+to_string(t)+":");
		J_history.push_back(l1[l1[0]]);

	}

	ofstream coutput_file("costs/"+file_name);
	for (const auto &e : J_history) coutput_file << e << " ";
		

	ofstream Woutput_file("W/"+file_name);
	for (const auto &e : theta) Woutput_file << e << " ";

}