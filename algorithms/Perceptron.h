#include <armadillo>
#include <iostream>
#include "base.h"

using namespace std;
using namespace arma;


mat Ls(mat X,mat y,mat theta)
{
	mat loss= zeros<vec>(1);
    mat t = X * theta;
    int n= X.n_rows;
    for(int i=0;i< n;i++) {
        if (y[i] * t[i] < 0)
            loss[0]++;
    }
    return loss / n;
}

ofstream f;

void perceptron(const mat X,const mat y,mat& theta,string file_name)
{
	int t=1 , n = X.n_cols;

	theta.reshape(n,1);
	theta.zeros();

	mat l(1,1),_l =zeros<vec>(1) ;

	mat e = Ls(X,y,theta) ;
	vector<double> J_history ={e[0]} ;
	
	int Tmax = 100;
	for(int t=0;t<=Tmax or t<=Tmax;t++)
	{

		for(int i=1;i<=n;i++)
		{
			mat p = theta.t()*X(i);
			if(p[0]*y[i] < 0)
			{
				theta = theta +y[i]*X(i);
				t++;
			}
		}
		_l=l;
		l = Ls(X,y,theta);
		l.print("Ls "+to_string(t)+":");
		J_history.push_back(l[0]);

	}//while(abs((l-_l)[0])> 0.00001);

	f = ofstream("costs/"+file_name);
	for (const auto &e : J_history) f << e << " ";
		

	f = ofstream("W/"+file_name);
	for (const auto &e : theta) f << e << " ";

}