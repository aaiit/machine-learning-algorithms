#include <armadillo>
#include <iostream>

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

void perceptron(const mat X,const mat y,mat& theta,string file_name)
{
	theta.randu();
	int t=1 , n = X.n_rows;
	mat l ;

	vector<double> J_history;

	do
	{
		for(int i=1;i<=n;i++)
		{
			mat p = theta.t()*X(i);
			if(p[0]*y[i] < 0)
			{
				theta = theta +y[i]*X[i];
				t++;
			}
		}
		l = Ls(X,y,theta);
		l.print("Ls "+to_string(t)+":");
		J_history.push_back(l[l[0]]);

	}while(l[0]!=0);

	ofstream coutput_file("costs/"+file_name);
	for (const auto &e : J_history) coutput_file << e << " ";
		

	ofstream Woutput_file("W/"+file_name);
	for (const auto &e : theta) Woutput_file << e << " ";

}