// Algoruthm :   Logistic Regression with Nonlinear Transformation 
// Data : microchips.csv

#include <iostream>
#include <vector>
#include "rapidcsv.h"
#include "GradientDescent.h"

using namespace std;

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

vector<pair<int,int>> PQ(int deg)
{
	vector<pair<int,int>> v;
	for(int p=0;p<=deg;p++)
	{
		for(int q=0; p+q< deg;q++ )
		{
			v.push_back(make_pair(p,q));
		}
	}
	return v;
}

mat transformation(mat x,int np)
{
	vector<pair<int,int>> pq = PQ(np);
	mat X(x.n_rows , pq.size());
	X.col(0).ones();

	int k=0;
	for(int i=1;i<=np;i++)
	{
		mat xx=X.col(i-1);
		for(int j=0;j<x.n_rows;j++)
		{

			pair<int,int> ppqq = pq[k++];
			int p=ppqq.first , q= ppqq.second;
			X.col(i)[j]= pow(x[0],p)*pow(x[1],q);
		}
	}
	return X;
}
int main(int argc, char const *argv[])
{
	int np ;
	cout<<"NP (6 ) : "<<endl;
	cin>>np;

	rapidcsv::Document doc("data/microchips.csv");

  	vector<float> c1 = doc.GetColumn<float>("x");
  	vector<float> c2 = doc.GetColumn<float>("y");
  	vector<float> c3 = doc.GetColumn<float>("z");

  	int m= c1.size();
	
	mat X(m, np),x(m,2);  
	for(int i=0;i<m;i++)
	{
		x(i,0)=c1[i];  
		x(i,1)=c2[i];
	} 

	X=transformation(x,np); 

    mat y(m, 1);

	for(int i=0;i<m;i++)y[i]=c3[i]; 

	mat theta = arma::zeros<vec>(PQ(np).size());


	gradientDescent(X, y, theta,LeastSquaesCost,LeastSquaesGradient, "pressure_p"+to_string(np)) ;

	theta.print("Theta found by gradient descent:"); 


	return 0;
}