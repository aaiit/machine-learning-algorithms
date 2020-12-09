// Algoruthm :   Logistic Regression with Nonlinear Transformation 
// Data : microchips.csv

#include <iostream>
#include <vector>
#include "rapidcsv/rapidcsv.h"
#include "algorithms/GradientDescent.h"

using namespace std;

vector<pair<int,int>> PQ(int deg)
{
	vector<pair<int,int>> v;
	for(int p=0;p<=deg;p++)
	{
		for(int q=0; p+q<= deg;q++ )
		{
			v.push_back(make_pair(p,q));
		}
	}
	return v;
}

mat transformation(mat x,int np)
{
	vector<pair<int,int>> pq = PQ(np);
	int K = pq.size();

	cout<< "New Size : "<< K<<endl;
	mat X(x.n_rows , K);
	X.col(0).ones();

	for(int i=0;i< x.n_rows;i++)
	{
		for(int j=0;j<K;j++)
		{

			pair<int,int> ppqq = pq[j];
			int p=ppqq.first , q= ppqq.second;
			X(i,j)= pow(x[0],p)*pow(x[1],q);
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
	// X.print("X :");

    mat y(m, 1);

	for(int i=0;i<m;i++)y[i]=c3[i]; 

	mat theta = arma::zeros<vec>(PQ(np).size());


	gradientDescent(X, y, theta,LeastSquaesCost,LeastSquaesGradient, "microchips_tr"+to_string(np)) ;

	theta.print("Theta found by gradient descent:"); 

	return 0;
}
