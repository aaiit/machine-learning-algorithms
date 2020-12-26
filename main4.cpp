// Algoruthm :   Logistic Regression with Nonlinear Transformation
// Data : microchips.csv

#include <iostream>
#include <vector>
#include "algorithms/GradientDescent.h"

using namespace std;

vector<pair<int, int>> PQ(int deg)
{
	vector<pair<int, int>> v;
	for (int p = 0; p <= deg; p++)
	{
		for (int q = 0; p + q <= deg; q++ )
		{
			v.push_back(make_pair(p, q));
		}
	}
	return v;
}

mat transformation(mat x, int np)
{
	vector<pair<int, int>> pq = PQ(np);
	int K = pq.size();

	cout << "New Size : " << K << endl;
	mat X(x.n_rows , K);
	X.col(0).ones();

	for (int i = 0; i < x.n_rows; i++)
	{
		for (int j = 0; j < K; j++)
		{

			pair<int, int> ppqq = pq[j];
			int p = ppqq.first , q = ppqq.second;
			X(i, j) = pow(x(i,1), p) * pow(x(i,2), q);
		}
	}
	return X;
}




int main(int argc, char const *argv[])
{
	int np;
	cout << "np :" << endl;
	cin >> np;

	mat X, y,theta;

	csv_to_xy("datasets/microchips.csv", {"x", "y"}, "z", X, y);

	X= transformation(X,np);

	gradientDescent(X, y, theta, logisticCost, logisticGradient) ;

	theta.print("Theta found :");
	return 0;
}
