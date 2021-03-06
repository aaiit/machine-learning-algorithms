#include <bits/stdc++.h>
#include "algorithms/base.h"
#include "algorithms/NN.h"
using namespace std;

// g++ ann.cpp -o bin/exe -llapack -lblas -larmadillo

int main(int argc, char const *argv[])
{

	NN myNet = NN({6,100,100,1});

	myNet.setData("datasets/airfoil_self_noise.csv", {"X1", "X2" , "X3", "X4" , "X5"}, "X6");

	myNet.train(50);

	// cout << myNet.predict(myNet.X.row(10))<<endl;
 //    cout << myNet.y(10,0);

	return 0;
}