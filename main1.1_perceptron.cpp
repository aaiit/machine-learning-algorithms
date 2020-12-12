// Algorithm : Perceptron learning

#include <bits/stdc++.h>
#include "algorithms/Perceptron.h"
using namespace std;


int main(int argc, char const *argv[])
{
	//  x,y,label

	mat X, y,theta;
	csv_to_xy("data/blobs.csv", {"x", "y"}, "z", X, y);
	perceptron(X, y, theta, "perceptron-blobs" );

	return 0;
}