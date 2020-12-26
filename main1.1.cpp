// Algorithm : Perceptron learning

#include <bits/stdc++.h>
#include "algorithms/Perceptron.h"
using namespace std;


int main(int argc, char const *argv[])
{
	//  x,y,label

	mat X, y,theta;
	csv_to_xy("datasets/blobs.csv", {"x", "y"}, "label", X, y);
	perceptron(X, y, theta);
	theta.print("theta :");
	return 0;
}