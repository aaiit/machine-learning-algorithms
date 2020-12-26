// Algorithm : adaline


#include <bits/stdc++.h>
#include "algorithms/Adaline.h"
using namespace std;



int main(int argc, char const *argv[])
{
	//  x,y,label

	mat X, y,theta;
	csv_to_xy("datasets/blobs.csv", {"x", "y"}, "label", X, y);
	adaline(X, y, theta);
	theta.print("theta");
	return 0;
}