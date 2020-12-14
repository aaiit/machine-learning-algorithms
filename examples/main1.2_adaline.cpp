// Algorithm : adaline


#include <bits/stdc++.h>
#include "algorithms/Adaline.h"
using namespace std;



int main(int argc, char const *argv[])
{
	//  x,y,label

	mat X, y,theta;
	csv_to_xy("data/blobs.csv", {"x", "y"}, "label", X, y);
	adaline(X, y, theta, "daline-blobs", 100);

	return 0;
}