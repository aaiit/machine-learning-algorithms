// Algorithm : pocket

#include <bits/stdc++.h>
#include "algorithms/Pocket.h"
using namespace std;


int main(int argc, char const *argv[])
{
	//  x,y,label
	mat X, y,theta;
	csv_to_xy("data/blobs.csv", {"x", "y"}, "label", X, y);
	pocket(X, y, theta, "pocket-blobs" + s);

	return 0;
}