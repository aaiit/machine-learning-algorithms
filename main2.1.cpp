// Algorithm : Linear Regression
// Data : cars.csv

#include <iostream>
#include <vector>
#include "algorithms/GradientDescent.h"

using namespace std;



int main(int argc, char const *argv[])
{

	mat X, y,theta;
	csv_to_xy("datasets/cars.csv", {"speed"}, "dist", X, y);


	gradientDescent(X, y, theta, LeastSquaesCost, LeastSquaesGradient) ;

	theta.print("Theta found by gradient descent:");


	return 0;
}
