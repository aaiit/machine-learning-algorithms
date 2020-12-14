// Algorithm :Linear Regression
// Data : pop.csv

#include <iostream>
#include <vector>
#include "algorithms/GradientDescent.h"

using namespace std;


int main(int argc, char const *argv[])
{
	mat X, y,theta;
	csv_to_xy("data/pop.csv", {"X1" ,"X2" ,"X3" , "X4"}, "X5", X, y);

	gradientDescent(X, y, theta, LeastSquaesCost, LeastSquaesGradient, "pop") ;

	theta.print("Theta found by gradient descent:");



	return 0;
}
