// Algoruthm : Logistic Regression
// Data : binary.csv
#include <iostream>
#include <vector>
#include "algorithms/GradientDescent.h"

using namespace std;


int main(int argc, char const *argv[])
{

	mat X, y ,theta;
	csv_to_xy("data/binary.csv", {"gre" ,"gpa","rank"}, "admit", X, y);


	// Fist case :theta initialized by zeros
	// theta.zeros();
	
	// Second case :theta initialized by ones
	// theta.ones();

	// Third case :theta initialized by Gaussian/normal distribution with μ = 0 and σ = 1
	theta.randn();

	// X.print("X :");

	gradientDescent(X, y, theta,logisticCost, logisticGradient, "binary_zero+rank","0.01") ;


	theta.print("Theta found by logistic gradient descent") ;
	


	return 0;
}
