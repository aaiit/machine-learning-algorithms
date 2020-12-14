// Algoruthm : Logistic Regression
// Data : binary.csv
#include <iostream>
#include <vector>
#include "algorithms/GradientDescent.h"
#include "algorithms/Newton.h"


using namespace std;


int main(int argc, char const *argv[])
{

	mat X, y ,theta;
	csv_to_xy("data/binary.csv", {"gre" ,"gpa","rank"}, "admit", X, y);
	// csv_to_xy("data/binary.csv", {"gre" ,"gpa"}, "admit", X, y);

	// Fist case :theta initialized by zeros
	// theta.zeros();
	
	// Second case :theta initialized by ones
	// theta.ones();

	// Third case :theta initialized by Gaussian/normal distribution with μ = 0 and σ = 1
	theta.randn();

	// X.print("X :");

	gradientDescent(X, y, theta,logisticCost, logisticGradient, "binary_rank","armijo") ;
	// newton(X, y, theta, logisticCost, logisticGradient,logisticHessian, "newton_binary")  ;


	theta.print("Theta found") ;
	


	return 0;
}
