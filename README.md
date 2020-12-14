<!-- 
git add -A; git commit -m "moon" ; git push
https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
bfgs https://koriavinash1.github.io/ai/optimization/svm/Unconstrained-Optimization/
C H  : https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function
-->
## Requirements
	The implementation of logistic regression based on Armadillo library

## Command line
	$ git clone https://github.com/aaiit/machine-learning-algorithms.git
	$ cd machine-learning-algorithms
	$ g++ main_exemple.cpp -o bin/exe -llapack -lblas -larmadillo

## What is included?

As of today, the following algorithms have been implemented:

- [x]  [Perceptron](algorithms/Perceptron.h)
- [x]  [Adaline](algorithms/Adaline.h)
- [x]  [Pocket](algorithms/Pocket.h)

- [x]  [Gradient Descent](algorithms/GradientDescent.h)
- [x]  [Gradient Descent Stochastic](algorithms/GradientDescentStochastic.h)
- [x]  [Newton](algorithms/Newton.h)
- [x]  [Momentum](algorithms/Momentum.h)
