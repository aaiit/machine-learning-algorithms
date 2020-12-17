## What is it?
	Implementing some algorithms of machine learning used to solve classification and regression tasks
## Requirements
	The implementation based on Armadillo library
	$ sudo apt-get install libarmadillo-dev

## Command line
	$ git clone https://github.com/aaiit/machine-learning-algorithms.git
	$ cd machine-learning-algorithms
	$ g++ exemples/main****.cpp -o bin/exe -llapack -lblas -larmadillo

## What is included?

As of today, the following algorithms have been implemented:

- [x]  [Gradient Descent](algorithms/GradientDescent.h)
- [x]  [Newton](algorithms/Newton.h)
- [x]  [Perceptron](algorithms/Perceptron.h)
- [x]  [Adaline](algorithms/Adaline.h)
- [x]  [Pocket](algorithms/Pocket.h)
- [ ]  Dual-Form Perceptron


# Quasi-Newton
- [x]  BFGS 
- [x]  Broyden
- [x]  DFP
- [x]  SR1

# Stochastic Gradient
- [x]  [Gradient Descent Stochastic](algorithms/GradientDescentStochastic.h)
- [x]  [Momentum](algorithms/Momentum.h)
- [ ]  AdaGrad
- [ ]  RMSprop
- [ ]  Adadelta
- [ ]  NAG
- [ ]  Adam
- [ ]  AdaMax
- [ ]  Nadam
- [ ]  AMSGrad



## [Report](https://docs.google.com/document/d/1tDZneH6ZPYIAbiMrY0CzOF6ZCR4k3EEQJaOPl8QeyXo/edit?usp=sharing)
<!--
## References
1. https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
2. https://koriavinash1.github.io/ai/optimization/svm/Unconstrained-Optimization/
3. Calcul Hissien for logistic cost :https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function
3. Quasi Neuwton : http://thatdatatho.com/2019/08/07/newtons-method-bfgs-linear-regression/
4. wolfe rule : http://thatdatatho.com/2019/07/01/introduction-gradient-descent-line-search/

-->
