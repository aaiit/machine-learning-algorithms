#pragma once
#ifndef BASE_H
#define BASE_H




#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wformat-security"


#include <armadillo>
#include <bits/stdc++.h>
#include "../includes/conio.h"
#include "../includes/rapidcsv.h"


using namespace std;
using namespace arma;



double LeastSquaesCost(const mat& X, const mat& y, const mat& parameters)
{
    vec tmp(X * parameters - y);
    tmp = dot(tmp, tmp);
    double s = sum(tmp);
    return s / y.n_rows;
}

mat LeastSquaesGradient(const mat& X, const mat& y, const mat& parameters)
{
    int m = X.n_rows;
    return 2 * arma::trans(X) * (X * parameters - y) / m;
}

mat sigmoid(mat z)
{
    return 1 / (1 + exp(-z));
}
double sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

double logisticCost(const mat& X, const mat& y, const mat& parameters)
{
    double loss = 0;
    int m = X.n_rows;
    for (int i = 0; i < m; i++)
    {
        double z = as_scalar(X.row(i) * parameters);

        loss -= y[i] * log(sigmoid(-z)) + (1 - y[i]) * log(1 - sigmoid(-z));
    }

    return loss / m;
}

mat logisticGradient(const mat& X, const mat& y, const mat& parameters)
{

    int m = X.n_rows;
    return 2 * arma::trans(X) * (sigmoid(X * parameters) - y) / m;

}

mat logisticHessian(const mat& X, const mat& y, const mat& parameters)
{
    int m = y.n_rows;
    mat D(m, m); D.zeros();
    for (int i = 0; i < m; i++) 
    {
        double z = as_scalar(X.row(i) * parameters);
        D(i, i) = sigmoid(-z)*(1-sigmoid(-z));
    }
    return trans(X) * D *X ;
}

mat LeastSquaesHessian(const mat& X, const mat& y, const mat& parameters)
{
    return 2*trans(X)*X;
}

double armijo(const mat& X, const mat& y, const mat& parameters, mat gradient, double computeCost(const mat& X, const mat& y, const mat& parameters)
             ) {
    // Armijo Hyperparameters

    double eps = .001;
    double eta = 10;

    double alpha = 10.e-15;
    mat yy = -trans(gradient) * gradient;
    double phiPrZero = yy[0];
    double phiZero = computeCost(X, y, parameters);
    while (computeCost(X, y, parameters - alpha * gradient) <= alpha * phiPrZero * eps + phiZero)
        alpha *= eta;

    return alpha / eta;
}


void csv_to_xy(string training_data, vector<string> x_labels, string y_label , mat&X, mat&y)
{
    rapidcsv::Document doc(training_data);

    int n = x_labels.size();

    vector<double> c[n + 1];
    for (int i = 0; i < n; i++) c[i] = doc.GetColumn<double>(x_labels[i]);


    int m = c[0].size();

    X.reshape(m, n + 1);
    y.reshape(m, 1);

    for (int j = 0; j < m; j++)
    {
        X(j, 0) = 1;
        for (int i = 0; i < n; i++) {
            X(j, i + 1) = c[i][j];
        }
    }

    c[n] = doc.GetColumn<double>(y_label);
    for (int j = 0; j < m; j++)y[j] = c[n][j];
}
#endif // BASE_H