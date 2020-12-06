#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
    mat features(100, 1);

    // features << 6.110100 << endr
      
    for(int i=0;i<100;i++)
    {
        features << i+1<<endr;
    }
    mat targets(100, 1);

    // targets << 17.59200 << endr
    for(int i=0;i<100;i++)
    {
        targets << 40*(i+1)+50<<endr;
    }

    mat theta = ones(features.n_cols + 1);

    mat features_new = join_horiz(ones(features.n_rows), features);

    double alpha = 0.01;
    double con = alpha*(1.0 / features.n_rows);

    int j = 0;

    while (j < 1000){
        mat step_error = (features_new*theta - targets);
        step_error.print("iteration : ");
        theta = theta - con * (features_new.t() * step_error);
        j++;
    }

    theta.print("theta:");

    return 0;
}