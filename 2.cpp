#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{

    mat features(10, 2);

    features << 2104 << 3 << endr
         << 1600 << 3 << endr
         << 2400 << 3 << endr
         << 1416 << 2 << endr
         << 3000 << 4 << endr
         << 1985 << 4 << endr
         << 1534 << 3 << endr
         << 1427 << 3 << endr
         << 1380 << 3 << endr
         << 1494 << 3 << endr;

    mat m = mean(features, 0);
    mat s = stddev(features, 0, 0);

    int i,  j;

    //normalization
    for (i = 0; i < features.n_rows; i++)
    {
        for (j = 0; j < features.n_cols; j++)
        {
            features(i, j) = (features(i, j) - m(j))/s(j);
        }
    }

    mat targets(10, 1);

    targets << 399900 << endr
        << 329900 << endr
        << 369000 << endr
        << 232000 << endr
        << 539900 << endr
        << 299900 << endr
        << 314900 << endr
        << 198999 << endr
        << 212000 << endr
        << 242500 << endr;


    mat theta = ones(features.n_cols + 1);

    mat features_new = join_horiz(ones(features.n_rows), features);

    double alpha = 0.01;
    double con = alpha*(1.0 / features.n_rows);

    while (j < 20000){
        mat step_error = (features_new*theta - targets);
        theta = theta - con * (features_new.t() * step_error);
        j++;
    }

    cout << theta << endl;

    system("pause");

    return 0;
}