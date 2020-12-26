#include "base.h"
#define DG(v) cout<<#v<<" "<<v<<endl;

class NN
{
	vector<vec> Theta;
	vector<int> layers;
	int nl;
	int nf;

	double alpha;
	vec gradient;
public:

	mat X;
	mat y;

	NN(vector<int> layers)
	{
		this->layers = layers;
		nl = layers.size();

		Theta.push_back(zeros<mat>(nf, layers[0]));

		for (int i = 0; i < nl - 1; i++)
		{
			Theta.push_back(zeros<mat>(layers[i], layers[i + 1]));
		}

		Theta.push_back(zeros<mat>(layers[nl], 1));

	}
	void setData(string dataset, vector<string> features , string label)
	{
		csv_to_xy(dataset, features, label, X, y);
		nf = X.n_cols;

	}

	mat sigmoid(mat z) {
		return 1 / (1 + exp(-z));
	}


	void train()
	{
		int it = 0;

		while (it<500)
		{
			it++;

			mat a[nl + 1];
			a[0] = X.t();

			for (int i = 1; i <= nl + 1; i++)
			{
				a[i] = sigmoid(Theta[i] * a[i - 1]);
			}

			for (int i = nl + 1; i > 0; i--)
			{
				gradient = logisticGradient2(X, y, a[i]);
				alpha =  0.01;
				Theta[i] = Theta[i] - alpha * gradient;
			}
		}
	}
};