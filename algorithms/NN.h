// #include "base.h"
#define DG(v) cout<<#v<<" "<<v<<endl;

class NN
{
	field<mat> Theta;
	vector<int> layers;
	int nl;
	int nf;
	int m;
	double alpha;
	vec gradient;
public:

	mat X;
	mat y;

	NN(vector<int> layers)
	{
		this->layers = layers;
		nl = layers.size();

		Theta = field<mat>(nl - 1);
		for (int i = 0; i < nl - 1; i++)
		{
			Theta(i) = mat(layers[i], layers[i + 1]).randu();
		}


	}
	void setData(string dataset, vector<string> features , string label)
	{
		csv_to_xy(dataset, features, label, X, y);
		nf = X.n_cols;
		m = X.n_rows;
	}

	mat sigmoid(mat z) {
		return 1 / (1 + exp(-z));
	}
	mat sigmoidGradient(mat z) {
		return sigmoid(z) % (1 - sigmoid(z));
	}
	double predict(vec x)
	{
		for (auto w : Theta)x = w * x;
		return x[0];
	}

	void train()
	{
		int it = 0;

		while (it < 500)
		{
			it++;

			field<mat> a(nl+1);
			field<mat> sign(nl);

			a(0) = X.t();

			for (int i = 0; i < nl-1; i++)
			{
				cout << "forward " << i << endl;
				a(i + 1) = sigmoid(trans(Theta(i)) * a(i));
			}



			for (int i = nl ; i > 0; i--)
			{
				cout << "back " << i << endl;

				if ( i == nl ) {sign(i) = trans(a(i+1)) - y;}
				else
				{
					sign(i) =   Theta(i)*sign(i + 1);
				}
				gradient = 2 * a(i - 1) * sign(i) / m;
				// cout << "TS " << Theta(i).n_elem << endl;
				Theta(i) = Theta(i) - 0.01 * gradient;

			}
		}
	}
};