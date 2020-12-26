#include "includes/conio.h"

class NN
{
	field<mat> Theta;
	vector<int> layers;
	int nl;
	int nf;
	int m;
	double alpha;
	vec g = ones<vec>(10);
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
	double Cost(const mat& y_pred)
	{

		return sum(dot(y - y_pred, (y - y_pred))) / m;
	}
	void train(int MAX_ITER, double tol = 1e-6)
	{
		int it = 0;

		while (norm(g) > tol and it < MAX_ITER)
		{
			it++;
			cout << it << endl;
			field<mat> a(nl);
			field<mat> sign(nl);

			a(0) = X.t();

			for (int i = 0; i < nl - 1; i++)
			{
				a(i + 1) = sigmoid(trans(Theta(i)) * a(i));
			}

			for (int i = nl - 1 ; i > 0; i--)
			{
				if ( i == nl - 1 ) {sign(i) = trans(a(i)) - y;}
				else
				{
					sign(i) =   sign(i + 1) * trans(Theta(i));
				}
				auto gradient = 2 * a(i - 1) * sign(i) / m;

				if (i == nl - 1)g = gradient;

				Theta(i - 1) = Theta(i - 1) - 0.01 * gradient;


			}

			// if ( kbhit() ) {
			// 	char ch = getch();
			// 	if (int(ch) == 27)break;
			// 	if (ch == '+' )alpha *= 1.2;
			// 	if (ch == '-' )alpha /= 1.2;

			// }
			
			
			// g.print("g :");
			cout << "loss :" << Cost(trans(a(nl - 1))) << endl;
		}
	}
};