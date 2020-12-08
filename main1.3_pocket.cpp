// Algorithm : pocket

#include <bits/stdc++.h>
#include "algorithms/Pocket.h"
#include "rapidcsv.h"
using namespace std;

int main(int argc, char const *argv[])
{
	// NS : R1590.csv S : R1352.csv   x,y,label

	string s= "R1590";
	rapidcsv::Document doc("data/" + s + ".csv");

	vector<float> c1 = doc.GetColumn<float>("x");
	vector<float> c2 = doc.GetColumn<float>("y");
	vector<float> c3 = doc.GetColumn<float>("label");

	int m = c1.size() , n = 2;
	mat X(m,n),y(m,1),theta(n,1);

	for(int i=0;i<m;i++)
	{
		X(i,0)=c1[i];
		X(i,1)=c2[i];
		y[i]=c3[i];
	}

	// X.print("X :");
	// y.print("y :");
	pocket(X,y,theta,"pocket"+s);
	return 0;
}