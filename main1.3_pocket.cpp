// Algorithm : pocket

#include <bits/stdc++.h>
#include "algorithms/Pocket.h"
#include "includes/rapidcsv.h"
using namespace std;

void test(string s)
{
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

	pocket(X,y,theta,"binary-classification_pocket"+s);
}
int main(int argc, char const *argv[])
{
	//  x,y,label

	test("moon");
	test("blobs");
	
	return 0;
}