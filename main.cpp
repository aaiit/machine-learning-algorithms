#include <bits/stdc++.h>
using namespace std;

vector<pair<int,int>> pq(int deg)
{
	vector<pair<int,int>> v;
	for(int p=0;p<=deg;p++)
	{
		for(int q=0; p+q< deg;q++ )
		{
			v.push_back(make_pair(p,q));
		}
	}
	return v;
}

int main(int argc, char const *argv[])
{
	//  p + q <= deg
	cout<<"size of Q(6) = "<<pq(6).size()<<endl;
	for(auto e:pq(6))cout<<e.first<<" "<<e.second<<endl;
	return 0;
}