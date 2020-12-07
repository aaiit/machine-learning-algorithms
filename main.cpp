#include <iostream>
#include <vector>
#include "rapidcsv.h"

int main()
{
  rapidcsv::Document doc("data/pop.csv");
  std::vector<float> col = doc.GetColumn<float>("X1");
  std::cout << "Read " << col.size() << " values." << std::endl;
}