#include "base.h"

void nadam(const mat&    X,
                     const mat&    Y,
                     mat&    theta,
                     mat  computeCost(const mat& X, const mat& y, const mat& theta),
                     mat  computeGradient(const mat& X, const mat& y, const mat& theta),
                     string file_name ,
                     int batch_size,
                     int iterations)
{
	
	double alpha = 0.002, bita1 = 0.9, bita2 = 0.999, ep =1e-7;


}