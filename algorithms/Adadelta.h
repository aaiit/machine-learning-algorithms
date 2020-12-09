#include "base.h"

void adadelta(const mat&    X,
                     const mat&    Y,
                     mat&    theta,
                     mat  computeCost(const mat& X, const mat& y, const mat& theta),
                     mat  computeGradient(const mat& X, const mat& y, const mat& theta),
                     string file_name ,
                     int batch_size,
                     int iterations)
{
	
	double beta = 0.95, ep = 1e-6;


}