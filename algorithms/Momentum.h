void momentum(const mat&    X,
                     const mat&    Y,
                     mat&    theta,
                     mat  computeCost(const mat& X, const mat& y, const mat& theta),
                     mat  computeGradient(const mat& X, const mat& y, const mat& theta),
                     string file_name ,
                     int batch_size,
                     int iterations,
                     double bita) // 0.9
{
	int m,n;
	m = Y.n_rows;
	n = theta.n_elem;
	mat M(n,1),gradient;

	M.zeros(); 

	vector<double> J_history;
	
	while(iterations--)
	{
		//  Select Batch
		int p=iterations%m,q=(iterations+batch_size)%m ; 
		if(p>q){iterations++;continue;}

		mat x=X.rows(p, q) , y= Y.rows(p ,q);

		gradient = computeGradient(x,y,theta) ;

		M = bita * M + (1 - bita) * gradient; // Momentum

		double  alpha = armijo(x,y,theta,M,computeCost);
		theta = theta-alpha*M ;
		mat J = computeCost(x, y, theta);
		J.print("J: ");
		J_history.push_back(J[0]);
	
	}
	ofstream output_file("costs/training-error_"+file_name);
    for (const auto &e : J_history) output_file << e << " ";

	ofstream Woutput_file("W/"+file_name);
	for (const auto &e : theta) Woutput_file << e << " ";
}

