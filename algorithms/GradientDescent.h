#include "base.h"

void gradientDescent(const mat&    X,
                     const mat&    y,
                     mat&    parameters,
                     double  computeCost(const mat& X, const mat& y, const mat& parameters),
                     mat  computeGradient(const mat& X, const mat& y, const mat& parameters),
                     string file_name,
                     bool debug,
                     string step,
                     double tol)
{
	// parameters.randu();
	mat _parameters;

	int it=0;
	int m,n;
	m = y.n_rows;
	n = parameters.n_elem;
	mat gradient(n,1),error;
	gradient.ones();

	int training_size = m*.8;
	mat training_X = X;//.rows(0,training_size);
	mat training_y = y;//.rows(0,training_size);
	mat testing_X = X.rows(training_size,m-1);
	mat testing_y = y.rows(training_size,m-1);

	vector<double> training_error_history , testing_error_history;
	do
	{
		it++;_parameters = parameters;

		gradient = computeGradient(X,y,parameters) ;

		double  alpha;
		if(step == "armijo") alpha=  armijo(X,y,parameters,gradient,computeCost);
		else alpha = atof(step.c_str());


		parameters = parameters-alpha*gradient ;

		error = computeCost(X, y, parameters);
		if(debug){
			error.print("training error ("+to_string(it)+"): ");
			cout<<"alpha : "<<alpha<<endl;
		}
		training_error_history.push_back(error[0]);

		error = computeCost(testing_X, testing_y, parameters);

		testing_error_history.push_back(error[0]);



		if ( kbhit() ) { 
  
            // Stores the pressed key in ch 
            char ch = getch(); 
            // Terminates the loop 
            // when escape is pressed 
            if (int(ch) == 27) 
                break; 
  
        } 
	
	}while(norm(parameters - _parameters)>tol );
	
	ofstream coutput_file("costs/training-error_"+file_name);
	for (const auto &e : training_error_history) coutput_file << e << " ";
	
	coutput_file=ofstream("costs/testing-error_"+file_name);
	for (const auto &e : testing_error_history) coutput_file << e << " ";		
	
	ofstream Woutput_file("W/"+file_name);
	for (const auto &e : parameters) Woutput_file << e << " ";

	cout<<"END"<<endl;
			
}



