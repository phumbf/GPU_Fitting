/*
 *C++ function which is compiled into a dynamic link library (.dll) on compilation.
 The function contains a wrapper for the call to "gpufit" - gpufit.readthedocs.io/en/latest/introduction.html

 gpufit takes in flattened (1D) C-style arrays. The runGpuFit is called from a separate python script where 
 the python objects are passed directly in via the python module "ctypes"
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include "gpufit.h"

//To output a .dll for external usage
extern "C" {

	__declspec(dllexport)

	void runGpuFit(double *data,
		        double *steps,
		        double *initparams,
	         	double *finalparams,
		        double *chi,
			double *iter,
		        double *state,
		        int n_steps,
	         	int width,
	        	int height) {

		std::cout << "C++: C++ code now executing" << std::endl;

		//Number of fits, number of points per fit
		size_t const n_fits = width * height;
		size_t const n_points_per_fit = n_steps;

		//GPUFit model ID and number of model parameters
		int const model_id = GAUSS_1D;
		// Currently use 4 parameters, A0 - amplitude, A1 - mean, A2 - std dev, A3 - offset
		size_t const n_model_parameters = 4;

		//Initial parameters
		std::vector< REAL > initial_parameters;
		initial_parameters.assign(initparams, initparams + n_fits*n_model_parameters);

		//X-coordinates
		std::vector<float> xcoords;
		xcoords.assign(steps, steps + n_steps);

		//X-coord size 
		const size_t xsize = n_steps * sizeof(float);

		//Data
		std::vector< REAL > fitdata;
		fitdata.assign(data, data + n_points_per_fit * n_fits);

		//Tolerance - determines whether fit has converged
		REAL const tolerance = 0.001f;

		//Maximum number of iterations per fit 
		int const max_number_iterations = 100;

		//Estimator ID - currently set to maximum likelihood fit
		//Alternatively, can use a Maximum likelihood estimator, MLE
		int const estimator_id = LSE;

		//Parameters to fit (all of them)
		std::vector< int > parameters_to_fit(n_model_parameters, 1);

		//Output parameters
		std::vector< REAL > output_parameters(n_fits * n_model_parameters);
		std::vector< int > output_states(n_fits);
		std::vector< REAL > output_chi_square(n_fits);
		std::vector< int > output_number_iterations(n_fits);

		//Call to GPUFit 
		std::cout << "C++: Commencing GPUFit for " <<  n_fits << " fits" << std::endl;

		//Start a clock
		std::chrono::steady_clock::time_point beginfit = std::chrono::steady_clock::now();
		int const status = gpufit
		(
			n_fits, //Number of fits
			n_points_per_fit, //Number of data points per fit
			fitdata.data(), //The flattened fit data
			0, //Flattened array of weights for data points
			model_id, //ID number specifying shape to fit 
			initial_parameters.data(), //Flattened array of initial floating values
			tolerance, //Tolerance to determine when has converged
			max_number_iterations, //Max number of iterations to try
			parameters_to_fit.data(), //Array of flags to set values as constant
			estimator_id, //ID number of estimator used in fitting
			xsize, //Size of the other input data array 
			reinterpret_cast<char*>(xcoords.data()), //Other data info eg. additional params, additional independent variables etc.
			output_parameters.data(), //Array to store the resultant parameters for each fit
			output_states.data(), //Array contatining information such as whether fit converges
			output_chi_square.data(), //Chisq value for each fit
			output_number_iterations.data() //Final number of iterations required to converge
		);
		
		//Stop the clock
		std::chrono::steady_clock::time_point endfit = std::chrono::steady_clock::now();
		std::cout << "C++: Fits complete - time taken to fit: " << std::chrono::duration_cast<std::chrono::milliseconds>(endfit - beginfit).count() << "ms" << std::endl;

		// Check for any gpufit() errors
		if (status != ReturnState::OK)
		{
			std::cout << "C++: RETURN STATE ERROR" << std::endl;
			throw std::runtime_error(gpufit_get_last_error());
		}

		//Save the output. This is slightly clunky and unoptimised at the moment
		//The reason for separate arrays is to do with the specific call to GPUFit which 
		//expects 32 bit floats whereas python "floats" are 64 bit objects
		else {
			std::cout << "C++: All Fits finished" << std::endl;
			for (int i = 0; i < output_parameters.size(); i++) {
				finalparams[i] = output_parameters[i];
			}

			for (int i = 0; i < n_fits; i++) {
				chi[i] = output_chi_square[i];
				iter[i] = output_number_iterations[i];
				state[i] = output_states[i];
			}
		}
	}
}
