#pragma once

#include <random>
#include <optional>

#include "qubit.h"

class Error
{
	/*
	Represents a generic quantum error

	Members:
		/
	*/

public:
	virtual void
	apply(Qubit* q);
};

void
Error::apply(Qubit* q)
{
	/*
	Applys the error to a given qubit

	Args:
		q (Qubit): Qubit to apply the error to

	Returns:
		/
	*/
}

class AttenuationError : public Error
{
	/*
	Represents a possible loss of a qubit due to attenuation effects

	Members:
		_attenuation (double): attenuation of the channel
	*/

public:
	double _attenuation;
	std::mt19937 _gen;
	std::uniform_real_distribution<> _dis;

	AttenuationError(double length = 1.0, double attenuation_coefficient = -0.16)
	{
		/*
		Instantiates an attenuation error
        
        Args:
            length (float): length of the fiber in km
            attenuation_coefficient (float): attenuation of fiber in db/km default -0.16
            
        Returns:
            /
		*/

		_attenuation = std::pow(10, (length * attenuation_coefficient / 10));
		std::random_device _rd;
		std::mt19937 _gen(_rd());
		std::uniform_real_distribution<> _dis(0.0, 1.0);
	}

	void
	apply(Qubit* q)
	{
		/*
		Applys the error to the qubit, if the qubit is lost its index is set to -1
        
        Args:
            q (Qubit): qubit to apply error to
            
        Returns:
            /
		*/

		double random_num = _dis(_gen);
		if (random_num > _attenuation)
		{
			q->measure();
			q->_index = -1;
		}
	}
};

class RandomError : public Error
{
	/*
	Represents a random rotation along x and z axis with normal distributed angles

	Members:
		_variance (double): variance of the gaussian distributed angles
	*/

public:
	double _variance;
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<> _n_dist;
	
	RandomError(double variance = 1.0)
	{
		/*
		Instantiates a random error
        
        Args:
            variance (double): variance of normal distributed angles
            
        Returns:
            /
		*/

		_variance = variance;
		_n_dist = std::normal_distribution<>{0.0, _variance};
	}

	void
	apply(Qubit* q)
	{
		/*
		Applys the error to the qubit
        
        Args:
            q (Qubit): qubit to apply error to
            
        Returns:
            /
		*/

		if (q->_index == 0xffffffff)
			return;

		double theta_x = _n_dist(gen);
		double theta_z = _n_dist(gen);

		q->Rx(theta_x);
		q->Rz(theta_z);

	}
};

class SystematicError : public Error
{
	/*
	Represents a random unitary error that is the same for each qubit

	Members:
		theta_x (double): angle with which to rotate around the x-axis
		theta_Z (double): angle with which to rotate around the z-axis
	*/

public:

	double theta_x;
	double theta_z;

	SystematicError(double variance)
	{
		/*
		Instantiates a systematic error
        
        Args:
            variance (double): variance of normal distributed angles
            
        Returns:
            /
		*/

		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> _n_dist{0.0, variance};

		theta_x = _n_dist(gen);
		theta_z = _n_dist(gen);
	}

	void
	apply(Qubit* q) override
	{
		/*
		Applys the error to the qubit
        
        Args:
            q (Qubit): qubit to apply error to
            
        Returns:
            /
		*/

		if (q->_index == 0xffffffff)
			return;

		q->Rx(theta_x);
		q->Rz(theta_z);
	}
};