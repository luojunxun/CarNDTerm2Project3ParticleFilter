/*
* particle_filter.cpp
*
*  Created on: Dec 12, 2016
*      Author: Tiffany Huang
*/

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle_filter.h"

#define EPS 1.0e-6

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	if (fabs(yaw_rate) > EPS) {
		for (int i = 0; i < num_particles; ++i) {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
	}
	else {
		for (int i = 0; i < num_particles; ++i) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
	}

	// Add random Gaussian noise
	default_random_engine gen;
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);
	for (int i = 0; i < num_particles; ++i) {
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); ++i) {
		double dist_min = numeric_limits<double>::max();
		int index_min = -1;

		for (unsigned int j = 0; j < predicted.size(); ++j) {
			double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (d < dist_min) {
				dist_min = d;
				index_min = predicted[j].id; // id of landmark in map
			}
		}

		observations[i].id = index_min;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double weight_normalizer = 0.0;
	for (int i = 0; i < num_particles; ++i) {
		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double sin_theta_p = sin(particles[i].theta);
		double cos_theta_p = cos(particles[i].theta);

		// Predicted LandmarObs within sensor range
		vector<LandmarkObs> landmark_list_predicted;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			double d = dist(x_p, y_p, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			if (d <= sensor_range) {
				landmark_list_predicted.push_back(
					LandmarkObs{ map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f });
			}
		}

		// Observations in the car coordinate shall be transformed into map coordinates
		std::vector<LandmarkObs> landmark_list_obs_trans;
		for (unsigned int j = 0; j < observations.size(); ++j) {
			double x = x_p + cos_theta_p * observations[j].x - sin_theta_p * observations[j].y;
			double y = y_p + sin_theta_p * observations[j].x + cos_theta_p * observations[j].y;
			landmark_list_obs_trans.push_back(LandmarkObs{ observations[j].id, x, y });
		}

		// Data assocation betwwen predicted landmarks and observations, in map coordinates
		dataAssociation(landmark_list_predicted, landmark_list_obs_trans);

		// Calculate weights
		particles[i].weight = 1.0;
		for (unsigned int j = 0; j < landmark_list_obs_trans.size(); ++j) {
			double x_predict, y_predict;

			for (unsigned int k = 0; k < landmark_list_predicted.size(); ++k) {
				if (landmark_list_predicted[k].id == landmark_list_obs_trans[j].id) {
					x_predict = landmark_list_predicted[k].x;
					y_predict = landmark_list_predicted[k].y;
				}
			}

			particles[i].weight *= 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1])
				* exp(-(pow((landmark_list_obs_trans[j].x - x_predict), 2) / (2.0 * pow(std_landmark[0], 2))
					+ pow((landmark_list_obs_trans[j].y - y_predict), 2) / (2.0 * pow(std_landmark[1], 2))));
		}
		weight_normalizer += particles[i].weight;
	}

	for (int i = 0; i < num_particles; ++i) {
		particles[i].weight /= weight_normalizer;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    // get all of the current weights
  	std::vector<double> weights;
  	for (int i = 0; i < num_particles; i++) {
    	weights.push_back(particles[i].weight);
  	}
  
	std::vector<Particle> particles_resample;
	default_random_engine gen;
	uniform_int_distribution<int> particle_index(0, num_particles - 1);
	int index = particle_index(gen);
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> random_weight(0.0, max_weight * 2.0);
	double beta = 0.0;
	for (int i = 0; i < num_particles; ++i) {
		beta += random_weight(gen);
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		particles_resample.push_back(particles[index]);
	}
	particles = particles_resample;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
	const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
