#ifndef UPDATE_RULE
#define UPDATE_RULE

#include <cmath>
#include "layer/layer.h"

namespace nn {

template <typename T>
class SGDUpdate {
 public:
  void update(Eigen::Ref<T> param, const Eigen::Ref<const T> &gradient,
              float learning_rate) {
	param += gradient * learning_rate;
  }
};
template <>
class SGDUpdate<float> {
 public:
  void update(float &param, float gradient, float learning_rate) {
	param += gradient * learning_rate;
  }
};

template <typename T>
class AdamUpdate {
 public:
  AdamUpdate(int size)
      : t(0),
        gradient_average(T::Zero(size)),
        squared_gradient_average(T::Zero(size)) {}
  AdamUpdate(int rows, int columns)
      : t(0),
        gradient_average(T::Zero(rows, columns)),
        squared_gradient_average(T::Zero(rows, columns)) {}
  void update(Eigen::Ref<T> param, const Eigen::Ref<const T> &gradient,
              float learning_rate) {
	t++;
	// https://arxiv.org/pdf/1412.6980.pdf page 2
	gradient_average.array() =
	    k_beta1 * gradient_average.array() + (1 - k_beta1) * gradient.array();
	squared_gradient_average.array() =
	    k_beta2 * squared_gradient_average.array() +
	    (1 - k_beta2) * (gradient.array() * gradient.array());
	float alpha =
	    learning_rate * sqrtf(1.0f - powf(k_beta2, t)) / (1 - powf(k_beta1, t));
	param.array() = param.array() -
	                alpha * gradient_average.array() /
	                    (squared_gradient_average.array().sqrt() + k_epsilon);
  }

  static constexpr float k_beta1 = 0.9f;
  static constexpr float k_beta2 = 0.999f;
  static constexpr float k_epsilon = 1e-8f;
  float t;
  T gradient_average;
  T squared_gradient_average;
};

template <>
class AdamUpdate<float> {
 public:
  AdamUpdate() : gradient_average(0), squared_gradient_average(0) {}
  void update(float &param, float gradient, float learning_rate) {
	t++;
	gradient_average = k_beta1 * gradient_average + (1 - k_beta1) * gradient;
	squared_gradient_average = k_beta2 * squared_gradient_average +
	                           (1 - k_beta2) * (gradient * gradient);
	float alpha =
	    learning_rate * sqrtf(1 - powf(k_beta2, t)) / (1 - powf(k_beta1, t));
	param = param - alpha * gradient_average /
	                    (sqrtf(squared_gradient_average) + k_epsilon);
  }

  static constexpr float k_beta1 = 0.9f;
  static constexpr float k_beta2 = 0.999f;
  static constexpr float k_epsilon = 1e-8f;
  float t;
  float gradient_average;
  float squared_gradient_average;
};

};  // namespace nn
#endif
