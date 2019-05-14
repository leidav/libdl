#ifndef UPDATE_RULE
#define UPDATE_RULE

#include <cmath>
#include "layer/layer.h"

namespace nn {

template <typename T>
class AdamUpdate {
 public:
  AdamUpdate(int size)
      : m_t(0),
        m_gradient_average(T::Zero(size)),
        m_squared_gradient_average(T::Zero(size)) {}
  AdamUpdate(int rows, int columns)
      : m_t(0),
        m_gradient_average(T::Zero(rows, columns)),
        m_squared_gradient_average(T::Zero(rows, columns)) {}
  void update(T &param, const T &gradient, float learning_rate) {
	m_t++;
	// https://arxiv.org/pdf/1412.6980.pdf page 2
	m_gradient_average.array() =
	    k_beta1 * m_gradient_average.array() + (1 - k_beta1) * gradient.array();
	m_squared_gradient_average.array() =
	    k_beta2 * m_squared_gradient_average.array() +
	    (1 - k_beta2) * (gradient.array() * gradient.array());
	float alpha = learning_rate * sqrtf(1.0f - powf(k_beta2, m_t)) /
	              (1 - powf(k_beta1, m_t));
	param.array() = param.array() -
	                alpha * m_gradient_average.array() /
	                    (m_squared_gradient_average.array().sqrt() + k_epsilon);
  }

 private:
  static constexpr float k_beta1 = 0.9f;
  static constexpr float k_beta2 = 0.999f;
  static constexpr float k_epsilon = 1e-8f;
  float m_t;
  T m_gradient_average;
  T m_squared_gradient_average;
};

template <>
class AdamUpdate<float> {
 public:
  AdamUpdate() : m_gradient_average(0), m_squared_gradient_average(0) {}
  void update(float &param, float gradient, float learning_rate) {
	m_t++;
	m_gradient_average =
	    k_beta1 * m_gradient_average + (1 - k_beta1) * gradient;
	m_squared_gradient_average = k_beta2 * m_squared_gradient_average +
	                             (1 - k_beta2) * (gradient * gradient);
	float alpha = learning_rate * sqrtf(1 - powf(k_beta2, m_t)) /
	              (1 - powf(k_beta1, m_t));
	param = param - alpha * m_gradient_average /
	                    (sqrtf(m_squared_gradient_average) + k_epsilon);
  }

 private:
  static constexpr float k_beta1 = 0.9f;
  static constexpr float k_beta2 = 0.999f;
  static constexpr float k_epsilon = 1e-8f;
  float m_t;
  float m_gradient_average;
  float m_squared_gradient_average;
};

};  // namespace nn
#endif
