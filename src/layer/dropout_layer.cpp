#include "dropout_layer.h"
#include <chrono>
#include <functional>
#include <random>

namespace nn {

static thread_local std::random_device g_seed_generator;
static thread_local std::mt19937 g_mersenne_twister(g_seed_generator());
static thread_local std::uniform_int_distribution<uint32_t>
    g_integer_distribution;
static thread_local uint32_t g_counter = 0;
static thread_local uint32_t g_bits =
    g_integer_distribution(g_mersenne_twister);

float bernoulli(float) {
  float val = (g_bits & 0x1) ? 1.0f : 0.0f;
  if (g_counter < 32) {
	g_bits >>= 1;
	g_counter++;
  } else {
	g_counter = 0;
	g_bits = g_integer_distribution(g_mersenne_twister);
  }
  return val;
}

DropOutLayer::DropOutLayer(int batch_size, int layer_size)
    : Layer(layer_size, layer_size), m_mask(batch_size, layer_size) {}

DropOutLayer::~DropOutLayer() {}

void DropOutLayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  if (train) {
	m_mask = m_mask.unaryExpr(std::ptr_fun(bernoulli));
	y = x * m_mask;
  } else {
	y = x;
  }
}

void DropOutLayer::backward(ArrayRef dx, const ConstArrayRef &x,
                            const ConstArrayRef &y, const ConstArrayRef &dy) {
  dx = m_mask * dy;
}

};  // namespace nn
