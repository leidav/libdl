#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Core>
#include <cstdint>

namespace nn {

// fnv1a hash
static constexpr uint64_t layerNameHash(const char* name) {
  uint64_t hash = 14695981039346656037ull;
  int i = 0;
  while (name[i] != 0) {
	hash ^= static_cast<uint8_t>(name[i]);
	hash *= 1099511628211ull;
	i++;
  }
  return hash;
}

class Layer {
 public:
  Layer(int input_size, int output_size);
  virtual ~Layer() = default;
  using Array =
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ArrayRef = Eigen::Ref<Array>;
  using ConstArrayRef = Eigen::Ref<const Array>;

  virtual void forward(ArrayRef y, const ConstArrayRef& x, bool train) = 0;

  int inputSize();
  int outputSize();

  virtual void backward(ArrayRef dx, const ConstArrayRef& x,
                        const ConstArrayRef& y, const ConstArrayRef& dy) = 0;
  virtual float regularizationLoss();

  virtual void update(float learning_rate);

  virtual uint64_t id() = 0;

  virtual int paramCount() = 0;

  virtual ArrayRef param(int param) = 0;

  // virtual bool setParam(int param, const ConstArrayRef& values) = 0;

 protected:
  int m_input_size;
  int m_output_size;
};

class OutputLayer : public Layer {
 public:
  OutputLayer(int input_size, int output_size);
  virtual ~OutputLayer() = default;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

  virtual float loss(ArrayRef dx, const ConstArrayRef& x,
                     const ConstArrayRef& labels) = 0;
};
};  // namespace nn

#endif
