#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Core>
#include <cstdint>

namespace nn {

/// \brief fnv1a hash function.
/// Use this to calculate the the layers id()
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

/// \brief Abastract base class for all Layers.
/// New layer should subclass this and implement the abstract interface.
class Layer {
 public:
  /// \param input_size The output size of the previous layer
  /// \param output_size The input size of the next layer
  Layer(int input_size, int output_size);

  virtual ~Layer() = default;

  /// We use row major storage order
  using Array =
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  using ArrayRef = Eigen::Ref<Array>;

  using ConstArrayRef = Eigen::Ref<const Array>;

  /// \brief Forward pass of the layer
  /// \param y  A pre allocated Eigen Array with size (batch_size,output_size)
  /// \param x  The input array with size (batch_size,input_size)
  /// \param train Specifies if we are in train or inference mode (can be
  /// useful / for dropout¸)
  virtual void forward(ArrayRef y, const ConstArrayRef& x, bool train) = 0;

  /// \brief The size of an input row
  /// \return input size
  int inputSize();

  /// \brief The size of an output row
  /// \return output size
  int outputSize();

  /// Backward pass of the layer
  /// \param dx The output gradient
  /// \param x Must be the same input used in the forward pass
  /// \param y Must be the saved result produced by the forward pass
  /// \param dy The gradient of the previous layer (as counted from output to
  /// input layer)
  virtual void backward(ArrayRef dx, const ConstArrayRef& x,
                        const ConstArrayRef& y, const ConstArrayRef& dy) = 0;
  /// \brief Optional regularization penality (i.e L2)
  /// \return The regularization loss
  virtual float regularizationLoss();

  /// \brief Update the layer
  ///\param learning_rate Should be a small number
  /// \note this function Should be called after the backward pass
  virtual void update(float learning_rate);

  /// \brief A unique hash value per layer type to identify the layer type in
  /// the save file
  /// \return Unique hash
  virtual uint64_t id() = 0;

  /// \brief Get the number of learnable parameters
  /// \return The number of learnable parameters
  virtual int paramCount() = 0;

  /// \brief Get a learnable parameter
  ///\param param The number of the parameter.Should be between 0 and
  /// paramCount()-1
  /// \return Writable reference to the parameter
  virtual ArrayRef param(int param) = 0;

 protected:
  int m_input_size;
  int m_output_size;
};

/// \brief Base class for output layers
class OutputLayer : public Layer {
 public:
  OutputLayer(int input_size, int output_size);

  virtual ~OutputLayer() = default;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

  /// Loss function
  ///\param dx The output gradient regarding the loss
  ///\param x The input
  ///\param labels The ground truth that is compared against x
  /// \return The loss
  virtual float loss(ArrayRef dx, const ConstArrayRef& x,
                     const ConstArrayRef& labels) = 0;
};
};  // namespace nn

#endif
