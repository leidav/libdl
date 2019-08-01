#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <memory>
#include <vector>

#include "layer/layer.h"
#include "param_saver.h"

namespace nn {

/// \brief implementation of the a neural network
class NeuralNetwork {
 public:
  ///\param batch_size The number of samples/images used at once to train the
  /// network
  NeuralNetwork(int batch_size);

  ~NeuralNetwork();

  ///\brief Adds a layer at the end of the network and before the output layer
  /// \param layer Has to be a pointer to a subclass of Layer
  /// \note Be aware of the shared ownership of the pointer
  void addHiddenLayer(std::shared_ptr<Layer> layer);

  /// \brief Set the output layer
  /// \param output_layer Has to be a shared pointer to a subclass of
  /// OutputLayer
  void addOutputLayer(std::shared_ptr<OutputLayer> output_layer);

  /// \brief Adds a residual connection between the current layer and a previous
  /// layer
  ///\param input The index of a previous layer. The output of the first
  /// layer has the index 0. -1 specifies the neural networks input
  void addResidual(int input);

  /// \brief Calculates the forward pass of all layers
  ///\param input A 2-dimensional array with size (batch_size,input_size)
  /// the input size has to be the input size of the first layer
  ///\param ground_truth Used for validation of the network output and to
  /// calculate the loss and gradient
  ///\param train Specifies if the network is in train or test mode
  /// \return The network loss
  float forward(const Layer::ConstArrayRef& input,
                const Layer::ConstArrayRef& ground_truth, bool train);

  /// \brief Can be used for inference of a single input sample (batch_size of
  /// 1)
  ///\param input A 2-dimensional array with size (1,input_size)
  void inference(const Layer::ConstArrayRef& input);

  /// \brief Calculates the backward pass of all layers
  /// \param x Has to be the same input as used by the forward pass
  /// \param learning_rate Should be a small value
  void backward(const Layer::ConstArrayRef& x, float learning_rate);

  /// \brief Get the result of the last forward pass
  /// \return Constant reference to the networks result array
  const Layer::ConstArrayRef result();

  /// \brief Get the result of inference forward pass (batch_size of 1)
  /// \return Constant reference to the networks inference result array
  const Layer::ConstArrayRef inferenceResult();

  ///\brief Open a new file that should be used for saving the learned layer
  /// parameters
  /// \param file The path to the file
  /// \param epoch_offset The files epoch offset
  /// \note An existing file gets overwritten
  void openSaveFile(const char* file, int epoch_offset);

  /// \brief Save all learnable paramaters from all layers
  /// \param train_loss The current train loss
  /// \param test_loss The current test loss
  void saveParameters(float train_loss, float test_loss);

  /// \brief Load the parameters associated with the smallest test loss from the
  /// given file
  /// \param file The path to the file
  void loadBestParameters(const char* file);

  /// \brief Load the parameters associated with the last epoch from the given
  /// file
  /// \param file The path to the file
  void loadLastParameters(const char* file);

  /// \brief Load the parameters associated with a specific epoch from the given
  /// file
  /// \param file The path to the file
  /// \param epoch The epoch that should be loaded
  void loadParameters(const char* file, int epoch);

 private:
  struct LayerTrainData {
	LayerTrainData() : y(), dx() {}
	LayerTrainData(int batch_size, int input_size, int output_size)
	    : y(batch_size, output_size), dx(batch_size, input_size) {}
	Layer::Array y;
	Layer::Array dx;
  };

  struct ResidualInfo {
	bool residual;
	int layer;
  };
  void saveLayer(Layer& layer, ParamWriter& writer);

  int loadEpoch(ParamReader& reader, int epoch);

  void loadLayer(Layer& loader, ParamReader& reader);

  int m_batch_size;
  std::vector<std::shared_ptr<Layer>> m_hidden_layer;
  std::shared_ptr<OutputLayer> m_output_layer;
  std::vector<LayerTrainData> m_hidden_layer_data;
  std::vector<ResidualInfo> m_residual_info;
  std::vector<ResidualInfo> m_residual_gradient_info;
  LayerTrainData m_output_layer_data;
  ParamWriter m_param_writer;
};
};  // namespace nn

#endif
