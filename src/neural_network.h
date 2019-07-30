#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <memory>
#include <vector>

#include "layer/layer.h"
#include "param_saver.h"

namespace nn {
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

class NeuralNetwork {
 public:
  NeuralNetwork(int batch_size);

  ~NeuralNetwork();

  void addHiddenLayer(std::shared_ptr<Layer> layer);

  void addOutputLayer(std::shared_ptr<OutputLayer> output_layer);

  void addResidual(int input);

  float forward(const Layer::ConstArrayRef& input,
                const Layer::ConstArrayRef& ground_truth, bool train);

  void inference(const Layer::ConstArrayRef& input);

  void backward(const Layer::ConstArrayRef& x, float learning_rate);

  const Layer::ConstArrayRef result();

  const Layer::ConstArrayRef inferenceResult();

  void openSaveFile(const char* file, int epoch_offset);

  void saveParameters(float train_loss, float test_loss);

  void loadBestParameters(const char* file);

  void loadLastParameters(const char* file);

  void loadParameters(const char* file, int epoch);

 private:
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
