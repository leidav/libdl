#ifndef PARAM_SAVER_H
#define PARAM_SAVER_H

#include <layer/layer.h>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>

namespace nn {

struct FileHeader {
  uint32_t magic_number;
  uint32_t epoch_offset;
  uint32_t num_epochs;
  uint32_t epoch_size;
  uint32_t num_layers;
  uint32_t best_train_epoch;
  uint32_t best_test_epoch;
};

class ParamWriter {
 public:
  ParamWriter();
  ~ParamWriter();
  bool open(const char *file);
  void close();
  void startFile(int epoch_offset);
  void startEpoch(float train_loss, float test_loss);
  void startLayer(uint64_t id, int num_params);
  void addParam(const Layer::ConstArrayRef &values);
  void endEpoch();

 private:
  void updateFileHeader();

  float m_min_train_loss;
  float m_min_test_loss;
  FileHeader m_file_header;
  std::FILE *m_file;
};

class ParamReader {
 public:
  ParamReader();
  ~ParamReader();
  bool open(const char *file);
  void close();
  int epochCount();
  void setLoadingEpoch(int epoch);
  void readLayerInfo(uint64_t &id, int &num_params);
  bool readParam(Layer::ArrayRef values);
  void epochLosses(float &train_loss, float &test_loss);
  int bestTrainEpoch();
  int bestTestEpoch();
  int epochOffset();

 private:
  float m_train_loss;
  float m_test_loss;
  FileHeader m_file_header;
  std::FILE *m_file;
};

};  // namespace nn
#endif
