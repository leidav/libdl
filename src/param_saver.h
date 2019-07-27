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
  uint32_t num_epochs;
  uint32_t epoch_size;
  uint32_t num_layers;
};

class ParamSaver {
 public:
  ParamSaver();
  ~ParamSaver();
  bool open(const char *file);
  void close();
  void startFile();
  void startEpoch();
  void startLayer(uint64_t id, int num_params);
  void addParam(const Layer::ConstArrayRef &values);
  void endEpoch();

 private:
  void updateFileHeader();

  FileHeader m_file_header;
  std::FILE *m_file;
};

class ParamLoader {
 public:
  ParamLoader();
  ~ParamLoader();
  bool open(const char *file);
  void close();
  int epochCount();
  void setLoadingEpoch(int epoch);
  void loadLayerInfo(uint64_t &id, int &num_params);
  bool loadParam(Layer::ArrayRef values);

 private:
  FileHeader m_file_header;
  std::FILE *m_file;
};

};  // namespace nn
#endif
