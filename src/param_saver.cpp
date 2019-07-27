#include "param_saver.h"

namespace nn {

constexpr uint32_t magic_number = 0x6c647970;

ParamSaver::ParamSaver() : m_file_header({0, 0, 0, 0}), m_file(nullptr) {}

ParamSaver::~ParamSaver() { close(); }

bool ParamSaver::open(const char *file) {
  m_file = std::fopen(file, "wb");
  if (m_file == nullptr) {
	return false;
  }
  return true;
}

void ParamSaver::close() {
  if (m_file != nullptr) {
	std::fclose(m_file);
  }
  m_file = nullptr;
}

void ParamSaver::startFile() {
  m_file_header.magic_number = magic_number;
  m_file_header.num_epochs = 0;
  m_file_header.num_layers = 0;
  m_file_header.epoch_size = 0;
  std::fwrite(&m_file_header, sizeof(m_file_header), 1, m_file);
}

void ParamSaver::startEpoch() {
  m_file_header.epoch_size = 0;
  m_file_header.num_epochs++;
  m_file_header.num_layers = 0;
}

void ParamSaver::startLayer(uint64_t id, int num_params) {
  std::fwrite(&id, sizeof(id), 1, m_file);
  std::fwrite(&num_params, sizeof(num_params), 1, m_file);
  m_file_header.epoch_size += sizeof(id) + sizeof(num_params);
  m_file_header.num_layers++;
}

void ParamSaver::addParam(const Layer::ConstArrayRef &values) {
  int length = values.size();
  std::fwrite(&length, sizeof(length), 1, m_file);
  std::fwrite(values.data(), sizeof(float), length, m_file);
  m_file_header.epoch_size += sizeof(length) + sizeof(float) * length;
}

void ParamSaver::endEpoch() {
  updateFileHeader();
  fflush(m_file);
}

void ParamSaver::updateFileHeader() {
  std::fseek(m_file, 0, SEEK_SET);
  std::fwrite(&m_file_header, sizeof(m_file_header), 1, m_file);
  std::fseek(m_file, 0, SEEK_END);
}

ParamLoader::ParamLoader() : m_file_header({0, 0, 0, 0}), m_file(nullptr) {}

ParamLoader::~ParamLoader() { close(); }

bool ParamLoader::open(const char *file) {
  m_file = std::fopen(file, "rb");
  if (m_file == nullptr) {
	return false;
  }
  std::fread(&m_file_header, sizeof(m_file_header), 1, m_file);
  return true;
}

void ParamLoader::close() {
  if (m_file != nullptr) {
	std::fclose(m_file);
  }
  m_file = nullptr;
}

int ParamLoader::epochCount() { return m_file_header.num_epochs; }

void ParamLoader::setLoadingEpoch(int epoch) {
  std::fseek(m_file, m_file_header.epoch_size * epoch + sizeof(m_file_header),
             SEEK_SET);
}

void ParamLoader::loadLayerInfo(uint64_t &id, int &num_params) {
  std::fread(&id, sizeof(id), 1, m_file);
  std::fread(&num_params, sizeof(num_params), 1, m_file);
}

bool ParamLoader::loadParam(Layer::ArrayRef values) {
  int size = 0;
  std::fread(&size, sizeof(size), 1, m_file);
  if (values.size() != size) {
	return false;
  }
  std::fread(values.data(), sizeof(float), size, m_file);
  return true;
}

};  // namespace nn
