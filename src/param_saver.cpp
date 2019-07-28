#include "param_saver.h"

namespace nn {

constexpr uint32_t magic_number = 0x6c647970;

ParamWriter::ParamWriter()
    : m_min_train_loss(std::numeric_limits<float>::max()),
      m_min_test_loss(std::numeric_limits<float>::max()),
      m_file_header({0, 0, 0, 0, 0, 0}),
      m_file(nullptr) {}

ParamWriter::~ParamWriter() { close(); }

bool ParamWriter::open(const char *file) {
  m_file = std::fopen(file, "wb");
  if (m_file == nullptr) {
	return false;
  }
  return true;
}

void ParamWriter::close() {
  if (m_file != nullptr) {
	std::fclose(m_file);
  }
  m_file = nullptr;
}

void ParamWriter::startFile(int epoch_offset) {
  m_file_header.magic_number = magic_number;
  m_file_header.epoch_offset = epoch_offset;
  m_file_header.num_epochs = 0;
  m_file_header.num_layers = 0;
  m_file_header.epoch_size = 0;
  m_file_header.best_train_epoch = 0;
  m_file_header.best_test_epoch = 0;
  std::fwrite(&m_file_header, sizeof(m_file_header), 1, m_file);
  m_min_train_loss = std::numeric_limits<float>::max();
  m_min_test_loss = std::numeric_limits<float>::max();
}

void ParamWriter::startEpoch(float train_loss, float test_loss) {
  if (train_loss < m_min_train_loss) {
	m_min_train_loss = train_loss;
	m_file_header.best_train_epoch = m_file_header.num_epochs;
  }
  if (test_loss < m_min_test_loss) {
	m_min_test_loss = test_loss;
	m_file_header.best_test_epoch = m_file_header.num_epochs;
  }
  m_file_header.epoch_size = sizeof(float) * 2;
  m_file_header.num_epochs++;
  m_file_header.num_layers = 0;
  std::fwrite(&train_loss, sizeof(float), 1, m_file);
  std::fwrite(&test_loss, sizeof(float), 1, m_file);
}

void ParamWriter::startLayer(uint64_t id, int num_params) {
  std::fwrite(&id, sizeof(id), 1, m_file);
  std::fwrite(&num_params, sizeof(num_params), 1, m_file);
  m_file_header.epoch_size += sizeof(id) + sizeof(num_params);
  m_file_header.num_layers++;
}

void ParamWriter::addParam(const Layer::ConstArrayRef &values) {
  int length = values.size();
  std::fwrite(&length, sizeof(length), 1, m_file);
  std::fwrite(values.data(), sizeof(float), length, m_file);
  m_file_header.epoch_size += sizeof(length) + sizeof(float) * length;
}

void ParamWriter::endEpoch() {
  updateFileHeader();
  fflush(m_file);
}

void ParamWriter::updateFileHeader() {
  std::fseek(m_file, 0, SEEK_SET);
  std::fwrite(&m_file_header, sizeof(m_file_header), 1, m_file);
  std::fseek(m_file, 0, SEEK_END);
}

ParamReader::ParamReader() : m_file_header({0, 0, 0, 0}), m_file(nullptr) {}

ParamReader::~ParamReader() { close(); }

bool ParamReader::open(const char *file) {
  m_file = std::fopen(file, "rb");
  if (m_file == nullptr) {
	return false;
  }
  std::fread(&m_file_header, sizeof(m_file_header), 1, m_file);
  return true;
}

void ParamReader::close() {
  if (m_file != nullptr) {
	std::fclose(m_file);
  }
  m_file = nullptr;
}

int ParamReader::epochCount() { return m_file_header.num_epochs; }

void ParamReader::setLoadingEpoch(int epoch) {
  std::fseek(m_file, m_file_header.epoch_size * epoch + sizeof(m_file_header),
             SEEK_SET);
  std::fread(&m_train_loss, sizeof(float), 1, m_file);
  std::fread(&m_test_loss, sizeof(float), 1, m_file);
}

void ParamReader::epochLosses(float &train_loss, float &test_loss) {
  train_loss = m_train_loss;
  test_loss = m_test_loss;
}

int ParamReader::bestTrainEpoch() { return m_file_header.best_train_epoch; }

int ParamReader::bestTestEpoch() { return m_file_header.best_test_epoch; }

int ParamReader::epochOffset() { return m_file_header.epoch_offset; }

void ParamReader::readLayerInfo(uint64_t &id, int &num_params) {
  std::fread(&id, sizeof(id), 1, m_file);
  std::fread(&num_params, sizeof(num_params), 1, m_file);
}

bool ParamReader::readParam(Layer::ArrayRef values) {
  int size = 0;
  std::fread(&size, sizeof(size), 1, m_file);
  if (values.size() != size) {
	return false;
  }
  std::fread(values.data(), sizeof(float), size, m_file);
  return true;
}

};  // namespace nn
