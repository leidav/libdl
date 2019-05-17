#ifndef TANH_LAYER_H
#define TANH_LAYER_H

#include "layer.h"

namespace nn
{
class TanhLayer : public Layer
{
public:
	TanhLayer(int batch_size, int layer_size);
	virtual ~TanhLayer();

	void forward(const Array &x, bool train) final;

	void backward(const Array &x, const Array &dy) final;
};
};  // namespace nn

#endif
