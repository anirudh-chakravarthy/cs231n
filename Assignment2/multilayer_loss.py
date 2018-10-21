# multi-layer network backprop

def loss(self, X, y=None):

	input = X
	
	# forward prop
	for i in range(1, self.num_layers + 1):
		Wname = 'W' + str(i)
		bname = 'b' + str(i)
		cache = []

		# last layer
		if i == self.num_layers:
			input, c = affine_forward(input, self.params[Wname], self.params[bname])
			cache.append(c)

		else:
			input, c = affine_relu_forward(input, self.params[Wname], self.params[bname])
			cache.append(c)

	scores = input

	# Backprop variables
	loss, grads = 0.0, {}

	loss, der = softmax_loss(input, y)

	for i in range(self.num_layers, 0, -1):
		Wname = W + str(i)
		bname = b + str(i)

		if i == self.num_layers:
			der, dw, db = affine_backward(der, cache[i])

			dw += self.reg * self.params[Wname]
			loss += 0.5 * self.reg * np.sum(self.params[Wname] ** 2)

			grads[Wname] = dw
			grads[bname] = db

		else:
			der, dw, db = affine_relu_backward(der, cache[i])
			grads[Wname] = dw
			grads[bname] = db