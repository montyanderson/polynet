class InputNeuron {
	constructor() {
		this.input = 0;
	}

	get output() {
		return this.input;
	}
}

class HiddenNeuron {
	constructor(previousLayerLength) {
		this.weights = [];

		for(let i = 0; i < previousLayerLength; i++)
			this.weights[i] = Math.random();

		this.inputs = 0;
		this.bias = Math.random();
		this.output = 0;
	}
}

module.exports = class PolyNet {
	constructor(...layers) {
		this.f = x => 1 / (1 + Math.exp(0 - x));
		this.createLayers(layers);
	}

	createLayers(layers) {
		const [ inputLayerLength, ...hiddenLayerLengths ] = layers;

		this.layers = [];

		const inputLayer = Array.from({ length: inputLayerLength })
			.map(() => new InputNeuron());

		this.layers.push(inputLayer);

		for(let i = 0; i < hiddenLayerLengths.length; i++) {
			const layer = Array.from({ length: hiddenLayerLengths[i] })
				.map(() => new HiddenNeuron(layers[i]));

			this.layers.push(layer);
		}
	}

	update(inputs) {
		for(let i = 0; i < inputs.length; i++)
			this.layers[0][i].input = inputs[i];

		for(let i = 1; i < this.layers.length; i++) {
			const previousLayer = this.layers[i - 1];
			const layer = this.layers[i];

			for(const neuron of layer) {
				neuron.inputs = neuron.weights.map(
					(weight, index) => previousLayer[index].output * weight
				);

				neuron.output = this.f(neuron.inputs.reduce(
					(sum, input) => sum + input
				) + neuron.bias);
			}
		}

		return this.layers[this.layers.length - 1].map(neuron => neuron.output);
	}

	train(ideal, { incr = 0.001, iterations = 10000 } = {}) {
		for(let a = 0; a < iterations; a++) {
			for(let i = 1; i < this.layers.length; i++) {
				const previousLayer = this.layers[i - 1];
				const layer = this.layers[i];

				for(const neuron of layer) {
					for(let j = 0; j < neuron.weights.length; j++) {
						const weights = [
							neuron.weights[j],
							neuron.weights[j] + incr,
							neuron.weights[j] - incr
						];

						const errors = weights.map(weight => {
							neuron.weights[j] = weight;

							return ideal.reduce((sum, [ inputs, outputs ]) =>
								sum + (
									this.update(inputs).reduce((sum, output, i) =>
										sum + Math.abs(output - outputs[i]), 0
									) / outputs.length
								), 0
							);
						});

						const smallestError = Math.min(...errors);

						neuron.weights[j] = weights[errors.indexOf(smallestError)];
					}
				}
			}
		}
	}
}
