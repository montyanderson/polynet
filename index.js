const {
  Worker, isMainThread, parentPort, workerData
} = require('worker_threads');

class InputNeuron {
	constructor() {
		this.output = 0;
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

if (!isMainThread) {
	parentPort.on("message", ({ previousLayer, neuron }) => {
		neuron.inputs = neuron.weights.map(
			(weight, index) => previousLayer[index].output * weight
		);

		const f = x => x / (1 + Math.abs(x));

		neuron.output = f(neuron.inputs.reduce(
			(sum, input) => sum + input
		) + neuron.bias);

		parentPort.postMessage(neuron.output);
	});

	return;
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
			this.layers[0][i].output = inputs[i];

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

	async updateThreaded(inputs) {
		for(let i = 0; i < inputs.length; i++)
			this.layers[0][i].output = inputs[i];

		if(typeof this.workers === 'undefined') {
			this.workers = [];
			const totalWorkers = Math.max(...this.layers.slice(1).map(layer => layer.length));

			for(let i = 0; i < totalWorkers; i++)
				this.workers.push(new Worker(__filename));
		}

		if(this._lock === true)
			throw new Error("Can only run one updateThreaded instance at a time");

		this._lock = true;

		for(let i = 1; i < this.layers.length; i++) {
			const previousLayer = this.layers[i - 1];
			const layer = this.layers[i];

			await Promise.all(layer.map(async (neuron, i) => {
				this.workers[i].postMessage({ previousLayer, neuron });

				return new Promise(resolve => {
					this.workers[i].once('message', output => {
						neuron.output = output;
						resolve();
					});
				});
			}));
		}

		this._lock = false;

		return this.layers[this.layers.length - 1].map(neuron => neuron.output);
	}

	async trainThreaded(ideal, { incr = 0.001, iterations = 10000 } = {}) {
		for(let a = 0; a < iterations; a++) {
			console.log(`Iteration: ${a}/${iterations}`);

			for(let i = 1; i < this.layers.length; i++) {
				const previousLayer = this.layers[i - 1];
				const layer = this.layers[i];

				for(const neuron of layer) {
					console.log(layer.indexOf(neuron));

					for(let j = 0; j < neuron.weights.length; j++) {
						const weights = [
							neuron.weights[j],
							neuron.weights[j] + incr,
							neuron.weights[j] - incr
						];

						const errors = [];

						for(const weight of weights) {
							neuron.weights[j] = weight;

							let sum = 0;

							for(const [ inputs, outputs ] of ideal) {
								const currentOutputs = await this.updateThreaded(inputs);

								let _sum = 0;

								currentOutputs.forEach((output, i) =>
									_sum += Math.abs(output - outputs[i])
								);

								sum += _sum / outputs.length;
							}

							errors[weights.indexOf(weight)] = sum;
						}

						const smallestError = Math.min(...errors);

						neuron.weights[j] = weights[errors.indexOf(smallestError)];
					}
				}
			}
		}
	}
}
