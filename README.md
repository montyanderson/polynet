# polynet

A neural network library for Node

## Usage

``` javascript
const PolyNet = require("./");

const network = new PolyNet(2, 3, 2);

network.train([
	[ [ 0, 1 ], [ 1, 0 ] ],
	[ [ 1, 1 ], [ 0, 0 ] ],
	[ [ 1, 0 ], [ 0, 1 ] ],
	[ [ 0, 0 ], [ 1, 1 ] ]
]);

const inputs = 	[
	[ 0, 1 ],
	[ 1, 1 ],
	[ 1, 0 ],
	[ 0, 0 ]
];

for(const input of inputs) {
	console.log(input, network.update(input));
}
```
