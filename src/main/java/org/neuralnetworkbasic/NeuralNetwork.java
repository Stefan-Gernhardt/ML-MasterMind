/*
 * MIT License
 * 
 * Copyright (c) 2019 Sebastian Gössl
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

package org.neuralnetworkbasic;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

import org.neuralnetworkbasic.la.Matrix;

/**
 * Neural network implementation. Supports operations for forward propagation
 * and backpropagation. (Matricies and layers are zero indexed!)
 * 
 * Input & output sets are stored in matricies in which every row represents an
 * dataset and every column gets feed into an node (e.g. element[1][2] is a
 * parameter of a dataset with index [1] and gets feed into the node with index
 * [2])
 * 
 * The network is build up from an array of layers. Every layer consists of the
 * weights leading to it from the previous layer, its nodes (neurons) and its
 * activation function (and derivative for backpropagation = optimization =
 * training) that gets applied to the values in every node. The first layer
 * (index: 0) is the first hidden layer. The last layer (index: layers.lenght-1)
 * is the output layer.
 * 
 * The weights are aranged in matricies. The conection of an single weight
 * represent the indices of the nodes it connects where the row indicate the
 * previous node and the column the node it leads into. (e.g. weight[2][1] of
 * layer[3] connects the node[2] of layer[2] with node[1] of layer[3])
 * 
 */
public class NeuralNetwork implements UnaryOperator<Matrix> {

	/**
	 * Layers of the network.
	 */
	private final Layer[] layers;

	/**
	 * Constructs a copy of the given NeuralNetwork.
	 * 
	 * @param other network to copy
	 */
	public NeuralNetwork(NeuralNetwork other) {
		this(other.getNumberOfInputs(), other.getLayerSizes(), other.getActivationFunctions(),
				other.getActivationFunctionPrimes());
		setWeights(other.getWeights());
	}

	/**
	 * Constructs a new NeuralNetwork.
	 * 
	 * @param numberOfInputs           number of input nodes
	 * @param layerSizes               number of nodes in the hidden layer and
	 *                                 output layer
	 * @param activationFunctions      activation functions in the hidden layers
	 * @param activationFunctionPrimes derivatives of the activation functions in
	 *                                 the hidden layers
	 */
	public NeuralNetwork(int numberOfInputs, int[] layerSizes, DoubleUnaryOperator[] activationFunctions,
			DoubleUnaryOperator[] activationFunctionPrimes) {

		if (layerSizes.length < 2) {
			throw new RuntimeException("No Output layer definded. Hint: Increase count of layers");
		}

		if (layerSizes.length > activationFunctions.length) {
			throw new RuntimeException("Not all activation functions passed!");
		}

		if (layerSizes.length > activationFunctionPrimes.length) {
			throw new RuntimeException("Not all derived activation functions passed!");
		}

		layers = new Layer[layerSizes.length];

		layers[0] = new Layer(numberOfInputs, layerSizes[0], activationFunctions[0], activationFunctionPrimes[0]);
		for (int i = 1; i < layers.length; i++) {
			layers[i] = new Layer(layerSizes[i - 1], layerSizes[i], activationFunctions[i],
					activationFunctionPrimes[i]);
		}

		seedWeightsXavier();
	}

	/**
	 * Returns the number of input nodes.
	 * 
	 * @return number of input nodes
	 */
	public int getNumberOfInputs() {
		return layers[0].getNumberOfInputs();
	}

	/**
	 * Returns the number of output nodes.
	 * 
	 * @return number of output nodes
	 */
	public int getNumberOfOutputs() {
		return layers[layers.length - 1].getNumberOfOutputs();
	}

	/**
	 * Returns the number of layers.
	 * 
	 * @return number of layers
	 */
	public int getNumberOfLayers() {
		return layers.length;
	}

	/**
	 * Returns the number of nodes in the specified layer.
	 * 
	 * @param layer index of the layer
	 * @return number of nodes in the specified layer
	 */
	public int getLayerSize(int layer) {
		return layers[layer].getNumberOfOutputs();
	}

	/**
	 * Returns all layer sizes as an array.
	 * 
	 * @return all layer sizes as an array
	 */
	public int[] getLayerSizes() {
		final int[] layerSizes = new int[getNumberOfLayers()];
		Arrays.setAll(layerSizes, (i) -> getLayerSize(i));

		return layerSizes;
	}

	/**
	 * Returns the weights of the specified layer.
	 * 
	 * @param layer index of the layer
	 * @return weights of the specified layer
	 */
	public Matrix getWeights(int layer) {
		return layers[layer].getWeights();
	}

	/**
	 * Returns all weights as an array.
	 * 
	 * @return all weights as an array
	 */
	public Matrix[] getWeights() {
		final Matrix[] weights = new Matrix[getNumberOfLayers()];
		Arrays.setAll(weights, (i) -> getWeights(i));

		return weights;
	}

	/**
	 * Replaces the weights of the specified layer with the given weights.
	 * 
	 * @param layer   index of the layer
	 * @param weights weights to replace the current weights with
	 */
	public void setWeights(int layer, Matrix weights) {
		layers[layer].setWeights(weights);
	}

	/**
	 * Replaces all weights with the given weights.
	 * 
	 * @param weights weights to replace the current weights with
	 */
	public void setWeights(Matrix[] weights) {
		for (int i = 0; i < getNumberOfLayers(); i++) {
			setWeights(i, weights[i]);
		}
	}

	/**
	 * Returns the activation function of the specified layer.
	 * 
	 * @param layer index of the layer
	 * @return activation function of the specified layer
	 */
	public DoubleUnaryOperator getActivationFunction(int layer) {
		return layers[layer].getActivationFunction();
	}

	/**
	 * Returns all activation functions as an array.
	 * 
	 * @return all activation functions as an array
	 */
	public DoubleUnaryOperator[] getActivationFunctions() {
		final DoubleUnaryOperator[] activationFunctions = new DoubleUnaryOperator[getNumberOfLayers()];
		Arrays.setAll(activationFunctions, (i) -> getActivationFunction(i));

		return activationFunctions;
	}

	/**
	 * Returns the derivative of the activation function of the specified layer.
	 * 
	 * @param layer index of the layer
	 * @return derivative of the activation function of the specified layer
	 */
	public DoubleUnaryOperator getActivationFunctionPrime(int layer) {
		return layers[layer].getActivationFunctionPrime();
	}

	/**
	 * Returns all activation function derivatives as an array.
	 * 
	 * @return all activation function derivatives as an array
	 */
	public DoubleUnaryOperator[] getActivationFunctionPrimes() {
		final DoubleUnaryOperator[] activationFunctionPrimes = new DoubleUnaryOperator[getNumberOfLayers()];
		Arrays.setAll(activationFunctionPrimes, (i) -> getActivationFunctionPrime(i));

		return activationFunctionPrimes;
	}

	/**
	 * Randomizes all weights based on the Xavier initialization algorithm.
	 */
	public void seedWeightsXavier() {
		seedWeightsXavier(new Random());
	}

	/**
	 * Randomizes all weights based on the Xavier initialization algorithm with a
	 * specified seed for the pseudorandom number generator.
	 * 
	 * @param seed the initial seed
	 */
	public void seedWeightsXavier(long seed) {
		seedWeightsXavier(new Random(seed));
	}

	/**
	 * Randomizes all weights based on the Xavier initialization algorithm with the
	 * given random number generator.
	 * 
	 * @param rand random number generator
	 */
	public void seedWeightsXavier(Random rand) {
		for (Layer layer : layers) {
			final double average = (layer.getNumberOfInputs() + layer.getNumberOfOutputs()) / 2;
			layer.getWeights().set(() -> rand.nextGaussian() / Math.sqrt(average));
		}
	}

	/**
	 * Limits all weights to a minimum of <code>-Double.MAX_VALUE / 2</code> and a
	 * maximum of <code>Double.MAX_VALUE / 2</code> while also eliminating NaNs.
	 */
	public void keepWeightsInBounds() {
		keepWeightsInBounds(-Double.MAX_VALUE / 2, Double.MAX_VALUE / 2);
	}

	/**
	 * Limits all weights to the given minimum and maximum while also eliminating
	 * NaNs by replacing the with the average of the given limits.
	 * 
	 * @param minimum minimum value for the weights
	 * @param maximum maximum value for the weights
	 */
	public void keepWeightsInBounds(double minimum, double maximum) {
		for (Layer layer : layers) {
			layer.getWeights().apply((x) -> {

				if (Double.isNaN(x)) {
					return (minimum + maximum) / 2;
				}
				return Math.min(Math.max(x, minimum), maximum);
			});
		}
	}

	/**
	 * Forward propagates the given input through the neural network and returns the
	 * result.
	 * 
	 * @param t input to propagate through the network
	 * @return output of the network
	 */
	@Override
	public Matrix apply(Matrix t) {
		Matrix a = t;

		// int l = 0;
		for (Layer layer : layers) {
			a = layer.apply(a);
			// System.out.println("Layer" + l + "result: " + a);
			// l++;
		}

		return a;
	}

	/**
	 * Forward propagates the given input through the neural network and returns the
	 * mean squared error by comparing its output to the given output.
	 * 
	 * @param input  input for the network
	 * @param output output to compare the networks output with
	 * @return mean squared error of the networks output and the given output
	 */
	public double cost(Matrix input, Matrix output) {
		// System.out.println("method::NeuralNetwork::cost");
		// System.out.println(" input vector size: " + input.getHeight() + " " +
		// input.getWidth());
		// System.out.println(" output vector size: " + output.getHeight() + " " +
		// output.getWidth());

		/*
		 * int countInputRows = input.getHeight(); int countInputCols =
		 * input.getWidth(); System.out.println("input"); for(int row=0; row <
		 * countInputRows; row++) { for(int col=0; col < countInputCols; col++) {
		 * System.out.print("" + input.get(row, col) + ", "); } System.out.println(); }
		 * 
		 * int countOutputRows = output.getHeight(); int countOutputCols =
		 * output.getWidth();
		 * 
		 * System.out.println("output"); for(int row=0; row < countOutputRows; row++) {
		 * for(int col=0; col < countOutputCols; col++) { System.out.print("" +
		 * output.get(row, col) + ", "); } System.out.println(); }
		 */

		final Matrix difference = output.subtract(apply(input));
		final Matrix squaredDifference = difference.multiplyElementwise(difference);

		double cost = 0;
		for (double error : squaredDifference) {
			cost += error;
		}
		cost /= 2;

		// System.out.println(" cost: " + cost);
		// System.out.println();
		return cost;
	}

	/**
	 * Returns the derivative of the cost with respect to every weight.
	 * 
	 * @param input  input for the network
	 * @param output wanted output of the network
	 * @return derivative of the cost with respect to every weight
	 */
	public Matrix[] costPrime(Matrix input, Matrix output) {
		final Matrix yHat = apply(input);
		final Matrix[] dJdW = new Matrix[layers.length];

		Matrix delta = yHat.subtract(output).multiplyElementwise(
				layers[layers.length - 1].z.applyNew(layers[layers.length - 1].activationFunctionPrime));

		for (int i = layers.length - 1; i > 0; i--) {
			dJdW[i] = layers[i].backpropagateToWeights(delta, layers[i - 1]);
			delta = layers[i - 1].backpropagate(delta, layers[i]);
		}

		dJdW[0] = input.transpose().multiply(delta);

		return dJdW;
	}

	/**
	 * Helper class which represents a single layer of the neural network. It
	 * consists of its nodes (neurons) and the weights (synapses) leading to it from
	 * the previous layer.
	 */
	private class Layer implements UnaryOperator<Matrix> {

		/**
		 * Activation function and its derivative.
		 */
		private final DoubleUnaryOperator activationFunction, activationFunctionPrime;
		/**
		 * Weights leading into this layer.
		 */
		private Matrix weights;
		/**
		 * Activation values needed for backpropagation. z = weighted and added inputs
		 * from the previous layer a = z with applied activation function
		 */
		private Matrix z, a;

		/**
		 * Constructs a copy of the given layer
		 * 
		 * @param other layer to copy
		 */
		public Layer(Layer other) {
			this(other.getNumberOfInputs(), other.getNumberOfOutputs(), other.getActivationFunction(),
					other.getActivationFunctionPrime());
			setWeights(other.getWeights());
		}

		/**
		 * Constructs a new layer on a neural network.
		 * 
		 * @param inputs                  number of nodes of the previous layer
		 * @param outputs                 number of nodes (outputs) of this layer
		 * @param activationFunction      activation function that gets applied in this
		 *                                layers nodes
		 * @param activationFunctionPrime derivative of the activation function
		 */
		public Layer(int inputs, int outputs, DoubleUnaryOperator activationFunction,
				DoubleUnaryOperator activationFunctionPrime) {

			this.activationFunction = activationFunction;
			this.activationFunctionPrime = activationFunctionPrime;

			weights = new Matrix(inputs, outputs);
		}

		/**
		 * Returns the number of inputs (number of nodes of the previous layer).
		 * 
		 * @return number of inputs
		 */
		public int getNumberOfInputs() {
			return weights.getHeight();
		}

		/**
		 * Returns the number of nodes (outputs).
		 * 
		 * @return number of nodes (outputs)
		 */
		public int getNumberOfOutputs() {
			return weights.getWidth();
		}

		/**
		 * Returns the weights.
		 * 
		 * @return weights
		 */
		public Matrix getWeights() {
			return weights;
		}

		/**
		 * Returns the activation function.
		 * 
		 * @return activation function
		 */
		public DoubleUnaryOperator getActivationFunction() {
			return activationFunction;
		}

		/**
		 * Returns the derivative of the activation function.
		 * 
		 * @return derivative of the activation function
		 */
		public DoubleUnaryOperator getActivationFunctionPrime() {
			return activationFunctionPrime;
		}

		/**
		 * Sets the weights to the given weights.
		 * 
		 * @param weights new weights
		 */
		public void setWeights(Matrix weights) {
			if (this.weights.getHeight() != weights.getHeight() || this.weights.getWidth() != weights.getWidth()) {
				throw new IllegalArgumentException("Wrong weights dimensions");
			}

			this.weights = weights;
		}

		/**
		 * Forward propagates the given input through the layer and returns the result.
		 * 
		 * @param t input to forward propagate through the layer
		 * @return output of the layer
		 */
		@Override
		public Matrix apply(Matrix t) {
			z = t.multiply(weights);
			a = z.applyNew(activationFunction);

			return a;
		}

		/**
		 * Backpropagates the given matrix which represents a derivative with respect to
		 * every node of this layer to a the derivative with respect to every node of
		 * the previous layer.
		 * 
		 * @param deltaNext derivative with respect to every node of this layer
		 * @param next      next layer
		 * @return
		 */
		public Matrix backpropagate(Matrix deltaNext, Layer next) {
			return deltaNext.multiply(next.weights.transpose())
					.multiplyElementwise(z.applyNew(activationFunctionPrime));
		}

		/**
		 * Backpropagates the given matrix which represents a derivative with respect to
		 * every node of this layer to a the derivative with respect to every weight of
		 * this layer.
		 * 
		 * @param delta    derivative with respect to every node of this layer
		 * @param previous previous layer
		 * @return derivative with respect to every weight of this layer
		 */
		public Matrix backpropagateToWeights(Matrix delta, Layer previous) {
			return previous.a.transpose().multiply(delta);
		}
	}

}
