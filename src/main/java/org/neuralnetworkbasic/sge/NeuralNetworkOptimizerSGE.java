package org.neuralnetworkbasic.sge;

import org.neuralnetworkbasic.NeuralNetwork;
import org.neuralnetworkbasic.NeuralNetworkOptimizable;

public abstract class NeuralNetworkOptimizerSGE extends OptimizerSGE<double[]> {

	public NeuralNetworkOptimizerSGE(NeuralNetwork net) {
		super(new NeuralNetworkOptimizable(net));
	}

}
