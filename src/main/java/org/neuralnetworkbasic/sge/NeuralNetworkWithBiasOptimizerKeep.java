package org.neuralnetworkbasic.sge;

import org.neuralnetworkbasic.NeuralNetworkWithBias;
import org.neuralnetworkbasic.NeuralNetworkWithBiasOptimizer;
import org.neuralnetworkbasic.Optimizable;

public class NeuralNetworkWithBiasOptimizerKeep extends NeuralNetworkWithBiasOptimizer{
	private int inputSize = 0;

	public NeuralNetworkWithBiasOptimizerKeep(NeuralNetworkWithBias net, int inputSizeParameter) {
		super(net);
		inputSize = inputSizeParameter;
	}

	@Override
	public boolean keepTraining(Optimizable<double[]> optimizable, int iteration, double cost) {
		return iteration < inputSize;
	}

}
