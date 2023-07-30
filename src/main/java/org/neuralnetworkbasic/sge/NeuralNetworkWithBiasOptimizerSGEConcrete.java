package org.neuralnetworkbasic.sge;

import org.neuralnetworkbasic.NeuralNetworkWithBias;
import org.neuralnetworkbasic.Optimizable;

public class NeuralNetworkWithBiasOptimizerSGEConcrete extends NeuralNetworkWithBiasOptimizerSGE {
	
	public NeuralNetworkWithBiasOptimizerSGEConcrete(NeuralNetworkWithBias net) {
		super(net);
	}

	@Override
	public boolean keepTraining(Optimizable<double[]> optimizable, int iteration, double cost) {
		return iteration < 1;	
	}

}
