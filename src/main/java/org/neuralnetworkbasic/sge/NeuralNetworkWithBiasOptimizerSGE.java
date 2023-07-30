package org.neuralnetworkbasic.sge;

import org.neuralnetworkbasic.NeuralNetworkWithBias;
import org.neuralnetworkbasic.NeuralNetworkWithBiasOptimizable;

public abstract class NeuralNetworkWithBiasOptimizerSGE extends OptimizerSGE<double[]> {
    public NeuralNetworkWithBiasOptimizerSGE(NeuralNetworkWithBias net) {
        super(new NeuralNetworkWithBiasOptimizable(net));
    }
}
