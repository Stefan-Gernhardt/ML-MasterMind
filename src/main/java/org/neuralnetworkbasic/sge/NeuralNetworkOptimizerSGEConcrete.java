package org.neuralnetworkbasic.sge;

import org.neuralnetworkbasic.NeuralNetwork;
import org.neuralnetworkbasic.Optimizable;

public class NeuralNetworkOptimizerSGEConcrete extends NeuralNetworkOptimizerSGE {

	public NeuralNetworkOptimizerSGEConcrete(NeuralNetwork net) {
		super(net);
	}

	@Override
	public boolean keepTraining(Optimizable optimizable, int iteration, double cost) {
		return iteration < 1;
		
		/*
		System.out.println("   method::keepTraining iteration: " + iteration + "  keepTraining cost: " + cost);
		
		
		boolean keepTraining = (iteration < n);
		// boolean keepTraining = (iteration < 100000) & (cost > 0.01);
		
		if(keepTraining) {
			return true;
		}
		else {
			System.out.println("   method::keepTraining Training stopped after iterations: " + iteration);
			return false;
		}
		
		// return (iteration < 1000) & (cost > 0.1);
		*/
	}
	
	
	/*
	@Override
	public boolean keepTraining(Optimizable<double[]> optimizable, int iteration, double cost) {
	}	
*/
}
