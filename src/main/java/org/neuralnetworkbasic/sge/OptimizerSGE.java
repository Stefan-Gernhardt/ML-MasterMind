package org.neuralnetworkbasic.sge;

import java.util.Arrays;

import org.neuralnetworkbasic.Optimizable;
import org.neuralnetworkbasic.Optimizer;

public abstract class OptimizerSGE<T> extends Optimizer<T> {
	

	public OptimizerSGE(Optimizable<T> optimizable) {
		super(optimizable);
	}

    /**
     * Loops over all input data
     * 
     * @param foo iteration function
     * @param input input sets
     * @param output output sets
     * @return mean cost of all training sets
     */
    protected double stochastic(IterationFunction<T> foo, T[] input, T[] output) {
    	/*
    	double lastCost = optimizable.cost(
    			Arrays.copyOfRange(input, 0, 1),
    			Arrays.copyOfRange(output, 0, 1));
    	*/
        
        // System.out.println("method::OptimizerSGE::trainingstochastic: lastcost: " + lastCost);

        // for(int i=0; keepTraining(optimizable, i, lastCost); i++) {
        for(int i=0; i<input.length; i++) {
        	// System.out.println("method::OptimizerSGE::trainingstochastic: iteration " + i);
            
            // System.out.println("method::OptimizerSGE::trainingstochastic: iteration: " + i + " input length: " + input.length);
            final int setNr = i % input.length;
            final T[] inputSet  = Arrays.copyOfRange(input, setNr, setNr + 1);
            final T[] outputSet = Arrays.copyOfRange(output, setNr, setNr + 1);
            
            /*
            int inputSetNumber = 0;
            // System.out.println("method::OptimizerSGE::trainingstochastic: iteration: " + i + "  setNumber: " + setNr + "  input data: ");
            for(T is : inputSet) {
                // System.out.println("method::OptimizerSGE::trainingstochastic: iteration: " + i + " inputSetNumber: " + inputSetNumber);
            	if(is instanceof double[]) {
            		double[] idv = (double[])is; 
                	System.out.print("         ");
            		for(double d : idv) {
                    	System.out.print("" + d + ", ");
            		}
                	System.out.println();
            	}
            	inputSetNumber++;
            }
            */
            
            // System.out.println("method::OptimizerSGE::trainingstochastic: iteration: " + i + " adapt neuronal network");
            final double currentCost = foo.apply(i, inputSet, outputSet);
            publish(optimizable, i, currentCost);
            // System.out.println("method::OptimizerSGE::trainingstochastic: iteration: " + i + " neuronal network adapted with new costs: " + currentCost);
            // lastCost = currentCost;
        }
        
        // System.out.println("method::OptimizerSGE::trainingstochastic: cost after last run " + lastCost);
        
        double totalCost = optimizable.cost(input, output);
        // System.out.println("method::OptimizerSGE::trainingstochastic: total cost: " + totalCost);
        // System.out.println("method::OptimizerSGE::trainingstochastic: mean cost: " + totalCost / (1.0*input.length));
        
        // System.out.println();
        return totalCost / (1.0*input.length);
    }
	

    public double trainStochasticGradientDescentAdam(double learningRate,
            double beta1, double beta2, T[] input, T[] output) {
    	
        
        final double[] firstMoment =
                new double[optimizable.getParameters().length];
        final double[] secondMoment =
                new double[optimizable.getParameters().length];
        
        return stochastic(
                (i, in, out) -> (gradientDescentAdam(learningRate, i,
                        beta1, beta2, in, out, firstMoment, secondMoment)),
                input, output);
    }

    public double trainStochasticGradientDescentAdam(T[] input, T[] output) {
        return stochasticGradientDescentAdam(1e-3, 0.9, 0.999, input, output);
    }
	
}
