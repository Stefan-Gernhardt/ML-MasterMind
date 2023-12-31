package org.neuralnetworkbasic;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;



/**
 * Mathematical optimizer.
 * Takes a single Optimizable object & optimizes it.
 * KeepTraining has to be implemented to check when the optimization process
 * is completed.
 * Override publish to output intermediate results.
 * 
 * @param <T> Input & output type
 */
public abstract class Optimizer<T> {
    
    /**
     * Object to optimize.
     */
    protected final Optimizable<T> optimizable;
    
    
    
    /**
     * Contructs a new Optimizer with the given Optimizable.
     * 
     * @param optimizable object to optimize
     */
    public Optimizer(Optimizable<T> optimizable) {
        this.optimizable = optimizable;
    }
    
    
    
    /**
     * Returns if the optimization process should continue.
     * 
     * @param optimizable object to optimize
     * @param iteration current iteration
     * @param cost current cost
     * @return if the optimization process should continue
     */
    public abstract boolean keepTraining(Optimizable<T> optimizable,
            int iteration, double cost);
    
    /**
     * Optional method that gets called every iteration to output intermediate
     * results.
     * 
     * @param optimizable object to optimize
     * @param iteration current iteration
     * @param cost current cost
     */
    public void publish(Optimizable<T> optimizable,
            int iteration, double cost) {}
    
    
    
    /**
     * Applies the given operand to all elements of the given array and
     * returns the result.
     * The given array is not modified.
     * 
     * @param operand operand
     * @param operator operator
     * @return result
     */
    protected double[] apply(double[] operand, DoubleUnaryOperator operator) {
        final double[] result = new double[operand.length];
        Arrays.setAll(result, (i) -> operator.applyAsDouble(operand[i]));
        
        return result;
    }
    
    /**
     * Applies the given operand to all elements of the given arrays
     * elementwise and returns the result.
     * The given arrays are not modified.
     * 
     * @param operand1 first operand
     * @param operand2 second operand
     * @param operator operator
     * @return result
     */
    protected double[] apply(double[] operand1, double[] operand2,
            DoubleBinaryOperator operator) {
        
        final double[] result =
                new double[Math.min(operand1.length, operand2.length)];
        Arrays.setAll(result,
                (i) -> operator.applyAsDouble(operand1[i], operand2[i]));
        
        return result;
    }
    
    
    
    /**
     * Performs a single gradient descent iteration.
     * 
     * @param learningRate learning rate
     * @param input input sets
     * @param output ouput sets
     * @return cost after iteration
     */
    protected double gradientDescent(double learningRate,
            T[] input, T[] output) {
        
        final double[] costPrime = optimizable.costPrime(input, output);
        final double[] update = apply(costPrime, (x) -> learningRate * x);
        
        final double[] oldParams = optimizable.getParameters();
        final double[] newParams = apply(oldParams, update, (x, y) -> x - y);
        optimizable.setParameters(newParams);
        
        return optimizable.cost(input, output);
    }
    
    /**
     * Performs a single gradient descent iteration with momentum.
     * 
     * @param learningRate learning rate
     * @param momentum momentum
     * @param input input sets
     * @param output ouput sets
     * @param velocity current parameter velocities (gets updated)
     * @return cost after iteration
     */
    protected double gradientDescentMomentum(double learningRate,
            double momentum, T[] input, T[] output, double[] velocity) {
        
        final double[] costPrime = optimizable.costPrime(input, output);
        final double[] velocityNew = apply(costPrime, velocity,
                (x, y) -> learningRate*x + momentum*y);
        
        final double[] oldParams = optimizable.getParameters();
        final double[] newParams = apply(oldParams, velocityNew,
                (x, y) -> x - y);
        optimizable.setParameters(newParams);
        
        System.arraycopy(velocityNew, 0, velocity, 0, velocityNew.length);
        
        return optimizable.cost(input, output);
    }
    
    /**
     * Performs a single Nesterov gradient descent iteration.
     * 
     * @param learningRate learning rate
     * @param momentum momentum
     * @param input input sets
     * @param output ouput sets
     * @param velocity current parameter velocities (gets updated)
     * @return cost after iteration
     */
    protected double gradientDescentNesterov(double learningRate,
            double momentum, T[] input, T[] output, double[] velocity) {
        
        final double[] costPrime = optimizable.costPrime(input, output);
        final double[] velocityNew = apply(velocity, costPrime,
                (x, y) -> momentum*x - learningRate*y);
        final double[] update = apply(velocity, velocityNew,
                (x, y) -> (1 + momentum)*y - momentum*x);
        
        final double[] oldParams = optimizable.getParameters();
        final double[] newParams = apply(oldParams, update, (x, y) -> x + y);
        optimizable.setParameters(newParams);
        
        System.arraycopy(velocityNew, 0, velocity, 0, velocityNew.length);
        
        return optimizable.cost(input, output);
    }
    
    /**
     * Performs a single AdaGrad gradient descent iteration.
     * 
     * @param learningRate learning rate
     * @param input input sets
     * @param output ouput sets
     * @param gradientsMean parameter gradients mean (gets updated)
     * @return cost after iteration
     */
    protected double gradientDescentAdagrad(double learningRate,
            T[] input, T[] output, double[] gradientsMean) {
        
        final double epsilon = 1e-8;
        
        final double[] costPrime = optimizable.costPrime(input, output);
        final double[] gradientsMeanNew = apply(gradientsMean, costPrime,
                (x, y) -> x + y*y);
        final double[] update = apply(costPrime, gradientsMeanNew,
                (x, y) -> learningRate / (Math.sqrt(y) + epsilon) * x);
        
        final double[] oldParams = optimizable.getParameters();
        final double[] newParams = apply(oldParams, update, (x, y) -> x - y);
        optimizable.setParameters(newParams);
        
        System.arraycopy(gradientsMeanNew, 0,
                gradientsMean, 0, gradientsMeanNew.length);
        
        return optimizable.cost(input, output);
    }
    
    /**
     * Performs a single RMSProp gradient descent iteration.
     * 
     * @param learningRate learning rate
     * @param decay decay of the gradient means
     * @param input input sets
     * @param output ouput sets
     * @param gradientsMean parameter gradients mean (gets updated)
     * @return cost after iteration
     */
    protected double gradientDescentRmsprop(double learningRate, double decay,
            T[] input, T[] output, double[] gradientsMean) {
        
        final double epsilon = 1e-8;
        
        final double[] costPrime = optimizable.costPrime(input, output);
        final double[] gradientsMeanNew = apply(gradientsMean, costPrime,
                (x, y) -> decay*x + (1 - decay)*y*y);
        final double[] update = apply(costPrime, gradientsMeanNew,
                (x, y) -> learningRate / (Math.sqrt(y) + epsilon) * x);
        
        final double[] oldParams = optimizable.getParameters();
        final double[] newParams = apply(oldParams, update, (x, y) -> x - y);
        optimizable.setParameters(newParams);
        
        System.arraycopy(gradientsMeanNew, 0,
                gradientsMean, 0, gradientsMeanNew.length);
        
        return optimizable.cost(input, output);
    }
    
    /**
     * Performs a single Adam gradient descent iteration.
     * 
     * @param learningRate learning rate
     * @param iteration current iteration
     * @param beta1 gradient average decay
     * @param beta2 squared gradient average decay
     * @param input input sets
     * @param output ouput sets
     * @param firstMoment gradient average (gets updated)
     * @param secondMoment squared gradient average (gets updated)
     * @return cost after iteration
     */
    protected double gradientDescentAdam(double learningRate, int iteration,
            double beta1, double beta2, T[] input, T[] output,
            double[] firstMoment, double[] secondMoment) {
        
        final double epsilon = 1e-8;
        
        final double[] costPrime = optimizable.costPrime(input, output);
        final double[] firstMomentNew = apply(firstMoment, costPrime,
                (x, y) -> beta1*x + (1 - beta1)*y);
        final double[] secondMomentNew = apply(secondMoment, costPrime,
                (x, y) -> beta2*x + (1 - beta2)*y*y);
        
        final double[] firstUnbias = apply(firstMomentNew,
                (x) -> x / (1 - Math.pow(beta1, iteration + 1)));
        final double[] secondUnbias = apply(secondMomentNew,
                (x) -> x / (1 - Math.pow(beta2, iteration + 1)));
        final double[] update = apply(firstUnbias, secondUnbias,
                (x, y) -> learningRate / (Math.sqrt(y) + epsilon) * x);
        
        final double[] oldParams = optimizable.getParameters();
        final double[] newParams = apply(oldParams, update, (x, y) -> x - y);
        optimizable.setParameters(newParams);
        
        System.arraycopy(firstMomentNew, 0,
                firstMoment, 0, firstMomentNew.length);
        System.arraycopy(secondMomentNew, 0,
                secondMoment, 0, secondMomentNew.length);
        
        return optimizable.cost(input, output);
    }
    
    
    /**
     * Loops while keepTraining returns true and calls the given function every
     * iteration with all input & outputs sets.
     * 
     * @param foo iteration function
     * @param input input sets
     * @param output ouput sets
     * @return last cost
     */
    protected double batch(IterationFunction<T> foo, T[] input, T[] output) {
        
        double lastCost = optimizable.cost(input, output);
        
        for(int i=0; keepTraining(optimizable, i, lastCost); i++) {
            final double currentCost = foo.apply(i, input, output);
            publish(optimizable, i, currentCost);
            lastCost = currentCost;
        }
        
        return lastCost;
    }
    
    
    
    /**
     * Loops over input data
     * 
     * @param foo iteration function
     * @param input input sets
     * @param output ouput sets
     * @return last cost
     */
    protected double stochastic(IterationFunction<T> foo, T[] input, T[] output) {
        
        // System.out.println("method::Optimizer::trainingstochastic: lastcost: " + lastCost);
    	double cost = 0;
        for(int i=0; i< input.length; i++) { // for(int i=0; keepTraining(optimizable, i, lastCost); i++) {
            // System.out.println("method::Optimizer::trainingstochastic: iteration: " + i + " input length: " + input.length);
            final int setNr = i % input.length;
            final T[] inputSet  = Arrays.copyOfRange(input, setNr, setNr + 1);
            final T[] outputSet = Arrays.copyOfRange(output, setNr, setNr + 1);
            
            /*
            int inputSetNumber = 0;
            System.out.println("method::Optimizer::trainingstochastic: iteration: " + i + "  setNumber: " + setNr + "  input data: ");
            for(T is : inputSet) {
                System.out.println("method::Optimizer::trainingstochastic: iteration: " + i + " inputSetNumber: " + inputSetNumber);
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
            
            // System.out.println("method::Optimizer::trainingstochastic: iteration: " + i + " adapt neuronal network");
            cost = foo.apply(i, inputSet, outputSet);
            publish(optimizable, i, cost);
        }
        
        // System.out.println("method::Optimizer::trainingstochastic: cost after last input set: " + lastCost);
        // System.out.println();
        return cost;
    }
  
    /**
     * Loops while keepTraining returns true and calls the given function every
     * iteration with a small batch of input & output sets.
     * 
     * @param foo iteration function
     * @param batchSize batchsize
     * @param input input sets
     * @param output ouput sets
     * @return last cost
     */
    protected double miniBatch(IterationFunction<T> foo,
            int batchSize, T[] input, T[] output) {
        
        final int numberOfBatches =
                (int)Math.ceil((double)input.length / batchSize);
        double lastCost = optimizable.cost(
                Arrays.copyOfRange(input, 0, batchSize),
                Arrays.copyOfRange(output, 0, batchSize));
        
        for(int i=0; keepTraining(optimizable, i, lastCost); i++) {
            
            final int batchNr = i % numberOfBatches;
            final int batchStart = batchSize * batchNr;
            final int batchEnd =
                    Math.min(batchSize*(batchNr + 1), input.length);
            
            final T[] inputSets =
                    Arrays.copyOfRange(input, batchStart, batchEnd);
            final T[] outputSets =
                    Arrays.copyOfRange(output, batchStart, batchEnd);
            
            final double currentCost = foo.apply(i, inputSets, outputSets);
            publish(optimizable, i, currentCost);
            lastCost = currentCost;
        }
        
        return lastCost;
    }
    
    
    
    /**
     * Performs batch gradient descent optimization.
     * 
     * @param learningRate learning rate
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double batchGradientDescent(double learningRate,
            T[] input, T[] output) {
        
        return batch(
                (i, in, out) -> gradientDescent(learningRate, in, out),
                input, output);
    }
    
    /**
     * Performs stochastic gradient descent optimization.
     * 
     * @param learningRate learning rate
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescent(double learningRate,
            T[] input, T[] output) {
        
        return stochastic(
                (i, in, out) -> gradientDescent(learningRate, in, out),
                input, output);
    }
    
    /**
     * Performs mini-batch gradient descent optimization.
     * 
     * @param learningRate learning rate
     * @param batchSize batch size
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double miniBatchGradientDescent(double learningRate, int batchSize,
            T[] input, T[] output) {
        
        return miniBatch(
                (i, in, out) -> gradientDescent(learningRate, in, out),
                batchSize, input, output);
    }
    
    
    /**
     * Performs stochastic gradient descent optimization with momentum.
     * Momentum = 0.9.
     * 
     * @param learningRate learning rate
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentMomentum(double learningRate,
            T[] input, T[] output) {
        
        return stochasticGradientDescentMomentum(
                learningRate, 0.9, input, output);
    }
    
    /**
     * Performs stochastic gradient descent optimization with momentum.
     * 
     * @param learningRate learning rate
     * @param momentum momentum
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentMomentum(double learningRate,
            double momentum, T[] input, T[] output) {
        
        final double[] velocity =
                new double[optimizable.getParameters().length];
        
        return stochastic(
                (i, in, out) -> gradientDescentMomentum(
                        learningRate, momentum, in, out, velocity),
                input, output);
    }
    
    
    /**
     * Performs Nesterov stochastic gradient descent optimization.
     * Momentum = 0.9.
     * 
     * @param learningRate learning rate
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentNesterov(double learningRate,
            T[] input, T[] output) {
        
        return stochasticGradientDescentNesterov(
                learningRate, 0.9, input, output);
    }
    
    /**
     * Performs Nesterov stochastic gradient descent optimization.
     * 
     * @param learningRate learning rate
     * @param momentum momentum
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentNesterov(double learningRate,
            double momentum, T[] input, T[] output) {
        
        final double[] velocity =
                new double[optimizable.getParameters().length];
        
        return stochastic(
                (i, in, out) -> gradientDescentNesterov(
                        learningRate, momentum, in, out, velocity),
                input, output);
    }
    
    
    /**
     * Performs AdaGrad stochastic gradient descent optimization.
     * Learning rate = 0.01.
     * 
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentAdagrad(T[] input, T[] output) {
        return stochasticGradientDescentAdagrad(0.01, input, output);
    }
    
    /**
     * Performs AdaGrad stochastic gradient descent optimization.
     * 
     * @param learningRate learning rate
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentAdagrad(double learningRate,
            T[] input, T[] output) {
        
        final double[] gradientsMean =
                new double[optimizable.getParameters().length];
        
        return stochastic(
                (i, in, out) -> gradientDescentAdagrad(
                        learningRate, in, out, gradientsMean),
                input, output);
    }
    
    
    /**
     * Performs RMSProp stochastic gradient descent optimization.
     * Learning rate = 0.01.
     * decay = 0.9.
     * 
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentRmsprop(T[] input, T[] output) {
        return stochasticGradientDescentRmsprop(0.01, 0.9, input, output);
    }
    
    /**
     * Performs RMSProp stochastic gradient descent optimization.
     * 
     * @param learningRate learning rate
     * @param decay decay
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentRmsprop(double learningRate,
            double decay, T[] input, T[] output) {
        
        final double[] gradientsMean =
                new double[optimizable.getParameters().length];
        
        return stochastic(
                (i, in, out) -> gradientDescentRmsprop(
                        learningRate, decay, in, out, gradientsMean),
                input, output);
    }
    
    
    /**
     * Performs Adam stochastic gradient descent optimization.
     * Learning rate = 0.001.
     * Beta1 = 0.9.
     * Beta2 = 0.999.
     * 
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentAdam(T[] input, T[] output) {
        return stochasticGradientDescentAdam(1e-3, 0.9, 0.999, input, output);
    }
    
    /**
     * Performs Adam stochastic gradient descent optimization.
     * 
     * @param learningRate learning rate
     * @param beta1 gradient average decay
     * @param beta2 squared gradient average decay
     * @param input input sets
     * @param output output sets
     * @return last cost
     */
    public double stochasticGradientDescentAdam(double learningRate,
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
    
    
    
    /**
     * Iteration function interface used for splitting the loops & iteration
     * functions.
     * 
     * @param <T> input & output set type
     */
    public interface IterationFunction<T> {
        /**
         * Single iteration that takes the current iteration, input & output
         * sets and optimizes the optimizable and returns the cost after the
         * iteration.
         * 
         * @param iteration current iteration
         * @param input input sets
         * @param output output sets
         * @return cost after iteration
         */
        double apply(int iteration, T[] input, T[] output);
    }
}
