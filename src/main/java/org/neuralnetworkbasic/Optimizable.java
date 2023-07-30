package org.neuralnetworkbasic;

/**
 * Interface used for mathematical optimizable classes.
 * 
 * @param <T> Input & output type
 */
public interface Optimizable<T> {
    
    /**
     * Returns all parameters arranged to an array.
     * 
     * @return parameters arranged to an array
     */
    public double[] getParameters();
    
    /**
     * Replaces all parameters with the given ones.
     * 
     * @param params new parameters
     */
    public void setParameters(double[] params);
    
    
    
    /**
     * Returns the error of this objects outputs compared to the given
     * outputs.
     * 
     * @param input input
     * @param output output
     * @return error of this objects outputs compared to the given outputs
     */
    public double cost(T[] input, T[] output);
    
    /**
     * Returns the derivative of the cost with respect to every parameter.
     * 
     * @param input input
     * @param output wanted output
     * @return derivative of the cost with respect to every parameter
     */
    public double[] costPrime(T[] input, T[] output);
}