package org.sge.mm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.neuralnetworkbasic.la.Matrix;

public class Guess {
	public String code  = "";
	public INDArray inputVector  = null;
	public INDArray outputVector = null;
	public INDArray learningVector = null;
	
	// public int index = -1;
	
	@Override
	public String toString() {
		return code;
	}
}
