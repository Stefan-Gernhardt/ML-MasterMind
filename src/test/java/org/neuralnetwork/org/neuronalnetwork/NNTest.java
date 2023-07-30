package org.neuralnetwork.org.neuronalnetwork;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.function.DoubleUnaryOperator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.neuralnetworkbasic.ActivationFunction;
import org.neuralnetworkbasic.NeuralNetwork;
import org.neuralnetworkbasic.NeuralNetworkWithBias;
import org.neuralnetworkbasic.la.Matrix;
import org.neuralnetworkbasic.sge.NeuralNetworkOptimizerSGE;
import org.neuralnetworkbasic.sge.NeuralNetworkOptimizerSGEConcrete;
import org.neuralnetworkbasic.sge.NeuralNetworkWithBiasOptimizerKeep;
import org.neuralnetworkbasic.sge.NeuralNetworkWithBiasOptimizerSGEConcrete;
import org.neuralnetworkbasic.sge.NeuralNetworkWithBiasSGE;
import org.sge.mm.ComputationUnit;
import org.sge.mm.MasterMind;

public class NNTest {

	double[][] input =  { { 
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		
		} };
	
	
	
	public Matrix f_nn(NeuralNetwork nn, double a1, double a2) {
		Matrix inputVector = new Matrix(1, 2);
		inputVector.set(0, 0, a1);
		inputVector.set(0, 1, a2);

		Matrix result = nn.apply(inputVector);
		System.out.println("f_nn:");
		System.out.println("input vector: " + inputVector.toString() + "result of nn: " + result.toString());
		return result;
	}
	

	public Matrix f_nn(NeuralNetworkWithBias nn, double a1, double a2) {
		Matrix inputVector = new Matrix(1, 2);
		inputVector.set(0, 0, a1);
		inputVector.set(0, 1, a2);

		Matrix result = nn.apply(inputVector);
		System.out.println("f_nn:");
		System.out.println("input vector: " + inputVector.toString() + "result of nn: " + result.toString());
		return result;
	}
	

	public Matrix f_nn4(NeuralNetwork nn, double a1, double a2, double a3, double a4) {
		Matrix inputVector = new Matrix(1, 4);
		inputVector.set(0, 0, a1);
		inputVector.set(0, 1, a2);
		inputVector.set(0, 2, a3);
		inputVector.set(0, 3, a4);

		Matrix result = nn.apply(inputVector);
		System.out.println("f_nn:");
		System.out.println("input vector: " + inputVector.toString() + "result of nn: " + result.toString());
		return result;
	}
	

	public Matrix f_nn4(NeuralNetworkWithBias nn, double a1, double a2, double a3, double a4) {
		Matrix inputVector = new Matrix(1, 4);
		inputVector.set(0, 0, a1);
		inputVector.set(0, 1, a2);
		inputVector.set(0, 2, a3);
		inputVector.set(0, 3, a4);

		Matrix result = nn.apply(inputVector);
		System.out.println("f_nn:");
		System.out.println("input vector: " + inputVector.toString() + "result of nn: " + result.toString());
		return result;
	}
	

	public double f_nn_cost(NeuralNetwork nn, double a1, double a2, double soll1, double soll2) {
		Matrix inputVector = new Matrix(1, 2);
		inputVector.set(0, 0, a1);
		inputVector.set(0, 1, a2);

		Matrix result = nn.apply(inputVector);
		
		double diff0 = result.get(0, 0) - soll1;
		double diff1 = result.get(0, 1) - soll2;

		double sqr0 = diff0 * diff0;
		double sqr1 = diff1 * diff1;
		
		double meanSqrError = sqr0 + sqr1;
		
		
		System.out.println("f_nn_cost:");
		System.out.println("input vector: " + inputVector.toString() + "result of nn: " + result.toString() +
                "soll  vector: [" + soll1 + ", " + soll2 + "]" + "\n" +
                "diff  vector: [" + diff0 + ", " + diff1 + "]");
		System.out.println("sqr   vector: [" + sqr0 + ", " + sqr1 + "]");
		System.out.println("meanSqrErSum: " + meanSqrError);
		System.out.println("meanSqrError: " + meanSqrError/2);
		System.out.println();
		
		return meanSqrError/2;
	}
	

	public Matrix calculate2LayersRelu(double v1, double v2, Matrix weightsLayer1, Matrix weightsLayer2) {
		double n1 = v1 * weightsLayer1.get(0, 0) + v2 * weightsLayer1.get(1, 0); 
		double n2 = v1 * weightsLayer1.get(0, 1) + v2 * weightsLayer1.get(1, 1); 
		
		if(n1 < 0) n1 = 0;
		if(n2 < 0) n2 = 0;
		
		System.out.println("n1 n2: " + n1 + "  " + n2);

		double o1 = n1 * weightsLayer2.get(0, 0) + n2 * weightsLayer2.get(1, 0); 
		double o2 = n1 * weightsLayer2.get(0, 1) + n2 * weightsLayer2.get(1, 1); 
		
		if(o1 < 0) o1 = 0;
		if(o2 < 0) o2 = 0;

		System.out.println("o1 o2: " + o1 + "  " + o2);
		
		Matrix result = new Matrix(1,2);
		result.set(0, 0, o1);
		result.set(0, 1, o2);
		
		return result;
	}
	
	
	@Test
	public void test01_propagate() {
		int numberInputNeurons = 2;
		int[] layerSizes = { 2, 2 };

		ActivationFunction reluActivationFunction = ActivationFunction.RELU;
		DoubleUnaryOperator reluOperator = reluActivationFunction.get();
		DoubleUnaryOperator reluOperatorDerived = reluActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { reluOperator, reluOperator };
		DoubleUnaryOperator[] activationFunctionsDerived = { reluOperatorDerived, reluOperatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions,
				activationFunctionsDerived);

		Matrix layer1 = nn.getWeights(0);
		Matrix layer2 = nn.getWeights(1);
		System.out.println("Matrix s weights1 : \n" + layer1.toString());
		System.out.println("Matrix s weights2 : \n" + layer2.toString());
		
		// null vector (0,0) test
		Matrix resultNullVector = f_nn(nn, 0, 0);
		assertEquals(0, resultNullVector.get(0, 0), 0.001);
		assertEquals(0, resultNullVector.get(0, 1), 0.001);
		
		// vector (1,1) test
		Matrix resultHand = calculate2LayersRelu(1, 1, layer1, layer2);
		Matrix resultAI   = f_nn(nn, 1, 1);
		assertEquals(resultHand.get(0, 0), resultAI.get(0, 0), 0.001);
		assertEquals(resultHand.get(0, 1), resultAI.get(0, 1), 0.001);
	}

	
	public void printWeightsOfNN(NeuralNetwork nn) {
		Matrix layer1 = nn.getWeights(0);
		Matrix layer2 = nn.getWeights(1);
		System.out.println("Matrix s weights1 : \n" + layer1.toString());
		System.out.println("Matrix s weights2 : \n" + layer2.toString());
	}
	
	
	public void printWeightsOfNN(NeuralNetworkWithBias nn) {
		Matrix layer1 = nn.getWeights(0);
		Matrix layer2 = nn.getWeights(1);
		Matrix b1      = nn.getBiases(0);
		Matrix b2      = nn.getBiases(1);
		
		System.out.println("Matrix s weights1 : \n" + layer1.toString());
		System.out.println("Matrix s biases1  : \n" + b1.toString());
		System.out.println("Matrix s weights2 : \n" + layer2.toString());
		System.out.println("Matrix s biases2  : \n" + b2.toString());
	}
	
	
	@Test
	public void test02_basicFunctionalityNeuralNetwork() {
		int numberInputNeurons = 2;
		int[] layerSizes = { 2, 2 };

		ActivationFunction tanhActivationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator tanhdOperator = tanhActivationFunction.get();
		DoubleUnaryOperator tanhOperatorDerived = tanhActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { tanhdOperator, tanhdOperator };
		DoubleUnaryOperator[] activationFunctionsDerived = { tanhOperatorDerived, tanhOperatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(656565);


		printWeightsOfNN(nn);
		
		f_nn(nn, 1,1);
		
		assertEquals(nn.getLayerSize(0), layerSizes[0]);
		assertEquals(nn.getLayerSize(1), layerSizes[1]);
		assertEquals(nn.getNumberOfInputs(), 2); 
		assertEquals(nn.getNumberOfLayers(), layerSizes.length);
		assertEquals(nn.getNumberOfOutputs(), 2);
	}
	
	
	@Test
	public void test03_testCostFunction() {
		int numberInputNeurons = 2;
		int[] layerSizes = { 2, 2 };

		ActivationFunction tanhActivationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator tanhdOperator = tanhActivationFunction.get();
		DoubleUnaryOperator tanhOperatorDerived = tanhActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { tanhdOperator, tanhdOperator };
		DoubleUnaryOperator[] activationFunctionsDerived = { tanhOperatorDerived, tanhOperatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(656565);
		printWeightsOfNN(nn);
		
		f_nn(nn, 0.5, 0.5);
		double f_nn_cost = f_nn_cost(nn, 0.5, 0.5, 0.5, 0.5);
		
		assertEquals(nn.getLayerSize(0), layerSizes[0]);
		assertEquals(nn.getLayerSize(1), layerSizes[1]);
		assertEquals(nn.getNumberOfInputs(), 2); 
		assertEquals(nn.getNumberOfLayers(), layerSizes.length);
		assertEquals(nn.getNumberOfOutputs(), 2);

		Matrix inputVector = new Matrix(1, 2);
		inputVector.set(0, 0, 0.5);
		inputVector.set(0, 1, 0.5);
		
		Matrix outputVector = new Matrix(1, 2);
		outputVector.set(0, 0, 0.5);
		outputVector.set(0, 1, 0.5);
		
		double nnCost = nn.cost(inputVector, outputVector);
		
		System.out.println("nn.cost     : " + nnCost);

		assertEquals(f_nn_cost, nnCost, 0.000001);
	}
	
	
	public void setWeightsToOne(NeuralNetwork nn) {
		Matrix layer0Weigts = new Matrix(2, 2);
		layer0Weigts.set(0, 0, 1.0);
		layer0Weigts.set(0, 1, 1.0);
		layer0Weigts.set(1, 0, 1.0);
		layer0Weigts.set(1, 1, 1.0);
		
		Matrix layer1Weigts = new Matrix(2, 2);
		layer1Weigts.set(0, 0, 1.0);
		layer1Weigts.set(0, 1, 1.0);
		layer1Weigts.set(1, 0, 1.0);
		layer1Weigts.set(1, 1, 1.0);
		
		nn.setWeights(0, layer0Weigts);
		nn.setWeights(1, layer1Weigts);
	}
	
	
	public void setWeightsToZero(NeuralNetworkWithBias nn) {
		Matrix layer0Weigts = new Matrix(2, 2);
		layer0Weigts.set(0, 0, 0);
		layer0Weigts.set(0, 1, 0);
		layer0Weigts.set(1, 0, 0);
		layer0Weigts.set(1, 1, 0);
		
		Matrix layer1Weigts = new Matrix(2, 2);
		layer1Weigts.set(0, 0, 0);
		layer1Weigts.set(0, 1, 0);
		layer1Weigts.set(1, 0, 0);
		layer1Weigts.set(1, 1, 0);
		
		nn.setWeights(0, layer0Weigts);
		nn.setWeights(1, layer1Weigts);
		
		Matrix bias = new Matrix(1, 2);
		bias.set(0, 0, 0);
		bias.set(0, 1, 0);
		
		nn.setBiases(0, bias);
		nn.setBiases(1, bias);
	}
	
	
	@Test
	public void test04_InputLayer2HiddenLayer2OutputLayer2_LinearFunctionWithoutBiasReluTwoInputTupel() {
		int numberInputNeurons = 2;
		int[] layerSizes = { 2, 2 };

		ActivationFunction reluhActivationFunction = ActivationFunction.RELU;
		DoubleUnaryOperator operator = reluhActivationFunction.get();
		DoubleUnaryOperator operatorDerived = reluhActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		setWeightsToOne(nn);
		printWeightsOfNN(nn);
		
		f_nn(nn, 0.5, 0.5);
		f_nn(nn, 0.7, 0.7);
		f_nn_cost(nn, 0.5, 0.5, 0.5, 0.5);
		
		NeuralNetworkOptimizerSGE nno = new NeuralNetworkOptimizerSGEConcrete(nn);
		  
		double[][] input =   { { 0.5, 0.5 }, { 0.7, 0.7 } };
		double[][] output =  { { 0.5, 0.5 }, { 0.7, 0.7 } };
		double[][] input05 =   { { 0.5, 0.5 } };
		double[][] output05=   { { 0.5, 0.5 } };
		double[][] input07 =   { { 0.7, 0.7 } };
		double[][] output07 =  { { 0.7, 0.7 } };
		Matrix inputMatrix = new Matrix(input);
		Matrix input05Matrix = new Matrix(input05);
		Matrix input07Matrix = new Matrix(input07);
		Matrix outputMatrix = new Matrix(output);
		Matrix output05Matrix = new Matrix(output05);
		Matrix output07Matrix = new Matrix(output07);
		
		
		double meanCost = nno.trainStochasticGradientDescentAdam(0.1, 0.9, 0.999, input, output);

		printWeightsOfNN(nn);
		assertEquals(nn.cost(input05Matrix, output05Matrix), 0.6094819881180416, 0000.1);
		assertEquals(nn.cost(input07Matrix, output07Matrix), 1.1945846967113614, 0000.1);
		assertEquals(nn.cost(inputMatrix, outputMatrix), 1.804066684829403, 0000.1);
		
		System.out.println("sge: " + meanCost);
		assertEquals(0.9020333424147015, meanCost, 0.00001);
	}
	
	
	@Test
	public void test04_InputLayer2HiddenLayer2OutputLayer2_LinearFunctionWithoutBiasReluTwoInputTupelNegativeOutput() {
		int numberInputNeurons = 2;
		int[] layerSizes = { 2, 2 };

		ActivationFunction reluhActivationFunction = ActivationFunction.RELU;
		DoubleUnaryOperator operator = reluhActivationFunction.get();
		DoubleUnaryOperator operatorDerived = reluhActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(4);
		NeuralNetworkOptimizerSGE nno = new NeuralNetworkOptimizerSGEConcrete(nn);
		printWeightsOfNN(nn);
		
		f_nn(nn, 0.5, 0.5);
		
		  
		double[][] input05 =  { { 0.5, 0.5 } };
		double[][] output05 = { { 0.5, 0.5 } };

		for(int i=0; i<500; i++) {
			nno.trainStochasticGradientDescentAdam(input05, output05);
		}

		f_nn(nn, 0.5, 0.5);
		printWeightsOfNN(nn);

		Matrix result = nn.apply(new Matrix(input05));
		System.out.println(result.get(0, 0));
		assertEquals(result.get(0, 0), 0.5, 00.1);
		System.out.println(result.get(0, 1));
		assertEquals(result.get(0, 1), 0.5, 00.1);
	}
	
	
	@Test
	public void test05_InputLayer2HiddenLayer2OutputLayer2_LinearFunctionWithoutBiasReluTwoInputTupelLoop() {
		int numberInputNeurons = 2;
		int[] layerSizes = { 2, 2 };

		ActivationFunction reluhActivationFunction = ActivationFunction.RELU;
		DoubleUnaryOperator operator = reluhActivationFunction.get();
		DoubleUnaryOperator operatorDerived = reluhActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		setWeightsToOne(nn);
		printWeightsOfNN(nn);
		
		NeuralNetworkOptimizerSGE nno = new NeuralNetworkOptimizerSGEConcrete(nn);
		  
		double[][] input =  { { 0.5, 0.5 }, { 0.7, 0.7 } };
		double[][] output = { { 0.5, 0.5 }, { 0.7, 0.7 } };
		
		double meanCost = 0;
		for(int trainingRun=0; trainingRun<500; trainingRun++) {
			meanCost = nno.trainStochasticGradientDescentAdam(input, output);
			
			System.out.println("" + trainingRun + "   mean cost: " + meanCost);
		}

		printWeightsOfNN(nn);
		
		f_nn(nn, 0.5, 0.5);
		f_nn(nn, 0.7, 0.7);
		
		double[][] input05 =  { { 0.5, 0.5 } };
		double[][] output05 = { { 0.5, 0.5 } };
		double[][] input07 =  { { 0.7, 0.7 } };
		double[][] output07 = { { 0.7, 0.7 } };
		Matrix input05Matrix = new Matrix(input05);
		Matrix input07Matrix = new Matrix(input07);
		Matrix output05Matrix = new Matrix(output05);
		Matrix output07Matrix = new Matrix(output07);
		
		f_nn_cost(nn, 0.5, 0.5, 0.5, 0.5);
		f_nn_cost(nn, 0.7, 0.7, 0.7, 0.7);
		
		assertEquals(nn.cost(input05Matrix, output05Matrix), 0, 0000.1);
		assertEquals(nn.cost(input07Matrix, output07Matrix), 0, 0000.1);
	}	
	
	
	@Test
	public void test06_InputLayer2HiddenLayer2OutputLayer2_WithoutBiasRelu() {
		int numberInputNeurons = 2;
		int[] layerSizes = { 2, 2 };

		ActivationFunction reluhActivationFunction = ActivationFunction.RELU;
		DoubleUnaryOperator operator = reluhActivationFunction.get();
		DoubleUnaryOperator operatorDerived = reluhActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		setWeightsToOne(nn);

		NeuralNetworkOptimizerSGE nno = new NeuralNetworkOptimizerSGEConcrete(nn);
		  
		double[][] input =  { { 0.6, 0.9 } };
		double[][] output = { { 0.4, 0.2 } };
		
		double meanCost = 0;
		for(int trainingRun=0; trainingRun<100; trainingRun++) {
			meanCost = nno.trainStochasticGradientDescentAdam(input, output);
			
			System.out.println("" + trainingRun + "   mean cost: " + meanCost);
		}

		double c = f_nn_cost(nn, 0.6, 0.9, 0.4, 0.2);
		
		System.out.println("c: " + c);
		
		assertEquals(meanCost, c, 0.00001);
	}
	

	
	@Test
	public void test07_InputLayer4HiddenLayer8OutputLayer8_WithoutBiasRelu() {
		int numberInputNeurons = 4;
		int[] layerSizes = { 8, 8 };

		ActivationFunction reluhActivationFunction = ActivationFunction.RELU;
		DoubleUnaryOperator operator = reluhActivationFunction.get();
		DoubleUnaryOperator operatorDerived = reluhActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };


		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(4325234);

		NeuralNetworkOptimizerSGE nno = new NeuralNetworkOptimizerSGEConcrete(nn);
		  
		double[][] input =  { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
		double[][] output = { 
				{ 0,0,0,0,1,0,0,0 }, 
				{ 0,0,1,0,0,0,0,0 },
				{ 0,0,0,0,0,0,1,0 },
				{ 1,0,0,0,0,0,0,0 }, };
		
		double meanCost = 0;
		for(int trainingRun=0; trainingRun<100; trainingRun++) {
			meanCost = nno.trainStochasticGradientDescentAdam(input, output);
			
			System.out.println("" + trainingRun + "   mean cost: " + meanCost);
		}

		f_nn4(nn, 1, 0, 0, 0);
		f_nn4(nn, 0, 1, 0, 0);
		f_nn4(nn, 0, 0, 1, 0);
		f_nn4(nn, 0, 0, 0, 1);
		
		assertTrue("meanCost < 0.4" + meanCost, meanCost < 0.4);
	}
	
	
	@Test
	public void test08_InputLayer4HiddenLayer8OutputLayer8_WithoutBiasTanh() {
		int numberInputNeurons = 4;
		int[] layerSizes = { 8, 8 };

		ActivationFunction tanhActivationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator tanhdOperator = tanhActivationFunction.get();
		DoubleUnaryOperator tanhOperatorDerived = tanhActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { tanhdOperator, tanhdOperator };
		DoubleUnaryOperator[] activationFunctionsDerived = { tanhOperatorDerived, tanhOperatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(656565);
		

		NeuralNetworkOptimizerSGE nno = new NeuralNetworkOptimizerSGEConcrete(nn);
		  
		double[][] input =  { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
		double[][] output = { 
				{ 0,0,0,0,1,0,0,0 }, 
				{ 0,0,1,0,0,0,0,0 },
				{ 0,0,0,0,0,0,1,0 },
				{ 1,0,0,0,0,0,0,0 }, };
		
		double meanCost = 0;
		for(int trainingRun=0; trainingRun<1000; trainingRun++) {
			meanCost = nno.trainStochasticGradientDescentAdam(input, output);
			
			System.out.println("" + trainingRun + "   mean cost: " + meanCost);
		}

		f_nn4(nn, 1, 0, 0, 0);
		f_nn4(nn, 0, 1, 0, 0);
		f_nn4(nn, 0, 0, 1, 0);
		f_nn4(nn, 0, 0, 0, 1);
		
		assertTrue("meanCost < 0.0001 " + meanCost, meanCost < 0.001);
	}
	

	@Test
	public void test09_InputLayer2HiddenLayer4OutputLayer2_ConstantFunction() {
		int numberInputNeurons = 2;
		int[] layerSizes = { 4, 2 };

		ActivationFunction sigmoidActivationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator sigmoidOperator = sigmoidActivationFunction.get();
		DoubleUnaryOperator sigmoidOperatorDerived = sigmoidActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { sigmoidOperator, sigmoidOperator };
		DoubleUnaryOperator[] activationFunctionsDerived = { sigmoidOperatorDerived, sigmoidOperatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(656565);

		Matrix s[] = nn.getWeights();
		System.out.println("Matrix s weights1: \n" + s[0].toString());
		System.out.println("Matrix s weights2: \n" + s[1].toString());

		assertEquals(nn.getLayerSize(0), layerSizes[0]);
		assertEquals(nn.getLayerSize(1), layerSizes[1]);

		assertEquals(nn.getNumberOfInputs(), 2);
		assertEquals(nn.getNumberOfLayers(), layerSizes.length);

		assertEquals(nn.getNumberOfOutputs(), 2);

		NeuralNetworkOptimizerSGE nno = new NeuralNetworkOptimizerSGEConcrete(nn);

		double[][] input = { { 0, 0 } };
		double[][] output = { { 0.3, 0.7 } };

		double meanCost = 0;
		for(int trainingRun=0; trainingRun<100; trainingRun++) {
			meanCost = nno.trainStochasticGradientDescentAdam(input, output);
			
			System.out.println("" + trainingRun + "   mean cost: " + meanCost);
		}
		

		Matrix w[] = nn.getWeights();
		System.out.println("Matrix w weights1: \n" + w[0].toString());
		System.out.println("Matrix w weights2: \n" + w[1].toString());

		double[][] testinput = { { 0, 0 } };
		Matrix testInputMatrix = new Matrix(testinput);
		Matrix testResult = nn.apply(testInputMatrix);
		System.out.println("result of { 0.0, 0.0 }  \n" + testResult.toString());
	}
	
	
	@Test
	public void test10_InputLayer4_HiddenLayer8_OutputLayer8_WithoutBiasSigmoid() {
		int numberInputNeurons = 4;
		int[] layerSizes = { 8, 8 };

		ActivationFunction sigmoidActivationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator sigmoidOperator = sigmoidActivationFunction.get();
		DoubleUnaryOperator sigmoidOperatorDerived = sigmoidActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { sigmoidOperator, sigmoidOperator };
		DoubleUnaryOperator[] activationFunctionsDerived = { sigmoidOperatorDerived, sigmoidOperatorDerived };

		NeuralNetwork nn = new NeuralNetwork(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(989898);

		NeuralNetworkOptimizerSGE nno = new NeuralNetworkOptimizerSGEConcrete(nn);
		  
		double[][] input =  { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
		double[][] output = { 
				{ 0,0,0,0,1,0,0,0 }, 
				{ 0,0,1,0,0,0,0,0 },
				{ 0,0,0,0,0,0,1,0 },
				{ 1,0,0,0,0,0,0,0 }, };
		
		double meanCost = 0;
		for(int trainingRun=0; trainingRun<100; trainingRun++) {
			meanCost = nno.trainStochasticGradientDescentAdam(input, output);
			
			System.out.println("" + trainingRun + "   mean cost: " + meanCost);
		}

		f_nn4(nn, 1, 0, 0, 0);
		f_nn4(nn, 0, 1, 0, 0);
		f_nn4(nn, 0, 0, 1, 0);
		f_nn4(nn, 0, 0, 0, 1);
		
		assertTrue("meanCost < 0.6 " + meanCost, meanCost < 0.6);
	}

	
	@Test
	public void test11_InputLayer4_HiddenLayer8_OutputLayer8_WithBiasSigmoid() {
		int numberInputNeurons = 4;
		int[] layerSizes = { 8, 8 };

		ActivationFunction sigmoidActivationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator sigmoidOperator = sigmoidActivationFunction.get();
		DoubleUnaryOperator sigmoidOperatorDerived = sigmoidActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { sigmoidOperator, sigmoidOperator };
		DoubleUnaryOperator[] activationFunctionsDerived = { sigmoidOperatorDerived, sigmoidOperatorDerived };

		NeuralNetworkWithBias nn = new NeuralNetworkWithBias(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(12121212);

		NeuralNetworkWithBiasOptimizerSGEConcrete nno = new NeuralNetworkWithBiasOptimizerSGEConcrete(nn);
		  
		double[][] input =  { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
		double[][] output = { 
				{ 0,0,0,0,1,0,0,0 }, 
				{ 0,0,1,0,0,0,0,0 },
				{ 0,0,0,0,0,0,1,0 },
				{ 1,0,0,0,0,0,0,0 }, };
		
		double meanCost = 0;
		for(int trainingRun=0; trainingRun<100; trainingRun++) {
			meanCost = nno.trainStochasticGradientDescentAdam(input, output);
			
			System.out.println("" + trainingRun + "   mean cost: " + meanCost);
		}

		f_nn4(nn, 1, 0, 0, 0);
		f_nn4(nn, 0, 1, 0, 0);
		f_nn4(nn, 0, 0, 1, 0);
		f_nn4(nn, 0, 0, 0, 1);
		
		assertTrue("meanCost < 0.5 " + meanCost, meanCost < 0.5);
	}
	
	
	@Test
	public void test12_InputLayer2_HiddenLayer2_OutputLayer2_WithBiasSigmoid() {
		int numberInputNeurons = 2;
		int[] layerSizes = { 2, 2 };

		ActivationFunction sigmoidActivationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator sigmoidOperator = sigmoidActivationFunction.get();
		DoubleUnaryOperator sigmoidOperatorDerived = sigmoidActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { sigmoidOperator, sigmoidOperator };
		DoubleUnaryOperator[] activationFunctionsDerived = { sigmoidOperatorDerived, sigmoidOperatorDerived };
		
		NeuralNetworkWithBias nn = new NeuralNetworkWithBias(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		setWeightsToZero(nn);
		printWeightsOfNN(nn);

		double[][] input = { { 0.53, 0.55 }, { 0.51, 0.52} };
		double[][] output = { { 0.33, 0.75}, { 0.31, 0.76 } };	

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		
		for(int i=0; i<100; i++) {
			double d = nno.stochasticGradientDescentAdam(input, output);
			System.out.println("i: " + i + "   error: " + d);
		}
		
		
		printWeightsOfNN(nn);
		f_nn(nn, 0, 0);		
		f_nn(nn, 1, 0);		
	}
	
	
	@Test
	public void test13_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasSigmoid() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 4, 4 };

		ActivationFunction sigmoidActivationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator sigmoidOperator = sigmoidActivationFunction.get();
		DoubleUnaryOperator sigmoidOperatorDerived = sigmoidActivationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { sigmoidOperator, sigmoidOperator };
		DoubleUnaryOperator[] activationFunctionsDerived = { sigmoidOperatorDerived, sigmoidOperatorDerived };
		
		NeuralNetworkWithBias nn = new NeuralNetworkWithBias(numberInputNeurons, layerSizes, activationFunctions,
				activationFunctionsDerived);

		double[][] input = { { 0, 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0, 0 } };
		double[][] output = { { 1, 0, 0, 0}, { 0, 0, 1, 0} };	

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		nn.seedWeightsXavier(656565);
		
		
		for(int i=0; i<100; i++) {
			double d = nno.stochasticGradientDescentAdam(input, output);
			System.out.println("i: " + i + "   error: " + d);
		}

		Matrix inputVector = new Matrix(input);
		Matrix result = nn.apply(new Matrix(input));
		System.out.println("input vector: " + inputVector.toString() + "result of nn: " + result.toString());
	}
	
	
	
	@Test
	public void test14_1_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasSigmoidTrainToZeroVector() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(0);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		Matrix inputMatrix = new Matrix(input);

		double[][] outputTraining =  { { 0, 0, 0, 0 }  };
		Matrix outputTrainingMatrix = new Matrix(outputTraining);

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		Matrix resultMatrix = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + resultMatrix.toString());
		System.out.println("training vector:\n" + outputTrainingMatrix.toString());

		Matrix rMatrix = null;
		for(int i=0; i<250; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), outputTrainingMatrix.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training: " + "\n" + rMatrix.toString());
		
		assertTrue(rMatrix.get(0, 0) < 0.4);
		assertTrue(rMatrix.get(0, 1) < 0.1);
		assertTrue(rMatrix.get(0, 2) < 0.2);
		assertTrue(rMatrix.get(0, 3) < 0.2);
	}	
	
	
	@Test
	public void test14_2_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasSigmoidTrainToOneVector() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(0);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		Matrix inputMatrix = new Matrix(input);

		double[][] outputTraining =  { { 1, 1, 1, 1 }  };
		Matrix outputTrainingMatrix = new Matrix(outputTraining);

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		Matrix resultMatrix = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + resultMatrix.toString());
		System.out.println("training vector:\n" + outputTrainingMatrix.toString());

		Matrix rMatrix = null;
		for(int i=0; i<250; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), outputTrainingMatrix.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training: " + "\n" + rMatrix.toString());
		
		assertTrue(rMatrix.get(0, 0) > 0.9);
		assertTrue(rMatrix.get(0, 1) > 0.4);
		assertTrue(rMatrix.get(0, 2) > 0.7);
		assertTrue(rMatrix.get(0, 3) > 0.8);
	}	
	
	
	@Test
	public void test14_3_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasSigmoidTrainToSpecificValueZero() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(0);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		Matrix inputMatrix = new Matrix(input);

		double[][] outputTraining =  { { 0.00, 0.13, 0.34, 0.44 }  };
		Matrix outputTrainingMatrix = new Matrix(outputTraining);

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		Matrix resultMatrix = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + resultMatrix.toString());
		System.out.println("training vector:\n" + outputTrainingMatrix.toString());

		Matrix rMatrix = null;
		for(int i=0; i<500; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), outputTrainingMatrix.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training: " + "\n" + rMatrix.toString());
		
		assertTrue(rMatrix.get(0, 0) < 0.11);
	}	
	
	
	@Test
	public void test14_4_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasSigmoidTrainToSpecificValueOne() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(0);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		Matrix inputMatrix = new Matrix(input);

		double[][] outputTraining =  { { 1.00, 0.13, 0.34, 0.44 }  };
		Matrix outputTrainingMatrix = new Matrix(outputTraining);

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		Matrix resultMatrix = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + resultMatrix.toString());
		System.out.println("training vector:\n" + outputTrainingMatrix.toString());

		Matrix rMatrix = null;
		for(int i=0; i<500; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), outputTrainingMatrix.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training: " + "\n" + rMatrix.toString());
		
		assertTrue(rMatrix.get(0, 0) > 0.9);
	}		
	
	@Test
	public void test14_5_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasSigmoidTrain() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(0);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		Matrix inputMatrix = new Matrix(input);

		double[][] outputTraining =  { { 0.76, 0.00, 0.00, 0.00 }  };
		Matrix outputTrainingMatrix = new Matrix(outputTraining);

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		Matrix resultMatrix = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + resultMatrix.toString());
		System.out.println("training vector:\n" + outputTrainingMatrix.toString());

		Matrix rMatrix = null;
		for(int i=0; i<500; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), outputTrainingMatrix.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training: " + "\n" + rMatrix.toString());
		
		assertTrue(rMatrix.get(0, 1) < 0.1);
		assertTrue(rMatrix.get(0, 2) < 0.1);
		assertTrue(rMatrix.get(0, 3) < 0.1);
	}	
	
	
	@Test
	public void test14_6_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasSigmoidTrain() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.SIGMOID;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(0);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		Matrix inputMatrix = new Matrix(input);

		double[][] outputTraining =  { { 0.76, 1.00, 1.00, 1.00 }  };
		Matrix outputTrainingMatrix = new Matrix(outputTraining);

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		Matrix resultMatrix = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + resultMatrix.toString());
		System.out.println("training vector:\n" + outputTrainingMatrix.toString());

		Matrix rMatrix = null;
		for(int i=0; i<500; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), outputTrainingMatrix.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training: " + "\n" + rMatrix.toString());
		
		assertTrue(rMatrix.get(0, 1) > 0.8);
		assertTrue(rMatrix.get(0, 2) > 0.9);
		assertTrue(rMatrix.get(0, 3) > 0.9);
	}	
	
	
	public int getWinner(NeuralNetworkWithBias nn, double input[], boolean verbose) {
		Matrix inputVector = new Matrix(1, 8);
		
		for(int c=0; c<8; c++) {
			inputVector.set(0, c, input[c]);
		}
		
		Matrix result = nn.apply(inputVector);
		if(verbose) System.out.println("input vector:\n" + inputVector.toString() + "result of nn:\n" + result.toString());
		
		double winnerValue = -2;
		int winner = -1;
		for(int c=0; c<4; c++) {
			double resultValue = result.get(0, c);
			if(c==0) {
				winner = c;
				winnerValue = resultValue; 
			}
			else {
				if(resultValue > winnerValue) {
					winner = c;
					winnerValue = resultValue; 
				}
			}
			
		}
		
		return winner;
	}
	
	
	@Test
	public void test14_InputLayer8_HiddenLayer4_OutputLayer4_WithBiasTanh() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 4, 4 };

		ActivationFunction activationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBias nn = new NeuralNetworkWithBias(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(1234);

		double[][] input = {  { 0, 1, 0, 0, 0, 0, 0, 0 },  { 0, 0, 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 1, 1, 1, 1 } };
		double[][] output = { { 1, 0, 0, 0},               { 0, 1, 0, 0},              { 0, 0, 1, 0} };	

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		
		for(int i=0; i<500; i++) {
			double d = nno.stochasticGradientDescentAdam(input, output);
			System.out.println("i: " + i + "   error: " + d);
		}

		Matrix inputVector = new Matrix(input);
		Matrix result = nn.apply(inputVector);
		System.out.println("input vector:\n" + inputVector.toString() + "result of nn:\n" + result.toString());

		System.out.println("winner value input0: "  + getWinner(nn, input[0], false));
		System.out.println("winner value input1: "  + getWinner(nn, input[1], false));
		System.out.println("winner value input2: "  + getWinner(nn, input[2], false));
		
		assertEquals(0, getWinner(nn, input[0], false));
		assertEquals(1, getWinner(nn, input[1], false));
		assertEquals(2, getWinner(nn, input[2], false));
	}
	
	
	@Test
	public void test16_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanh_Save() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 4, 4 };

		ActivationFunction activationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE("Input8Hidden8Output4", numberInputNeurons, layerSizes, activationFunctions,
				activationFunctionsDerived);
		
		System.out.println("Matrix s weights1 after initialization: \n" + nn.getWeights(0).toString());
		
		double initialized = nn.getWeights(0).get(0, 0);
		nn.save();

		double[][] input = { { 0, 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0, 0 }, { 1, 0, 0, 0, 0, 1, 0, 1 } };
		double[][] output = { { 1, 0, 0, 0}, { 0, 0, 1, 0}, { 0, 0, 0, 1} };	

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		
		for(int i=0; i<100; i++) {
			double d = nno.stochasticGradientDescentAdam(input, output);
			System.out.println("i: " + i + "   error: " + d);
		}

		System.out.println("Matrix s weights1 after training: \n" + nn.getWeights(0).toString());
		
		double trained = nn.getWeights(0).get(0, 0);

		nn.load();
		System.out.println("Matrix s weights1 after loading: \n" + nn.getWeights(0).toString());
		
		double loaded = nn.getWeights(0).get(0, 0);
		
		System.out.println("initialized: " + initialized);
		System.out.println("trained: " + trained);
		System.out.println("loaded: " + loaded);

		assertEquals(initialized, loaded, 0.000001);
	}	

	
	@Test
	public void test17_InputLayer8_HiddenLayer4_OutputLayer4_WithBiasTanh() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 4, 4 };

		ActivationFunction activationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(1);

		double[][] input = { { 1, 1, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 1, 1, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1, 1 } };
		double[][] output = { { 1, 0, 0, 0}, { 0, 1, 0, 0}, { 0, 0, 1, 0} };	

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		
		for(int i=0; i<500; i++) {
			double d = nno.stochasticGradientDescentAdam(input, output);
			System.out.println("i: " + i + "   error: " + d);
		}

		System.out.println("winner value input0: "  + getWinner(nn, input[0], false) + " should 0");
		System.out.println("winner value input1: "  + getWinner(nn, input[1], false) + " should 1");
		System.out.println("winner value input2: "  + getWinner(nn, input[2], false) + " should 2");
		assertEquals(0, getWinner(nn, input[0], false));
		assertEquals(1, getWinner(nn, input[1], false));
		assertEquals(2, getWinner(nn, input[2], false));
		
		double[][] input2 =  { { 0, 0, 1, 1, 0, 0, 0, 0 } };
		double[][] output2 = { { 0, 0, 0, 1} };	


		for(int i=0; i<100; i++) {
			double d = nno.stochasticGradientDescentAdam(0.1, 0.9, 0.999, input2, output2); // 0.5 not ok, 0.1 ok 0.05 notok  
			System.out.println("i: " + i + "   error: " + d);
		}

		System.out.println("winner value input3: "  + getWinner(nn, input2[0], false) + " should 1");
		assertEquals(3, getWinner(nn, input2[0], false));
	}	

	
	@Test
	public void test18_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanh() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(24324);

		double[][] input =  { { 1, 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 1, 1, 0, 0, 0, 0 }, { 0, 0, 0, 0, 1, 1, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1, 1 }  };
		double[][] output = { { 1, 0, 0, 0},              { 0, 1, 0, 0},              { 0, 0, 1, 0},              { 0, 0, 0, 1} };	

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		
		System.out.println("before training:");
		System.out.println("winner value input0: "  + getWinner(nn, input[0], false) + " should 0");
		System.out.println("winner value input1: "  + getWinner(nn, input[1], false) + " should 1");
		System.out.println("winner value input2: "  + getWinner(nn, input[2], false) + " should 2");
		System.out.println("winner value input3: "  + getWinner(nn, input[3], false) + " should 3");

		for(int i=0; i<500; i++) {
			nno.stochasticGradientDescentAdam(input, output);
			boolean expectedResult = (getWinner(nn, input[0], false) == 0) 
					&& (getWinner(nn, input[1], false) == 1) 
					&& (getWinner(nn, input[2], false) == 2) 
					&& (getWinner(nn, input[3], false) == 3); 
			if(expectedResult) {
				// System.out.println("found in round: " + i);
			}
		}

		
		System.out.println("after training:");
		System.out.println("winner value input0: "  + getWinner(nn, input[0], false) + " should 0");
		System.out.println("winner value input1: "  + getWinner(nn, input[1], false) + " should 1");
		System.out.println("winner value input2: "  + getWinner(nn, input[2], false) + " should 2");
		System.out.println("winner value input3: "  + getWinner(nn, input[3], false) + " should 3");

		assertEquals(0, getWinner(nn, input[0], false));
		assertEquals(1, getWinner(nn, input[1], false));
		assertEquals(2, getWinner(nn, input[2], false));
		assertEquals(3, getWinner(nn, input[3], false));
	}


	
	@Test
	public void test19_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanh() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(24324);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }, { 0, 1, 0, 0, 0, 1, 0, 0 }, { 0, 0, 1, 0, 0, 0, 1, 0 }, { 0, 0, 0, 1, 0, 0, 0, 1 }  };
		double[][] output = { { 1, 0, 0, 0},              { 0, 1, 0, 0},              { 0, 0, 1, 0},              { 0, 0, 0, 1} };	

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		
		System.out.println("before training:");
		System.out.println("winner value input0: "  + getWinner(nn, input[0], false) + " should 0");
		System.out.println("winner value input1: "  + getWinner(nn, input[1], false) + " should 1");
		System.out.println("winner value input2: "  + getWinner(nn, input[2], false) + " should 2");
		System.out.println("winner value input3: "  + getWinner(nn, input[3], false) + " should 3");

		for(int i=0; i<500; i++) {
			nno.stochasticGradientDescentAdam(input, output);
			boolean expectedResult = (getWinner(nn, input[0], false) == 0) 
					&& (getWinner(nn, input[1], false) == 1) 
					&& (getWinner(nn, input[2], false) == 2) 
					&& (getWinner(nn, input[3], false) == 3); 
			if(expectedResult) {
				// System.out.println("found in round: " + i);
			}
		}

		
		System.out.println("after training:");
		System.out.println("winner value input0: "  + getWinner(nn, input[0], false) + " should 0");
		System.out.println("winner value input1: "  + getWinner(nn, input[1], false) + " should 1");
		System.out.println("winner value input2: "  + getWinner(nn, input[2], false) + " should 2");
		System.out.println("winner value input3: "  + getWinner(nn, input[3], false) + " should 3");

		assertEquals(0, getWinner(nn, input[0], false));
		assertEquals(1, getWinner(nn, input[1], false));
		assertEquals(2, getWinner(nn, input[2], false));
		assertEquals(3, getWinner(nn, input[3], false));
	}
	

	@Test
	public void test20_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanh() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(24324);

		// double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }, { 0, 1, 0, 1, 0, 1, 0, 0 }, { 0, 0, 1, 0, 0, 1, 1, 0 }, { 1, 0, 1, 1, 0, 1, 1, 1 }  };
		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }, { 0, 1, 0, 1, 0, 1, 0, 0 }, { 0, 0, 1, 0, 0, 1, 1, 0 }, { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		double[][] output = { { 1, 0, 0, 0},              { 0, 1, 0, 0},              { 0, 0, 1, 0},              { 0, 0, 0, 1} };	

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		
		System.out.println("before training:");
		System.out.println("winner value input0: "  + getWinner(nn, input[0], false) + " should 0");
		System.out.println("winner value input1: "  + getWinner(nn, input[1], false) + " should 1");
		System.out.println("winner value input2: "  + getWinner(nn, input[2], false) + " should 2");
		System.out.println("winner value input3: "  + getWinner(nn, input[3], false) + " should 3");

		for(int i=0; i<1000; i++) {
			nno.stochasticGradientDescentAdam(input, output);
			boolean expectedResult = (getWinner(nn, input[0], false) == 0) 
					&& (getWinner(nn, input[1], false) == 1) 
					&& (getWinner(nn, input[2], false) == 2) 
					&& (getWinner(nn, input[3], false) == 3); 
			if(expectedResult) {
				// System.out.println("found in round: " + i);
			}
		}

		
		System.out.println("after training:");
		System.out.println("winner value input0: "  + getWinner(nn, input[0], false) + " should 0");
		System.out.println("winner value input1: "  + getWinner(nn, input[1], false) + " should 1");
		System.out.println("winner value input2: "  + getWinner(nn, input[2], false) + " should 2");
		System.out.println("winner value input3: "  + getWinner(nn, input[3], false) + " should 3");

		assertEquals(0, getWinner(nn, input[0], false));
		assertEquals(1, getWinner(nn, input[1], false));
		assertEquals(2, getWinner(nn, input[2], false));
		assertEquals(0, getWinner(nn, input[3], false));
	}

	
	@Test
	public void test21_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanhTraining() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(24324);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		
		Matrix inputMatrix = new Matrix(input);

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		Matrix resultMatrix = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + resultMatrix.toString());
		resultMatrix.set(0, 3, -1);
		
		System.out.println("before training:\n" + resultMatrix.toString());

		Matrix rMatrix = null;
		for(int i=0; i<250; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), resultMatrix.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training: " + "\n" + rMatrix.toString());
		
		assertTrue(rMatrix.get(0, 3) < -0.9);
	}

	
	@Test
	public void test22_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanhTrainToZeroVector() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(24324);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		
		Matrix inputMatrix = new Matrix(input);

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		Matrix resultMatrix = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + resultMatrix.toString());
		resultMatrix.set(0, 0, 0);
		resultMatrix.set(0, 1, 0);
		resultMatrix.set(0, 2, 0);
		resultMatrix.set(0, 3, 0);
		
		System.out.println("training vector:\n" + resultMatrix.toString());

		Matrix rMatrix = null;
		for(int i=0; i<250; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), resultMatrix.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training: " + "\n" + rMatrix.toString());
		
		assertTrue(rMatrix.get(0, 3) < 0.1);
	}
	

	@Test
	public void test22_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanhTrainToZeroValue() {
		int numberInputNeurons = 8;
		int[] layerSizes = { 8, 4 };

		ActivationFunction activationFunction = ActivationFunction.TANH;
		DoubleUnaryOperator operator = activationFunction.get();
		DoubleUnaryOperator operatorDerived = activationFunction.getPrime();

		DoubleUnaryOperator[] activationFunctions = { operator, operator };
		DoubleUnaryOperator[] activationFunctionsDerived = { operatorDerived, operatorDerived };
		
		NeuralNetworkWithBiasSGE nn = new NeuralNetworkWithBiasSGE(numberInputNeurons, layerSizes, activationFunctions, activationFunctionsDerived);
		nn.seedWeightsXavier(24324);

		double[][] input =  { { 1, 0, 0, 0, 1, 0, 0, 0 }  };
		
		Matrix inputMatrix = new Matrix(input);

		NeuralNetworkWithBiasOptimizerKeep nno = new NeuralNetworkWithBiasOptimizerKeep(nn, numberInputNeurons);
		Matrix resultMatrix = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + resultMatrix.toString());
		resultMatrix.set(0, 2, 0);
		
		System.out.println("training vector:\n" + resultMatrix.toString());

		Matrix rMatrix = null;
		for(int i=0; i<250; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), resultMatrix.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}

		System.out.println("after training: " + "\n" + rMatrix.toString());
		System.out.println("after training element: " + "\n" + rMatrix.get(0, 2));
		
		assertTrue(Math.abs(rMatrix.get(0, 2)) < 0.1);
	}
	
	
	//! todo enable again

	/*

	@Test
	public void test23_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanhTrainToZeroValue() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);
		Matrix inputMatrix = new Matrix(input);
		
		MultiLayerNetwork nn = masterMind.getComputationUnit().getNn();

		Matrix outputVector = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + outputVector.toString());
		
		outputVector.set(0, 2, 0);
		System.out.println("train vector:\n" + outputVector.toString());

		Matrix rMatrix = null;
		for(int i=0; i<50; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), outputVector.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training error sum: " + "\n" + rMatrix.toString()); 
		
		assertTrue(rMatrix.get(0,  2) < 1);
	}
	

	@Test
	public void test24_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanhTrainToSpecific() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);
		
		Matrix inputMatrix = new Matrix(input);
		
		NeuralNetworkWithBiasSGE nn = masterMind.getComputationUnit().getNn();
		NeuralNetworkWithBiasOptimizerKeep nno = masterMind.getComputationUnit().getNno();

		Matrix outputVector = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + outputVector.toString());
		
		for(int i=0; i<16; i++) outputVector.set(0, i, 0);
		outputVector.set(0, 2, 6651);
		outputVector.set(0, 4, 557);
		System.out.println("train vector:\n" + outputVector.toString());

		Matrix rMatrix = null;
		for(int i=0; i<10; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), outputVector.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		double errorSum = 0;
		for(int i=0; i<16; i++) { 
			errorSum = errorSum + Math.abs(rMatrix.get(0, i));
		}
		System.out.println("after training error sum: " + errorSum + "\n" + rMatrix.toString()); 
		
		// assertTrue(rMatrix.get(0, 3) < -0.9);
	}
	

	@Test
	public void test25_InputLayer8_HiddenLayer8_OutputLayer4_WithBiasTanhTrainToSpecific() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);
		
		Matrix inputMatrix = new Matrix(input);
		
		NeuralNetworkWithBiasSGE nn = masterMind.getComputationUnit().getNn();
		NeuralNetworkWithBiasOptimizerKeep nno = masterMind.getComputationUnit().getNno();

		Matrix outputVector = nn.apply(inputMatrix);
		System.out.println("inputMatrix: \n" + inputMatrix.toString());
		System.out.println("resultMatrix: \n" + outputVector.toString());
		
//		outputVector.set(0, 0, 1000);
//		outputVector.set(0, 1, 2000);
//		outputVector.set(0, 2, 3000);
		outputVector.set(0, 3, 8000);
//		outputVector.set(0, 4, 5000);
//		outputVector.set(0, 5, 6000);
//		outputVector.set(0, 6, 7000);
//		outputVector.set(0, 7, 8000);
//		outputVector.set(0, 8, 888);
//		outputVector.set(0, 9, 10000);
//		outputVector.set(0, 10, 11000);
//		outputVector.set(0, 11, 12000);
//		outputVector.set(0, 12, 13000);
//		outputVector.set(0, 13, 14000);
//		outputVector.set(0, 14, 15000);
//		outputVector.set(0, 15, 16000);
		System.out.println("train vector:\n" + outputVector.toString());

		Matrix rMatrix = null;
		for(int i=0; i<10; i++) {
			nno.stochasticGradientDescentAdam(inputMatrix.toArray(), outputVector.toArray());
			rMatrix = nn.apply(inputMatrix);
			// System.out.println("after training: " + i + "\n" + rMatrix.toString());
		}
		System.out.println("after training: " + "\n" + rMatrix.toString());
		
		// assertEquals(rMatrix.get(0, 0), 5000, 10);
	}
*/
}
