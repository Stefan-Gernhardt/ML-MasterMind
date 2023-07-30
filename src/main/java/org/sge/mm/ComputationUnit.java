package org.sge.mm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.RandomUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.sge.math.MathSge;

public class ComputationUnit {
	public final static int RNG_SEED= 4711;
	public final static double LEARNING_RATE = 0.001; // 0.0015

	public static final double UNSET_INPUT_NEURON_VALUE = 0.0;
	
	public static final int ALGO_NN = 0; 
	public static final int ALGO_EXCLUDE = 1; 
	public static final int ALGO_Q = 2; 
	public static final int ALGO_RANDOM = 3; 

	public static final int SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN = 0; 
	public static final int SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE = 1; 
	public static final int SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES = 2; 

	
	private int countColors=0;
	private int countDigits=0;
	private int countInputNeurons = 0;
	private int countOutputNeurons = 0;
	private int countCombinations = 0;
	private int countHiddenLayerNeurons = 0;
	private int countInputColorNeuronsPerGuess = 0;
	private int countInputRatingNeuronsPerGuess = 0;
	private int countInputNeuronsPerGuess = 0;
	
	boolean firstMoveFixed = false;
	String firstMoveFixedCode  = "";

	
	private MultiLayerNetwork nn = null;
	private MultiLayerNetwork nn2 = null;
	
	public MultiLayerNetwork getNN() {
		return nn;
	}

	public MultiLayerNetwork getNn() {
		return getNN();
	}
	public void setNN(MultiLayerNetwork mln) {
		nn = mln;
	}

	
	public ComputationUnit(int countColors, int countDigits, int mode, double constantWeight, long seed) {
		this.countColors = countColors;
		this.countDigits = countDigits;
		
		if((countColors<1) || (countColors>10) || (countDigits < 1) || (countDigits > 10)) {
			throw new RuntimeException("Invalid parameters " + countColors + " " + countDigits);
		}
		
		createNN(mode, constantWeight, seed);
	}

	
	private void createNN(int mode, double constantWeight, long seed) {
		computeCountCombinations();
		System.out.println("countCombinations: " + countCombinations);

		computeCountOutputNeurons_NeuronForEveryCombination();
		System.out.println("countOutputNeurons: " + countOutputNeurons);
		
		computeCountInputNeurons();
		System.out.println("countInputNeurons: " + countInputNeurons);
		
		countHiddenLayerNeurons = 0; // countInputNeurons; // *countInputNeurons;  
		
		createNet();
	}
	
	

	private void createNet() {
		createNetFirstLayer();
		createNetSecondLayer(countInputNeurons*2);
	}

	
	private void createNetFirstLayer() {
		Activation activation = Activation.SIGMOID;
		
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                // .seed(RNG_SEED) //include a random seed for reproducibility
                .seed((int)(Math.random() * 1000)) //include a random seed for reproducibility
                .activation(activation)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                // .l2(LEARNING_RATE * 0.001) // regularize learning model // * 0.005
                // .l2(0.0005) // ridge regression value
                .l2(LEARNING_RATE * 0.001)    // regularize learning model // * 0.005
                .list()
                .layer(new DenseLayer.Builder() //create the input and hidden layer
                        .nIn(countInputNeurons)
                        .activation(activation)
                        .nOut(countInputNeurons+countHiddenLayerNeurons)
                        .build())
                
                /*
                .layer(new DenseLayer.Builder() //create hidden layer 2
                        .nIn(countInputNeurons+countHiddenLayerNeurons)
                        .activation(activation)
                        .nOut(countInputNeurons+countHiddenLayerNeurons)
                        .build())
                */
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create output layer
                        .activation(Activation.SOFTMAX)
                        .nOut(countOutputNeurons)
                        .build())
                .build(); 	
    	
        nn = new MultiLayerNetwork(conf);
        nn.init();
	}



	private void createNetFirstLayerOutputColorCoded() {
		Activation activation = Activation.SIGMOID;
		
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                // .seed(RNG_SEED) //include a random seed for reproducibility
                .seed((int)(Math.random() * 1000)) //include a random seed for reproducibility
                .activation(activation)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                // .l2(LEARNING_RATE * 0.001) // regularize learning model // * 0.005
                // .l2(0.0005) // ridge regression value
                .l2(LEARNING_RATE * 0.001)    // regularize learning model // * 0.005
                .list()
                .layer(new DenseLayer.Builder() //create the input and hidden layer
                        .nIn(countInputNeurons)
                        .activation(activation)
                        .nOut(countInputNeurons*4)
                        .build())
                
                .layer(new DenseLayer.Builder() //create hidden layer 2
                        .nIn(countInputNeurons*4)
                        .activation(activation)
                        .nOut(countInputNeurons*4)
                        .build())
                
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create output layer
                        .activation(Activation.SOFTMAX)
                        .nOut(countOutputNeurons)
                        .build())
                .build(); 	
    	
        nn = new MultiLayerNetwork(conf);
        nn.init();
	}



	private void createNetSecondLayer(int layerCountInputNeurons) {
		Activation activation = Activation.SIGMOID;
		
		int countAdditionalNeurons = layerCountInputNeurons*4; //24
		
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed((int)(Math.random() * 1000)) //include a random seed for reproducibility
                .activation(activation)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .l2(LEARNING_RATE * 0.001)    // regularize learning model // * 0.005
                .list()
                .layer(new DenseLayer.Builder() //create the input and hidden layer
                        .nIn(layerCountInputNeurons)
                        .activation(activation)
                        .nOut(layerCountInputNeurons+countAdditionalNeurons)
                        .build())
                /*
                .layer(new DenseLayer.Builder() //create hidden layer
                        .nIn(layerCountInputNeurons+countAdditionalNeurons)
                        .activation(activation)
                        .nOut(layerCountInputNeurons+countAdditionalNeurons)
                        .build())
                        */
                .layer(new OutputLayer.Builder() //create output layer
                        .activation(activation)
                        .nOut(countOutputNeurons)
                        .lossFunction(LossFunctions.LossFunction.MSE) 
                       .build())
                .build(); 	
    	
        nn2 = new MultiLayerNetwork(conf);
        nn2.init();
	}



	private void createNetSecondLayerColorCoded(int layerCountInputNeurons) {
		Activation activation = Activation.SIGMOID;
		
		int countAdditionalNeurons = layerCountInputNeurons*4; //24
		
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed((int)(Math.random() * 1000)) //include a random seed for reproducibility
                .activation(activation)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .l2(LEARNING_RATE * 0.001)    // regularize learning model // * 0.005
                .list()
                .layer(new DenseLayer.Builder() //create the input and hidden layer
                        .nIn(layerCountInputNeurons)
                        .activation(activation)
                        .nOut(layerCountInputNeurons+countAdditionalNeurons)
                        .build())
                /*
                .layer(new DenseLayer.Builder() //create hidden layer
                        .nIn(layerCountInputNeurons+countAdditionalNeurons)
                        .activation(activation)
                        .nOut(layerCountInputNeurons+countAdditionalNeurons)
                        .build())
                        */
                .layer(new OutputLayer.Builder() //create output layer
                        .activation(activation)
                        .nOut(countOutputNeurons)
                        .lossFunction(LossFunctions.LossFunction.MSE) 
                       .build())
                .build(); 	
    	
        nn2 = new MultiLayerNetwork(conf);
        nn2.init();
	}



	private void createNetSoftMax() {
    // private void createNet() {
		Activation activation = Activation.SIGMOID;
		
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                // .seed(RNG_SEED) //include a random seed for reproducibility
                .seed((int)(Math.random() * 1000)) //include a random seed for reproducibility
                .activation(activation)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                // .l2(LEARNING_RATE * 0.001) // regularize learning model // * 0.005
                // .l2(0.0005) // ridge regression value
                .l2(LEARNING_RATE * 0.001)    // regularize learning model // * 0.005
                .list()
                .layer(new DenseLayer.Builder() //create the input and hidden layer
                        .nIn(countInputNeurons)
                        .activation(activation)
                        .nOut(countInputNeurons+countHiddenLayerNeurons)
                        .build())
                /*
                .layer(new DenseLayer.Builder() //create hidden layer 2
                        .nIn(countInputNeurons+countHiddenLayerNeurons)
                        .activation(activation)
                        .nOut(countInputNeurons+countHiddenLayerNeurons)
                        .build())
                */
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create output layer
                        .activation(Activation.SOFTMAX)
                        .nOut(countOutputNeurons)
                        .build())
                .build(); 	
    	
        nn = new MultiLayerNetwork(conf);
        nn.init();
	}


	public int getCountInputNeuronsPerGuess() {
		return countInputNeuronsPerGuess;
	}
	

	/*
	private void computeCountInputNeurons_NeuronForEveryCombination() {
		countInputColorNeuronsPerGuess  = countOutputNeurons;  
		countInputRatingNeuronsPerGuess = countDigits * (1+1); // white or black 
		
		countInputNeuronsPerGuess = countInputColorNeuronsPerGuess + countInputRatingNeuronsPerGuess;
		
		countInputNeurons = (Board.MAX_COUNT_GUESSES-1) * countInputNeuronsPerGuess;
	}
	*/

	
	private int computeCountInputNeurons(int turnNumber) {
		countInputColorNeuronsPerGuess  = countDigits * countColors;  
		countInputRatingNeuronsPerGuess = countDigits * (1+1); // white or black 
		
		return (countInputColorNeuronsPerGuess + countInputRatingNeuronsPerGuess) * turnNumber;
	}

	
	private void computeCountInputNeurons() {
		countInputColorNeuronsPerGuess  = countDigits * countColors;  
		countInputRatingNeuronsPerGuess = countDigits * (1+1); // white or black 
		
		countInputNeurons = countInputColorNeuronsPerGuess + countInputRatingNeuronsPerGuess;
		countInputNeuronsPerGuess = countInputNeurons; 
	}

	
	private void computeCountCombinations() {
		countCombinations = 1;
		for(int i=0; i<countDigits; i++) {
			countCombinations = countCombinations * countColors;
		}
	}

	
	private void computeCountOutputNeurons_NeuronForEveryCombination() {
		countOutputNeurons = 1;
		for(int i=0; i<countDigits; i++) {
			countOutputNeurons = countOutputNeurons * countColors;
		}
	}

	
	private void computeCountOutputNeurons_ForColorCodedOutput() {
		countOutputNeurons = countDigits * countColors;
	}

	
	public int getCountInputNeurons() {
		return countInputNeurons;
	}

	
	public int getCountOutputNeurons() {
		return countOutputNeurons;
	}
	
	
	private void setCodeGuessInputNeurons_NeuronForEveryCombination(INDArray inputVector, int moveNumber, Move move) {
		int startIndexMoveNumber = moveNumber * countInputNeuronsPerGuess;
		
		int index = MathSge.convertStringTo(countColors, move.guess.code);

		for(int i=0; i<countInputColorNeuronsPerGuess; i++) {
			if(index == i) inputVector.put(0, startIndexMoveNumber + i, 1);
			else inputVector.put(0, startIndexMoveNumber + i, 0);
		}
	}
	

	private void setColorInputNeurons(INDArray inputVector, int moveNumber, Move move) {
		int startIndexMoveNumber = moveNumber * countInputNeuronsPerGuess;
		
		for(int i=0; i<move.guess.code.length(); i++) {
			int colorNumber = Integer.valueOf("" + move.guess.code.charAt(i));
			inputVector.put(0, startIndexMoveNumber + i*countColors + colorNumber, 1);
		}
	}
	

	private ArrayList<Integer> code2OutputNeuronNumbers(String code) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		
		for(int i=0; i<code.length(); i++) {
			int colorNumber = Integer.valueOf("" + code.charAt(i));
			int index = i*countColors + colorNumber;
			list.add(index);
		}
		
		return list;
	}

	
	

	private void setRatingInputNeuronsWhite(INDArray inputVector, int moveNumber, Move move) {
		int startIndexWhite = (moveNumber * countInputNeuronsPerGuess) + countInputColorNeuronsPerGuess;
		
		for(int white=0; white<move.rating.countWhite; white++) {
			inputVector.put(0, startIndexWhite + white, 1);
		}
		
		for(int white=move.rating.countWhite; white<Board.getCountDigits(); white++) {
			inputVector.put(0, startIndexWhite + white, 0);
		}
	}
	
	
	private void setRatingInputNeuronsBlack(INDArray inputVector, int moveNumber, Move move) {
		int startIndexBlack = (moveNumber * countInputNeuronsPerGuess) + countInputColorNeuronsPerGuess + countDigits;
		
		for(int black=0; black<move.rating.countBlack; black++) {
			inputVector.put(0, startIndexBlack + black, 1);
		}
		
		for(int black=move.rating.countBlack; black<Board.getCountDigits(); black++) {
			inputVector.put(0, startIndexBlack + black, 0);
		}
	}
	
	
	private void setRatingInputNeurons(INDArray inputVector, int moveNumber, Move move) {
		setRatingInputNeuronsWhite(inputVector, moveNumber, move);
		setRatingInputNeuronsBlack(inputVector, moveNumber, move);
	}
	
	
	public void setInputVector(INDArray inputVector, int moveNumber, Move move) {
		if(move == null) {
			throw new RuntimeException("move is null");
		}
		
		// setCodeGuessInputNeurons_NeuronForEveryCombination(inputVector, moveNumber, move);
		// System.out.println("move: " + move.guess); 
		
		setColorInputNeurons(inputVector, moveNumber, move);
		// System.out.println("*** input vector after code: " + inputVector.toString()); 
		setRatingInputNeurons(inputVector, moveNumber, move);
		// System.out.println("*** input vector after rating: " + inputVector.toString());  
	}
	
	
	public INDArray computeInputVector(List<Move> listOfAlreadyPlayedMoves, int turnNumber) {
		// System.out.println("turnNumber: " + turnNumber); //! pd
		
		
		INDArray inputVector = Nd4j.valueArrayOf(1, computeCountInputNeurons(turnNumber), UNSET_INPUT_NEURON_VALUE);
		
		for(int i=0; i<turnNumber; i++) {
			setInputVector(inputVector, i, listOfAlreadyPlayedMoves.get(i));
		}

		// System.out.println("*** input vector: " + inputVector.toString()); 
		
		return inputVector;
	}
	
	
	public INDArray computeInputVector(Move move) {
		
		INDArray inputVector = Nd4j.valueArrayOf(1, countInputNeurons, UNSET_INPUT_NEURON_VALUE);
		
		setInputVector(inputVector, 0, move);
		
		return inputVector;
	}
	
	
	public INDArray computeInputVector(String codeToGuess, int countWhite,  int countBlack) {
		Move move = new Move();
		Guess guess = new Guess();
		guess.code = codeToGuess;
		move.guess = guess;
		
		Rating rating = new Rating();
		rating.countWhite = countWhite;
		rating.countBlack = countBlack;
		
		move.rating = rating;
		
		INDArray inputVector = computeInputVector(move);
		return inputVector;
	}
	
	
	public INDArray computeInputVector(List<Move> listOfAlreadyPlayedMoves) {
		return computeInputVector(listOfAlreadyPlayedMoves, listOfAlreadyPlayedMoves.size());
	}
	
	
	public double getSumInputVector(List<Move> listOfAlreadyPlayedMoves) {
		double sum = 0;
		INDArray inputVector = computeInputVector(listOfAlreadyPlayedMoves);
		for(int col=0; col<inputVector.columns(); col++) {
			sum = sum + inputVector.getDouble(0, col);
		}

		return sum;
	}
	
	
	public Guess getGuessAlgoExclude(List<Move> listOfAlreadyPlayedMoves, boolean verbose) {
		Guess guess = new Guess();
		guess.code = "";
		
		
		for(int n=0; n<countOutputNeurons; n++) {
			String guessCode = MathSge.convertDecTo(countColors, n, countDigits);
			if(guessCodeNotAlreadyPlayed(guessCode, listOfAlreadyPlayedMoves)) {
				if(validateGuessCodeWithAlreadyPlayedMoves(guessCode, listOfAlreadyPlayedMoves)) {
					guess.code = guessCode; 
					return guess;
				}
			}
		}
		
		return guess;
	}
	
	
	public String getWinnerOutputNeuronViaDiceRoll(boolean verbose) {
		String code = "";
		
		for(int digit=0; digit<countDigits; digit++) {
			code = code + RandomUtils.nextInt(0, countColors);
		}
    	if(verbose) System.out.println("dice roll: " + code); 
    	return code;
	}	

	
	/*
	public int getWinnerOutputNeuronViaDiceRoll(boolean verbose) {
		int r = RandomUtils.nextInt(0, countCombinations); 
    	if(verbose) System.out.println("dice roll: " + r); 
    	return r;
	}
	*/	

	
	public String getWinnertMaxValueMethod(INDArray v) {
		double maxValue = 0;
		int    max = 0;
		
		for(int c=0; c<v.columns(); c++) {
			double d = v.getDouble(0, c);
			if(d > maxValue) {
				max = c;
				maxValue = d;
			}
		}
		
		
		return MathSge.convertDecTo(countColors, max, countDigits);
	}
	
	
	public String getWinnerColorForDigitMaxValueMethod(INDArray v, int digit) {
		// System.out.println("v: " + v); 
		
		double maxValue = 0;
		int    maxColor = 0;
		
		for(int c=0; c<countColors; c++) {
			double d = v.getDouble(0, digit*countColors + c);
			if(d > maxValue) {
				maxColor = c;
				maxValue = d;
			}
		}
		
		
		return "" + maxColor;
	}
	

	public String getWinnerCodeFromOutputVectorMaxValueMethod(INDArray v) {
		double maxValue = 0;
		int    max = 0;
		
		for(int c=0; c<v.columns(); c++) {
			double d = v.getDouble(0, c);
			if(d > maxValue) {
				max = c;
				maxValue = d;
			}
		}
		
		
		return MathSge.convertDecTo(countColors, max, countDigits);
	}	

	
	public String getWinnerCodeFromOutputVectorColorCodedOutput(INDArray v) {
		String code = "";
		for(int digit=0; digit<countDigits; digit++) {
			code = code + getWinnerColorForDigit(v, digit);		
		}
		
		return code;
	}	
	

	public String getWinnerColorForDigit(INDArray v, int digit) {
		// System.out.println("v: " + v);  
		
		INDArray dVector = Nd4j.zeros(countColors);
		for(int c=0; c<countColors; c++) {
			double value = v.getDouble(0, digit*countColors + c);
			dVector.put(0,  c, value);
		}
		
		return "" + getWinnerOutputNeuronViaDiceRoll(dVector);
	}

	
	public int getWinnerOutputNeuronViaDiceRoll(INDArray v) {
    	double sum = 0;
    	for(int c=0; c<v.columns(); c++) {
    		sum = sum + v.getDouble(0, c);
    	}

    	double dice = Math.random() * sum;
    	
    	return getWinnerOutputNeuronViaDiceRoll(v, dice); 
	}	

	
	public int getWinnerOutputNeuronViaDiceRoll(INDArray v,  double dice) {
    	double border = 0;
    	for(int c=0; c<v.columns(); c++) {
    		border = border + v.getDouble(0, c);
    		if(dice < border) return c;
    	}
    	
    	return v.columns()-1;
	}	


	/*
	public int getWinnerOutputNeuronViaDiceRoll(INDArray v) {
		INDArray learningVector = Nd4j.zeros(1, countOutputNeurons);		
		
		int index = 0;
		int power = 1;
		for(int digit=countDigits-1; digit>=0; digit--) {
			INDArray colorVector = Nd4j.zeros(1, countColors);
			
			for(int c=0; c<countColors; c++) {
				double d = v.getDouble(0, digit*countColors + c);
				colorVector.put(0, c, d);
			}
			
			int wi = getWinnerOutputNeuronViaDiceRoll_NeuronForEveryCombination(colorVector);
			System.out.println("teilzahl: " + wi);
			index = index + wi*power;
			power = power * countColors;
			
			learningVector.put(0, wi + digit*countColors, 1.0);
		}
		
		return index;
	}
	*/	

	
	public INDArray feedForward(int moveNumber, INDArray inputVector) {
		if(moveNumber == 1 ) return nn.feedForward(inputVector, false).get(nn.getnLayers());
		if(moveNumber == 2 ) return nn2.feedForward(inputVector, false).get(nn2.getnLayers());
		
		return null;
	}

	
	public Guess getFirstMove(boolean verbose) {
		String code = "";

		if(firstMoveFixed) code = firstMoveFixedCode;
		else code = getWinnerOutputNeuronViaDiceRoll(verbose);
	
		if(verbose) System.out.println("guess color combination: " + code);
		
		Guess guess = new Guess();
		guess.code  = code;
		guess.inputVector = null; 
		guess.outputVector = null; 
		
		return guess; 
	}

	
	/*
	public Guess getFirstMove(int winnerNeuronIndex, boolean verbose) {
		if(verbose) System.out.println("winnerNeuronIndex: " + winnerNeuronIndex);
		
		String guessCode = MathSge.convertDecTo(countColors, winnerNeuronIndex, countDigits);
		if(verbose) System.out.println("guess color combination: " + guessCode);
		
		Guess guess = new Guess();
		guess.code  = guessCode;
		guess.index = winnerNeuronIndex;
		guess.inputVector = null;  // Nd4j.zeros(1, this.countInputNeurons);
		guess.outputVector = null; //Nd4j.zeros(1, this.countOutputNeurons);
		
		return guess; 
	}
	*/
	
	
	public Guess getGuessAlgoQ(List<Move> listOfAlreadyPlayedMoves, int moveNumber, double explorationRate, boolean verbose) {
		if(explorationRate > Math.random()) {
			return getGuessRandom(listOfAlreadyPlayedMoves, moveNumber, verbose);
		}

		INDArray inputVector  = computeInputVector(listOfAlreadyPlayedMoves, moveNumber);
		INDArray resultVector = feedForward(moveNumber, inputVector);		
		
		String guessCode = ""; 
		if(listOfAlreadyPlayedMoves.isEmpty()) {
			if(firstMoveFixed) guessCode = firstMoveFixedCode;
			else guessCode = getWinnerOutputNeuronViaDiceRoll(verbose);
		}
		else {
			int winnerNeuronIndex = getWinnerNeuronNumberRLTraining(resultVector);
			guessCode = MathSge.convertDecTo(countColors, winnerNeuronIndex, countDigits);

		}
		
		if(verbose) System.out.println("input vector:\n" + inputVector.toString());
		if(verbose) System.out.println("output vector:\n" + resultVector.toString());
		
		if(verbose) System.out.println("guess color combination: " + guessCode);
		
		Guess guess = new Guess();
		guess.code  = guessCode;
		guess.inputVector = inputVector;
		guess.outputVector = resultVector;
		
		return guess; 
	}
	
	
	public Guess getGuessAlgoNN(List<Move> listOfAlreadyPlayedMoves, int moveNumber, boolean verbose) {
		INDArray inputVector = computeInputVector(listOfAlreadyPlayedMoves);

        INDArray resultVector = feedForward(moveNumber, inputVector);		

        if(verbose) System.out.println("input vector:\n" + inputVector.toString());
		if(verbose) System.out.println("output vector:\n" + resultVector.toString());
		
		int winnerNeuronIndex = getWinnerNeuronNumber(resultVector, listOfAlreadyPlayedMoves);
		if(verbose) System.out.println("winnerNeuronIndex: " + winnerNeuronIndex);
		
		String guessCode = MathSge.convertDecTo(countColors, winnerNeuronIndex, countDigits);
		if(verbose) System.out.println("guess color combination: " + guessCode);
		
		Guess guess = new Guess();
		guess.code  = guessCode;
		// guess.index = winnerNeuronIndex;
		
		return guess; 
	}
	
	
	public Guess getGuessAlgoRLPlaying(List<Move> listOfAlreadyPlayedMoves, int moveNumber, boolean verbose) {
		INDArray inputVector = computeInputVector(listOfAlreadyPlayedMoves);
        INDArray resultVector = feedForward(moveNumber, inputVector);		
        
		if(verbose) System.out.println("input vector:\n" + inputVector.toString());
		if(verbose) System.out.println("output vector:\n" + resultVector.toString());
		
		int winnerNeuronIndex = getWinnerNeuronNumber(resultVector, listOfAlreadyPlayedMoves);
		if(verbose) System.out.println("winnerNeuronIndex: " + winnerNeuronIndex);

		String guessCode = MathSge.convertDecTo(countColors, winnerNeuronIndex, countDigits);
		if(verbose) System.out.println("guess color combination: " + guessCode);
		
		Guess guess = new Guess();
		guess.code = guessCode; 
		// guess.index = winnerNeuronIndex;
		
		return guess; 
	}
	
	
	public Guess getGuessAlgoRLTraining(List<Move> listOfAlreadyPlayedMoves, int moveNumber, boolean verbose) {
		INDArray inputVector = computeInputVector(listOfAlreadyPlayedMoves);
        INDArray resultVector = feedForward(moveNumber, inputVector);		
		if(verbose) System.out.println("input vector:\n" + inputVector.toString());
		if(verbose) System.out.println("output vector:\n" + resultVector.toString());
		
		int winnerNeuronIndex = getWinnerNeuronNumberRLTraining(resultVector);
		
		if(verbose) System.out.println("winnerNeuronIndex: " + winnerNeuronIndex);
		String guessCode = MathSge.convertDecTo(countColors, winnerNeuronIndex, countDigits);
		if(verbose) System.out.println("guess color combination: " + guessCode);
		
		Guess guess = new Guess();
		guess.code         = guessCode;
		guess.inputVector  = inputVector;
		guess.outputVector = resultVector;
		// guess.index        = winnerNeuronIndex;
		
		return guess; 
	}
	
	
	private boolean validateGuessCodeWithAlreadyPlayedMoves(String guessCode, List<Move> listOfAlreadyPlayedMoves) {
		for(int turn=0; turn<listOfAlreadyPlayedMoves.size(); turn++) {
			Move move = listOfAlreadyPlayedMoves.get(turn);
			Rating rating = Board.getRating(move.guess.code, guessCode);
			if(rating.countBlack != move.rating.countBlack) return false;
			if(rating.countWhite != move.rating.countWhite) return false;
		}

		return true;
	}

	
	boolean guessCodeNotAlreadyPlayed(String guessCode, List<Move> listOfAlreadyPlayedMoves) {
		//! performance improvement (use hashSet)
		for(Move move : listOfAlreadyPlayedMoves) {
			if(guessCode.contentEquals(move.guess.code)) return false;
		}		
			
		return true;	
	}
	

	/*
	boolean guessCodeNotAlreadyPlayed(int index, List<Move> listOfAlreadyPlayedMoves) {
		//! performance improvement (use hashSet)
		for(Move move : listOfAlreadyPlayedMoves) {
			if(index == move.guess.index) return false;
		}		
			
		return true;	
	}
	*/
	

	public int getWinnerNeuronNumber(INDArray outputNeurons, List<Move> listOfAlreadyPlayedMoves) {
		// performance
		int winnerIndex = 0;
		double winnerValue = -2;
		for(int i=0; i<countOutputNeurons; i++) {
			double value = outputNeurons.getDouble(0, i);
			if(value > winnerValue) {
				String guessCode = MathSge.convertDecTo(countColors, i, countDigits);
				if(guessCodeNotAlreadyPlayed(guessCode, listOfAlreadyPlayedMoves)) {
					winnerIndex = i;
					winnerValue = value;
				}
			}
		}
		
		return winnerIndex;
	}

	
	public int getWinnerNeuronNumber(INDArray outputNeurons) {
		// performance
		int winnerIndex = 0;
		double winnerValue = -2;
		for(int i=0; i<countOutputNeurons; i++) {
			double value = outputNeurons.getDouble(0, i);
			if(value > winnerValue) {
				winnerIndex = i;
				winnerValue = value;
			}
		}
		
		return winnerIndex;
	}

	
	public double getSumOfAllOutputNeurons(INDArray outputNeurons) {
		double sum = 0;
		for(int i=0; i<countOutputNeurons; i++) {
			double value = outputNeurons.getDouble(0, i);
			sum = sum + value;
		}
		
		return sum;
	}
	
	
	public int getWinnerIndexRL(INDArray outputNeurons, double randomValue) {
		double sum = 0;
		for(int i=0; i<countOutputNeurons; i++) {
			double value = outputNeurons.getDouble(0, i);
			sum = sum + value;
			
			if(sum>randomValue) return i;
		}
		
		return countOutputNeurons-1;
	}
	
	
	public int getWinnerNeuronNumberRLTraining(INDArray output) {
		INDArray outputNeurons = Nd4j.create(output.shapeDescriptor());
		
		double sumOfAllOutputNeurons = getSumOfAllOutputNeurons(outputNeurons);
		// System.out.println("sumOfAllOutputNeurons: " + sumOfAllOutputNeurons);
		
		Random rand = MathSge.getSgeRandom();
		double randomValue = rand.nextDouble() * sumOfAllOutputNeurons;
		// System.out.println("nextDouble: " + randomValue);  
		
		int winnerIndex = getWinnerIndexRL(outputNeurons, randomValue);
		// System.out.println("winnerIndex: " + winnerIndex);

		return winnerIndex;
	}
	
	
	public int ratingCompare(Rating rating1, Rating rating2) {
		// System.out.println("currentRatingSum: " + (currentRating.countWhite + currentRating.countBlack) + " lastRatingSum: " + (lastRating.countWhite + lastRating.countBlack));
		
		int ratingScore1 = rating1.countWhite + rating1.countBlack;
		int ratingScore2 = rating2.countWhite + rating2.countBlack;
		
		if(ratingScore1 == ratingScore2) return  0;
		if(ratingScore1 >  ratingScore2) return  1;
		if(ratingScore1 <  ratingScore2) return -1;
		
		return 0;
	}
	

	public LearnStatisticOneTraining learnNN(Board board, boolean verbose) {
		LearnStatisticOneTraining learnResultOneTraining = new LearnStatisticOneTraining();
		
		if(board.getTurnNumber() <= 1) return learnResultOneTraining;
		
		if(verbose) System.out.println(">>>> learn");
		

		if(-1 == ratingCompare(board.getLastMove().rating, board.getSecondLastMove().rating)) {
			learnResultOneTraining.counterTrainBadMove++;
			INDArray inputVector = computeInputVector(board.getListOfMoves(), board.getTurnNumber()-1);
			if(verbose) System.out.println("input vector for learning:\n" + inputVector.toString());

	        INDArray outputVector = feedForward(board.getTurnNumber(), inputVector);		
			if(verbose) System.out.println("output vector:\n" + outputVector.toString());
			
			if(verbose) System.out.println("last guess: " + board.getLastMove().guess); // '33' winnerNeuron 15
			if(verbose) System.out.println("last winner neuron: " + MathSge.convertStringTo(countColors, board.getLastMove().guess.code));
			int winnerNeuron = MathSge.convertStringTo(countColors, board.getLastMove().guess.code);
			
			for(int i=0; i<outputVector.columns(); i++) outputVector.put(0, i, 1);
			outputVector.put(0, winnerNeuron, 0);
			if(verbose) System.out.println("output vector using for learning:\n" + outputVector.toString());
			
			int winnerNeuronIndexBeforeTraining = winnerNeuron;

	        nn.fit(inputVector, outputVector);		

	        INDArray resultAfterTraining = feedForward(board.getTurnNumber(), inputVector);		
			if(verbose) System.out.println("output vector during training: \n" + resultAfterTraining.toString());
			
			int winnerNeuronIndexAfterTraining = getWinnerNeuronNumber(resultAfterTraining, board.getListOfMoves());
			
			if(winnerNeuronIndexBeforeTraining != winnerNeuronIndexAfterTraining) learnResultOneTraining.successfulTrainedFromBadToGoodMove++;
			
			if(verbose) System.out.println("output vector used for learning:\n" + outputVector.toString());
			if(verbose) {
				System.out.println("output vector after training:\n" + resultAfterTraining.toString());				
			}
		}
		if(verbose) System.out.println("<<<< learn");

		return learnResultOneTraining;
	}

	
	public LearnStatisticOneTraining learnRL(Board board, boolean verbose) {
		//		LearnStatisticOneTraining learnResultOneTraining = new LearnStatisticOneTraining();
		//		
		//		if(verbose) System.out.println(">>>> learn");
		//		if(verbose) System.out.println("<<<< learn");
		//
		//		return learnResultOneTraining;
		return null;
	}


	public double calculateReward(boolean win, int maxMoves, int countMoves ) {
		if(win) {
			return 1 - (1.0*countMoves) / (1.0*maxMoves);
		}
		else {
			return -1.0;
		}
	}


	public void trainAlgoQ_save(Board board) {
		// pseudo code (python)
		// 
		// unitVector = np.zeros(number_of_actions)
		// unitVector[selectedMove] = 1
		// action_prob_grads = unitVector - outputVector
	    // Y = outputVector + lambda * learning_rate * rewards * action_prob_grads // lambda = far decisions not important and recent decisions important
        // model.train_on_batch(X, Y)		

		double reward = calculateReward(board.codeFound(), Board.MAX_COUNT_GUESSES, board.getListOfMoves().size());
		
		
		for(int move = 0; move<board.getListOfMoves().size(); move++) {
        	double lambda = (1.0*(move+1))/(1.0*board.getListOfMoves().size());
        	double learning_rate = 1.0;
			
			Guess guess = board.getListOfMoves().get(move).guess;

			// System.out.println(guess.outputVector.toString());
			
			// unitVector = np.zeros(number_of_actions)
			INDArray unitVector = Nd4j.zeros(1, guess.outputVector.columns());

			// unitVector[selectedMove] = 1
			for(int d=0; d<countDigits; d++) {
				int c = (int)guess.code.charAt(d);
				unitVector.put(0, c + d*countColors, 1);
			}
			
			// action_prob_grads = unitVector - outputVector
			INDArray action_prob_grads = unitVector.sub(guess.outputVector);
			
		    // Y = outputVector + lambda * learning_rate * rewards * action_prob_grads // lambda = far decisions not important and recent decisions important
		
			INDArray y = guess.outputVector.add(action_prob_grads.mul(lambda * learning_rate * reward));
			// System.out.println(y.toString());

	        nn.fit(guess.inputVector, y);		
		}		
	}


	
	public final double epsilon = 0.001;
	
	public void trainAlgoQ_QFormular(Board board, boolean verbose) {
	// public void trainAlgoQ(Board board, boolean verbose) {
		// pseudo code (python)
		// 
		// e = np.zeros(number_of_actions)
		// e[selectedMove] = 1
		// Y = action_prob_output + lambda * learning_rate * rewards * (e - action_prob_output) 
		// model.train_on_batch(X, Y)
		
		// double gameReward = 0;
		// if(board.codeFound()) gameReward = 0;
		// else gameReward = -1;		
		
    	// if(!board.codeFound()) return;
    	// if(board.getListOfMoves().size() != (Board.MAX_COUNT_GUESSES-1)) return;
		
		if(verbose) System.out.println("------------------------------------------------------------------------------------");
		if(verbose) System.out.println("trainAlgoQ");
        
        int numberOfDecisions = board.getListOfMoves().size();
        if(verbose) {
        	System.out.println();
        	System.out.println("Results before training:");
	        for(int i=1; i<numberOfDecisions; i++) {
	        	Guess guess = board.getListOfMoves().get(i).guess;
	        	
	        	INDArray afterTrainingVector = feedForward(board.getTurnNumber(), guess.inputVector);	
	        	System.out.println("before training output: " + afterTrainingVector.toString());
	        }
        }

        for(int i=1; i<numberOfDecisions; i++) { // no training for the first guess
        	if(verbose) System.out.println("i: " + i);
        	
        	double score = board.getListOfMoves().get(i).rating.score;
        	if(verbose) System.out.println("score: " + score);
        	
        	Guess guess = board.getListOfMoves().get(i).guess;

            INDArray outputINDArray = Nd4j.zeros(1, this.countOutputNeurons);
			for(int c=0; c<outputINDArray.columns(); c++) {
				outputINDArray.put(0, c, guess.outputVector.getDouble(0, c));
				if(guess.outputVector.getDouble(0, c) < epsilon)     outputINDArray.put(0, c, epsilon);
				if(guess.outputVector.getDouble(0, c) > (1-epsilon)) outputINDArray.put(0, c, 1-epsilon);
			}
            
            if(verbose) System.out.println("outputVector:           " + outputINDArray);
        	
        	
			// unitVector = np.zeros(number_of_actions)
			INDArray unitVector = Nd4j.zeros(1, outputINDArray.columns());
			
			// unitVector[selectedMove] = 1
			for(int d=0; d<countDigits; d++) {
				int c = (int)guess.code.charAt(d);
				unitVector.put(0, c + d*countColors, 1);
			}
			if(verbose) System.out.println("unitVector:             " + unitVector.toString());
			
			// action_prob_grads = unitVector - outputVector
			INDArray action_prob_grads = unitVector.sub(outputINDArray);
			if(verbose) System.out.println("action_prob_grads:      " + action_prob_grads.toString());
			
		
			double learningRate = 0.1; 
			
			double reward = 0.0;
	    	if(board.codeFound()) {
	    		reward = 1.0;
	    	}
	    	else { 
	    		reward = -1.0;
	    	}

			// Y = outputVector + lambda * learningRated * rewards * action_prob_grads 
			INDArray y = outputINDArray.add(action_prob_grads.mul(learningRate * reward));
			
			if(verbose) System.out.println("learning vector:        " + y.toString());
        	if(board.codeFound()) {
        		nn.fit(guess.inputVector, y);		
        	}
        }
        
        if(verbose) {
	        System.out.println();
	        System.out.println("Results after training:");
	        
	        for(int i=1; i<numberOfDecisions; i++) {
	        	Guess guess = board.getListOfMoves().get(i).guess;
	        	
	        	INDArray afterTrainingVector = feedForward(board.getTurnNumber(), guess.inputVector);	
	        	System.out.println("after training output:  " + afterTrainingVector.toString());
	        }
        }
	}

	
	public void trainAlgoQ(Board board, int turnNumber, boolean verbose) {
	// public void trainAlgoQ_UnitVector(Board board, boolean verbose) {
		
		int moveNumber = turnNumber - 1;
		if(board.getListOfMoves().isEmpty()) return;
		if(board.getListOfMoves().size() < 2) return;
		
    	if(!board.codeFound()) return;
		
		if(verbose) System.out.println("------------------------------------------------------------------------------------");
		if(verbose) System.out.println("trainAlgoQ");
		if(verbose) System.out.println("moveNumberToTrain: " + moveNumber);
    	
    	double score = board.getListOfMoves().get(moveNumber).rating.score;
    	if(verbose) System.out.println("score: " + score);
    	
    	Guess guess = board.getListOfMoves().get(moveNumber).guess;

        if(verbose) System.out.println("outputVector:           " + guess.outputVector);
    	
		INDArray unitVector = Nd4j.zeros(1, countOutputNeurons);
		
		// System.out.println("guess.code: " + guess.code);
		
		// unitVector[selectedMove] = 1
		for(int d=0; d<countDigits; d++) {
			int c = (int)(guess.code.charAt(d) - '0');
			// System.out.println("unitVector c: " + c + "  d: " + d + " countColors: " + countColors + "  sum: " + c + d*countColors); 
			unitVector.put(0, c + d*countColors, 1);
		}
		
	
		double learningRate = 1.0; 
    	INDArray op = Nd4j.zeros(1, countOutputNeurons);
    	
    	if(board.codeFound()) {
    		op = distributePositive(guess.outputVector, guess.code, learningRate);
			if(verbose) System.out.println("input    vector:        " + guess.inputVector);
			if(verbose) System.out.println("learning vector:        " + op);

			if(moveNumber == 1) { 
				nn.fit(guess.inputVector, op);	
        	}
			
        	if(moveNumber == 2) { 
    			nn2.fit(guess.inputVector, op);		

    			// if(board.getCodeToFind().contentEquals("22")) { 
        		// double[] da = {  1.0000,         0,         0,         0,         0,    1.0000,         0,         0,         0,         0,         0,         0,         0,         0,         0,    1.0000,         0,         0,         0,    1.0000,         0,         0,         0,         0 };
        		// INDArray inputVector = Nd4j.create(da);
        		// if(MathSge.compareINDArrayDim1xNwithDimN(guess.inputVector, inputVector, 0.01)) {
        		
	    			// System.out.println("input    vector:        " + guess.inputVector);
	    			// System.out.println("learning vector:        " + op);
	    			// System.out.println();
	    			
	    			//if(board.getCodeToFind().contentEquals("22") || board.getCodeToFind().contentEquals("23") || board.getCodeToFind().contentEquals("32") || board.getCodeToFind().contentEquals("33")) {
		    			// System.out.println("codeToFind: " + board.getCodeToFind());
		    			// nn2.fit(guess.inputVector, op);		
	    			//}
	    			//else {
		    		//	System.out.println("codeToFind: " + board.getCodeToFind());
	    			//}
        		//}
        		//else {
	    			// if(board.getCodeToFind().contentEquals("22") || board.getCodeToFind().contentEquals("23") || board.getCodeToFind().contentEquals("32") || board.getCodeToFind().contentEquals("33")) {
		    		/*	
		    		if(board.getCodeToFind().contentEquals("23") ) {
		    			System.out.println("input    vector:        " + guess.inputVector);
		    			System.out.println("learning vector:        " + op);

		        		double[] da_b = {  1.0000,         0,         0,         0,         0,    1.0000,         0,         0,         0,         0,         0,         0,         0,         0,         0,    1.0000,         0,         0,         0,    1.0000,         0,         0,         0,         0 };
		        		INDArray inputVector_b = Nd4j.create(da_b);
		        		//if(MathSge.compareINDArrayDim1xNwithDimN(guess.inputVector, inputVector_b, 0.01)) {
		        			nn2.fit(guess.inputVector, op);		
		        		//}
		        		//else {
			    		//	System.out.println();
		        		//}
		    		}
        		}
				*/
        	}
    	}
    	else {
			if(verbose) System.out.println("learning vector:        " + "no training");

			// Method 1
			// op = distributePositive(guess.outputVector, guess.code, learningRate*score*0.5);
			// if(verbose) System.out.println("learning vector:        " + op);
        	// nn.fit(ip, op);		

        	// Method 2
        	// op = distributeNegative(guess.outputVector, guess.index, learningRate);
			// if(verbose) System.out.println("learning vector:        " + op);
        	// nn.fit(ip, op);		
    	}
        
        if(verbose) {
	        System.out.println();
	        System.out.println("Results after training:");
	        
        	INDArray afterTrainingVector = feedForward(board.getTurnNumber()-1, guess.inputVector);	
        	System.out.println("after training output:  " + afterTrainingVector.toString());
        }
	}

	
	boolean guessIs(INDArray inputVector, int white, int black) {
		
		double countWhite = inputVector.getDouble(0, countColors * countDigits + 0) + inputVector.getDouble(0, countColors * countDigits + 1);
		double countBlack = inputVector.getDouble(0, countColors * countDigits + 2) + inputVector.getDouble(0, countColors * countDigits + 3);
		
		if(Math.abs(countBlack - black) > 0.001) return false;
		if(Math.abs(countWhite - white) > 0.001) return false;
		
		return true;
	}
	
	
	public INDArray distributePositive(INDArray oldOP, String code, double learningRate) {
		INDArray newOP = Nd4j.zeros(1, oldOP.columns());
		
		int index = MathSge.convertStringTo(countColors, code);
        newOP.put(0, index, 1);
		
		INDArray lrOP = Nd4j.zeros(1, oldOP.columns());
		for(int i=0; i<oldOP.columns(); i++) {
			double lV = (1-learningRate) * oldOP.getDouble(0, i) + learningRate * newOP.getDouble(0, i);
			lrOP.put(0, i, lV); 
		}
		
		
		return lrOP;
	}
	

	public INDArray distributePositive_ForColorCodedOutput(INDArray oldOP, String code, double learningRate) {
		INDArray newOP = Nd4j.zeros(1, oldOP.columns());
		
		for(int d=0; d<countDigits; d++) {
			int c = (int)(code.charAt(d) - '0');
			newOP.put(0, c + d*countColors, 1);
		}
		
		INDArray lrOP = Nd4j.zeros(1, oldOP.columns());
		for(int i=0; i<oldOP.columns(); i++) {
			double lV = (1-learningRate) * oldOP.getDouble(0, i) + learningRate * newOP.getDouble(0, i);
			lrOP.put(0, i, lV); 
		}
		
		
		return lrOP;
	}
	

	public INDArray distributeNegative(INDArray oldOP, int index, double learningRate) {
		
		INDArray newOP = oldOP.dup();

		double value = oldOP.getDouble(0, index);
		int vectorLength = oldOP.columns();
		
		for(int i=0; i<vectorLength; i++) {
			double oldV = oldOP.getDouble(0, i);
			double newV = oldV + value / (vectorLength - 1);
			newOP.put(0, i, newV); 
		}
		newOP.put(0, index, 0.0); 

		
		INDArray lrOP = Nd4j.zeros(1, oldOP.columns());
		for(int i=0; i<oldOP.columns(); i++) {
			double lV = (1-learningRate) * oldOP.getDouble(0, i) + learningRate * newOP.getDouble(0, i);
			lrOP.put(0, i, lV); 
		}
		
		
		return lrOP;
	}

	
	public Guess getGuessRandom(List<Move> listOfAlreadyPlayedMoves, int moveNumber, boolean verbose) {
		INDArray inputVector = computeInputVector(listOfAlreadyPlayedMoves);

        INDArray resultVector = feedForward(moveNumber, inputVector);		
		
		if(verbose) System.out.println("input vector:\n" + inputVector.toString());
		if(verbose) System.out.println("output vector:\n" + resultVector.toString());

		String guessCode = getWinnerOutputNeuronViaDiceRoll(verbose);
		if(verbose) System.out.println("guess color combination: " + guessCode);
		
		Guess guess = new Guess();
		guess.code  = guessCode;
		guess.inputVector = inputVector;
		guess.outputVector = resultVector;
		
		return guess; 		
	}
	

	public void setFirstMoveFix(String code) {
		firstMoveFixed = true;
		firstMoveFixedCode  = code;
		
		if(code.length() != countDigits) setFirstMoveFixNatural();
	}
	
	
	public void setFirstMoveFixNatural() {
		String s = "";
		for(int i=0; i<countDigits; i++) {
			s = s + (i % countColors);
		}
		
		// System.out.println("fix Move: " + MathSge.convertStringTo(countColors, s) + " " + s); //!
		firstMoveFixed = true;
		firstMoveFixedCode  = s;
	}
	
	
	public void unsetFirstMove() {
		firstMoveFixed = false;
	}

	public int getCountColors() {
		return countColors;
	}
	
	
	public int getCountDigits() {
		return countDigits;
	}

}
