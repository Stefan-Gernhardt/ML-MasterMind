package org.neuralnetwork.org.sge.mm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.sge.mm.Board;
import org.sge.mm.ComputationUnit;
import org.sge.mm.Guess;
import org.sge.mm.MasterMind;
import org.sge.mm.Move;
import org.sge.mm.Rating;
import org.sge.mm.ResultOneGame;

public class ComputationUnitTest {
	@Test
	public void test01_SetInputNeuronenNoMoves() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE, 0, 0);
		ComputationUnit computationUnit = masterMind.getComputationUnit();
		
		List<Move> listOfAlreadyPlayedMoves = new ArrayList<Move>();
		
		INDArray inputVector = computationUnit.computeInputVector(listOfAlreadyPlayedMoves);
		
		for(int column=0; column<computationUnit.getCountInputNeurons(); column++) {
			double nv = inputVector.getDouble(0, column);
			assertEquals(ComputationUnit.UNSET_INPUT_NEURON_VALUE, nv, 0.00001);
		}
	}


	@Test
	public void test02_SetInputNeuronenOneMove() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE, 0, 0);
		ComputationUnit computationUnit = masterMind.getComputationUnit();
		
		List<Move> listOfAlreadyPlayedMoves = new ArrayList<Move>();
		Move move = new Move();
		
		Guess guess = new Guess();
		guess.code = "02";
		move.guess = guess;
		
		Rating rating = new Rating();
		rating.countBlack = 0;
		rating.countWhite = 2;
		move.rating = rating;
		
		listOfAlreadyPlayedMoves.add(move);
		
		INDArray inputVector = computationUnit.computeInputVector(listOfAlreadyPlayedMoves);

		// code
    	assertEquals(0, inputVector.getDouble(0, 0), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 1), 0.00001);
		assertEquals(1, inputVector.getDouble(0, 2), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 3), 0.00001);
		
		assertEquals(0, inputVector.getDouble(0, 4), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 5), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 6), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 7), 0.00001);
		
		assertEquals(0, inputVector.getDouble(0, 8), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 9), 0.00001);
		assertEquals(0, inputVector.getDouble(0,10), 0.00001);
		assertEquals(0, inputVector.getDouble(0,11), 0.00001);
		
		assertEquals(0, inputVector.getDouble(0,12), 0.00001);
		assertEquals(0, inputVector.getDouble(0,13), 0.00001);
		assertEquals(0, inputVector.getDouble(0,14), 0.00001);
		assertEquals(0, inputVector.getDouble(0,15), 0.00001);
		
		// ratings
		assertEquals(1, inputVector.getDouble(0,16), 0.00001);
		assertEquals(1, inputVector.getDouble(0,17), 0.00001);
		assertEquals(0, inputVector.getDouble(0,18), 0.00001);
		assertEquals(0, inputVector.getDouble(0,19), 0.00001);
		

		// remainder UNSET_INPUT_NEURON_VALUE
		for(int column=computationUnit.getCountInputNeuronsPerGuess(); column<computationUnit.getCountInputNeurons(); column++) {
			double nv = inputVector.getDouble(0, column);
			assertEquals(ComputationUnit.UNSET_INPUT_NEURON_VALUE, nv, 0.00001);
		}

	}


	@Test
	public void test03_SetInputNeuronenTwoMoves() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE, 0, 0);
		ComputationUnit computationUnit = masterMind.getComputationUnit();
		
		List<Move> listOfAlreadyPlayedMoves = new ArrayList<Move>();
		
		// move 1
		Move move1 = new Move();
		
		Guess guess1 = new Guess();
		guess1.code = "02";
		move1.guess = guess1;
		
		Rating rating1 = new Rating();
		rating1.countBlack = 0;
		rating1.countWhite = 2;
		move1.rating = rating1;		
		
		listOfAlreadyPlayedMoves.add(move1);
		
		// move 2
		Move move2 = new Move();
		
		Guess guess2 = new Guess();
		guess2.code = "20";
		move2.guess = guess2;
		
		Rating rating2 = new Rating();
		rating2.countBlack = 2;
		rating2.countWhite = 0;
		move2.rating = rating2;		
		
		move2.rating = rating2;
		listOfAlreadyPlayedMoves.add(move2);

		// compute input vector
		INDArray inputVector = computationUnit.computeInputVector(listOfAlreadyPlayedMoves);

		// code
		assertEquals(0, inputVector.getDouble(0, 0), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 1), 0.00001);
		assertEquals(1, inputVector.getDouble(0, 2), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 3), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 4), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 5), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 6), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 7), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 8), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 9), 0.00001);
		assertEquals(0, inputVector.getDouble(0,10), 0.00001);
		assertEquals(0, inputVector.getDouble(0,11), 0.00001);
		assertEquals(0, inputVector.getDouble(0,12), 0.00001);
		assertEquals(0, inputVector.getDouble(0,13), 0.00001);
		assertEquals(0, inputVector.getDouble(0,14), 0.00001);
		assertEquals(0, inputVector.getDouble(0,15), 0.00001);

		
		// ratings
		assertEquals(1, inputVector.getDouble(0,16), 0.00001);
		assertEquals(1, inputVector.getDouble(0,17), 0.00001);
		assertEquals(0, inputVector.getDouble(0,18), 0.00001);
		assertEquals(0, inputVector.getDouble(0,19), 0.00001);
		
		// code
		int offset = computationUnit.getCountInputNeuronsPerGuess();
		System.out.println("offset: " + offset);
		assertEquals(0, inputVector.getDouble(0, 0 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 1 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 2 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 3 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 4 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 5 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 6 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 7 + offset), 0.00001);
		assertEquals(1, inputVector.getDouble(0, 8 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0, 9 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0,10 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0,11 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0,12 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0,13 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0,14 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0,15 + offset), 0.00001);

		// ratings
		assertEquals(0, inputVector.getDouble(0,16 + offset), 0.00001);
		assertEquals(0, inputVector.getDouble(0,17 + offset), 0.00001);
		assertEquals(1, inputVector.getDouble(0,18 + offset), 0.00001);
		assertEquals(1, inputVector.getDouble(0,19 + offset), 0.00001);

		
		for(int column=0+computationUnit.getCountInputNeuronsPerGuess()*2; column<computationUnit.getCountInputNeurons(); column++) {
			double nv = inputVector.getDouble(0, column);
			assertEquals(ComputationUnit.UNSET_INPUT_NEURON_VALUE, nv, 0.00001);
		}
	}
	
	
	int countZerosOfInputVector(INDArray inputVector) {
		int numberOfZeros = 0;
		
		for(int c=0; c<inputVector.columns(); c++) {
			double v = inputVector.getDouble(0, c);
			if(Math.abs(v) < 0.001) numberOfZeros++;
		}
		
		return numberOfZeros;
	}
	

	int count_UNSET_INPUT_NEURON_VALUEOfInputVector(INDArray inputVector) {
		int numberOfZeros = 0;
		
		for(int c=0; c<inputVector.columns(); c++) {
			double v = inputVector.getDouble(0, c);
			if(Math.abs(v-ComputationUnit.UNSET_INPUT_NEURON_VALUE) < 0.001) numberOfZeros++;
		}
		
		return numberOfZeros;
	}
	

	@Test
	public void test04_InputVector() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0); 
		Board board = masterMind.getBoard();
		ComputationUnit computationUnit = masterMind.getComputationUnit();
		String codeToFind = "13";
		board.setCodeToFind(codeToFind);
		
		// move 0
		System.out.println("-------------------------------------------------------------");
		
		INDArray inputVector0 = computationUnit.computeInputVector(board.getListOfMoves());		
		System.out.println("input vector0: " + inputVector0.toString());
		System.out.println("number UNSET_INPUT_NEURON_VALUE of input vector: " + count_UNSET_INPUT_NEURON_VALUEOfInputVector(inputVector0));
		assertEquals(count_UNSET_INPUT_NEURON_VALUEOfInputVector(inputVector0), 12);
		
		Guess guess0 = new Guess();
		guess0.code = "31";
		
		Move move0 = board.setGuessOnBoard(guess0, true);
		System.out.println("code to find: " + codeToFind + "  guess: " + guess0.code + "  rating black: " + move0.rating.countBlack + "  white: " + move0.rating.countWhite);

		
		// move 1
		System.out.println("-------------------------------------------------------------");
		
		INDArray inputVector1 = computationUnit.computeInputVector(board.getListOfMoves());		
		System.out.println("input vector1: " + inputVector1.toString());
		System.out.println("number UNSET_INPUT_NEURON_VALUE of input vector: " + count_UNSET_INPUT_NEURON_VALUEOfInputVector(inputVector1));
		assertEquals(count_UNSET_INPUT_NEURON_VALUEOfInputVector(inputVector1), 8);

		assertTrue(Math.abs(inputVector1.getDouble(0, 0) - 0.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0, 1) - 0.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0, 2) - 0.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0, 3) - 1.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0, 4) - 0.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0, 5) - 1.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0, 6) - 0.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0, 7) - 0.0) < 0.0001);

		assertTrue(Math.abs(inputVector1.getDouble(0,  8) - 1.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0,  9) - 1.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0, 10) - 0.0) < 0.0001);
		assertTrue(Math.abs(inputVector1.getDouble(0, 11) - 0.0) < 0.0001);
		
		
		//! hier geht es weiter
		/*
		Guess guess1 = masterMind.getGuess(ComputationUnit.ALGO_NN, 0, true, true);
		Move move1 = board.setGuessOnBoard(guess1, true);
		System.out.println("code to find: " + codeToFind + "  guess: " + guess1.code + "  rating black: " + move1.rating.countBlack + "  white: " + move1.rating.countWhite);

		
		// move 2
		System.out.println("-------------------------------------------------------------");
		
		INDArray inputVector2 = computationUnit.computeInputVector(board.getListOfMoves());		
		System.out.println("input vector2: " + inputVector2.toString());
		System.out.println("number UNSET_INPUT_NEURON_VALUE of input vector: " + count_UNSET_INPUT_NEURON_VALUEOfInputVector(inputVector2));
		assertEquals(count_UNSET_INPUT_NEURON_VALUEOfInputVector(inputVector2), 180);
		*/
	}

	
	@Test
	public void test10_GetWinner() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE, 0, 0);
		ComputationUnit computationUnit = masterMind.getComputationUnit();

		INDArray outputVector = Nd4j.create(1, 4); 
		outputVector.put(0, 0, 0.9); // 0.0 - 0.9  1.0
		outputVector.put(0, 1, 0.9); // 0.9 - 1.8  1.5
		outputVector.put(0, 2, 0.2); // 1.8 - 2.0  1.9
		outputVector.put(0, 3, 0.4); // 2.0 - 2.4  2.2
		
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, -4.0), 0);
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 0.0), 0);
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 0.5), 0); //
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 0.9), 1);
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 1.0), 1);
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 1.5), 1); //
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 1.8), 2);
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 1.9), 2); //
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 2.0), 3);
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 2.2), 3); // 
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 2.4), 3);
		assertEquals(computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector, 3.0), 3);
		
	}	

	
	@Test
	public void test11_GetWinner() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE, 0, 0);
		ComputationUnit computationUnit = masterMind.getComputationUnit();

		INDArray outputVector = Nd4j.create(1, 4); 
		outputVector.put(0, 0, 0.9); // 0.0 - 0.9  1.0
		outputVector.put(0, 1, 0.9); // 0.9 - 1.8  1.5
		outputVector.put(0, 2, 0.2); // 1.8 - 2.0  1.9
		outputVector.put(0, 3, 0.4); // 2.0 - 2.4  2.2
		
		int winner = computationUnit.getWinnerOutputNeuronViaDiceRoll(outputVector);
		System.out.println("winner: " + winner);
		
		assertTrue((winner >= 0) && (winner < outputVector.columns()));
		
	}	
}
