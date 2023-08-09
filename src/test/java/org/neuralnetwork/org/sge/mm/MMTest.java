package org.neuralnetwork.org.sge.mm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.function.DoubleUnaryOperator;

import org.junit.FixMethodOrder;
import org.junit.jupiter.api.Test;
import org.neuralnetworkbasic.ActivationFunction;
import org.sge.math.MathSge;
import org.sge.mm.Board;
import org.sge.mm.ComputationUnit;
import org.sge.mm.GlobalSge;
import org.sge.mm.Guess;
import org.sge.mm.MasterMind;
import org.sge.mm.Move;
import org.sge.mm.Rating;
import org.sge.mm.ResultAllGames;
import org.sge.mm.ResultOneGame;
import org.junit.runners.MethodSorters;
import org.nd4j.linalg.factory.Nd4j;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class MMTest {


	@Test
	public void test02_ComputationCreation() {
		MasterMind masterMind = new MasterMind(6, 4, ActivationFunction.SIGMOID, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);
		ComputationUnit computationUnit = masterMind.getComputationUnit();
		
		assertEquals(1296, computationUnit.getCountOutputNeurons());
		assertEquals(32, computationUnit.getCountInputNeurons());
	}

	/*
	
	@Test
	public void test03_OneGuess() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0.5, 32452345);
		
		String code = "32";
		masterMind.getBoard().setCodeToFind(code);

		
		Guess guess = masterMind.getComputationUnit().getGuessAlgoNN(masterMind.getBoard().getListOfMoves(), true);
		Rating rating = masterMind.getBoard().setGuessOnBoard(guess).rating;
		System.out.println("code to find: " + code + "  guess: " + guess.code + "  rating black: " + rating.countBlack + "  white: " + rating.countWhite);

	
		assertEquals(masterMind.getBoard().getTurnNumber(), 1);
		assertTrue(!masterMind.getBoard().codeFound());
	}

	
	@Test
	public void test04_OneGameLost() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE, 0.5, 12341);
		ResultOneGame gameResult = masterMind.playMachineVsHuman(ComputationUnit.ALGO_NN, "32", false, true);
	
		assertTrue("countMoves < ComputationUnit.MAX_COUNT_GUESSES", gameResult.movesNeeded == masterMind.getBoard().getNumberPossibleCombinations()); 
		assertTrue("countMoves < ComputationUnit.MAX_COUNT_GUESSES", !gameResult.codeFound); 
	}

	
	@Test
	public void test05_OneGameWin() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE, 0.5, 12341);
		ResultOneGame gameResult = masterMind.playMachineVsHuman(ComputationUnit.ALGO_NN, "02", false, true);
	
		assertTrue("countMoves < ComputationUnit.MAX_COUNT_GUESSES", gameResult.movesNeeded <= Board.MAX_COUNT_GUESSES); 
		assertTrue("countMoves < ComputationUnit.MAX_COUNT_GUESSES", gameResult.codeFound); 
	}


	@Test
	public void test06_OneGameWinImmediatly() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE, 0.5, 12341);
		ResultOneGame gameResult = masterMind.playMachineVsHuman(ComputationUnit.ALGO_NN, "00", false, true);
	
		assertTrue("countMoves < ComputationUnit.MAX_COUNT_GUESSES", gameResult.movesNeeded == 1); 
		assertTrue("countMoves < ComputationUnit.MAX_COUNT_GUESSES", gameResult.codeFound); 
	}

	
	@Test
	public void test10_OneGuessHit() {
		MasterMind masterMind = new MasterMind(2, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0.5, 5436456);
		
		String codeCodeToFind = "01";
		masterMind.getBoard().setCodeToFind(codeCodeToFind);

		
		Guess guess = masterMind.getComputationUnit().getGuessAlgoNN(masterMind.getBoard().getListOfMoves(), true);
		Rating rating = masterMind.getBoard().setGuessOnBoard(guess).rating;
		System.out.println("code to find: " + codeCodeToFind + "  guess: " + guess.code + "  rating black: " + rating.countBlack + "  white: " + rating.countWhite);

	
		assertEquals(masterMind.getBoard().getTurnNumber(), 1);
		
		if("00".contentEquals(guess.code)) {
			assertTrue(!masterMind.getBoard().codeFound());
			assertEquals(rating.countBlack, 1);
			assertEquals(rating.countWhite, 0);
		}
		
		if("01".contentEquals(guess.code)) {
			assertTrue(masterMind.getBoard().codeFound());
			assertEquals(rating.countBlack, 2);
			assertEquals(rating.countWhite, 0);
		}
		
	}

	
	@Test
	public void test11_OneGuessTwoWhite() {
		MasterMind masterMind = new MasterMind(2, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0.5, 5675);
		
		String codeToFind = "10";
		masterMind.getBoard().setCodeToFind(codeToFind);
		
		Guess guess = masterMind.getComputationUnit().getGuessAlgoNN(masterMind.getBoard().getListOfMoves(), true);
		Rating rating = masterMind.getBoard().setGuessOnBoard(guess).rating;
		System.out.println("code to find: " + codeToFind + "  guess: " + guess.code + "  rating black: " + rating.countBlack + "  white: " + rating.countWhite);

	
		assertEquals(masterMind.getBoard().getTurnNumber(), 1);

		if("00".contentEquals(guess.code)) {
			assertTrue(!masterMind.getBoard().codeFound());
			assertEquals(rating.countBlack, 1);
			assertEquals(rating.countWhite, 0);
		}
		
		if("01".contentEquals(guess.code)) {
			assertTrue(masterMind.getBoard().codeFound());
			assertEquals(rating.countBlack, 0);
			assertEquals(rating.countWhite, 2);
		}

	}

	
	@Test
	public void test11_OneGuessFourWhite() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0.5, 5675);
		
		String codeToFind = "11";
		masterMind.getBoard().setCodeToFind(codeToFind);

		Guess guess = masterMind.getComputationUnit().getGuessAlgoNN(masterMind.getBoard().getListOfMoves(), true);
		Rating rating = masterMind.getBoard().setGuessOnBoard(guess).rating;
		System.out.println("code to find: " + codeToFind + "  guess: " + guess.code + "  rating black: " + rating.countBlack + "  white: " + rating.countWhite);

	
		assertEquals(masterMind.getBoard().getTurnNumber(), 1);

		if("00".contentEquals(guess.code)) {
			assertTrue(!masterMind.getBoard().codeFound());
			assertEquals(rating.countBlack, 0);
			assertEquals(rating.countWhite, 0);
		}
	}

	
	@Test
	public void test12_OneGuessOneWhite() {
		MasterMind masterMind = new MasterMind(2, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0.5, 86768);
		
		String code = "01";
		masterMind.getBoard().setCodeToFind(code);

		
		Guess guess = masterMind.getComputationUnit().getGuessAlgoNN(masterMind.getBoard().getListOfMoves(), true);
		Rating rating = masterMind.getBoard().setGuessOnBoard(guess).rating;
		System.out.println("code to find: " + code + "  guess: " + guess.code + "  rating black: " + rating.countBlack + "  white: " + rating.countWhite);
	
		assertEquals(masterMind.getBoard().getTurnNumber(), 1);
		assertTrue(!masterMind.getBoard().codeFound());
		assertEquals(rating.countBlack, 0);
		assertEquals(rating.countWhite, 2);
	}

	
	@Test
	public void test20_OneGame21() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0.5, 3425542);
		ResultOneGame gameResult = masterMind.playMachineVsHuman(ComputationUnit.ALGO_NN, "30", false, true);
	
		assertTrue("gameResult.movesNeeded == 16 " + gameResult.movesNeeded, gameResult.movesNeeded == 4); 
		assertTrue("gameResult.movesNeeded <= masterMind.getBoard().getNumberPossibleCombinations()", gameResult.movesNeeded <= masterMind.getBoard().getNumberPossibleCombinations()); 
		assertTrue("!gameResult.codeFound", gameResult.codeFound); 
	}


	@Test
	public void test30_AllCombinations1() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0.5, 3425542);
		ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, false, true);
	
		assertTrue("result.gamesInTotal: " + result.gamesInTotal, result.gamesInTotal == 16); 
		assertTrue("result.gamesWon: " + result.gamesWon, result.gamesWon == 9); 
		assertTrue("result.gamesLost: " + result.gamesLost, result.gamesLost == 7); 
		assertEquals("result.average: " + result.average, result.average, 10, 0.00001);
	}


	@Test
	public void test31_AllCombinations2() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_ALL_WEIGHTS_WITH_THE_SAME_VALUE, 0, 0);
		ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, false, true);
	
		assertTrue("gameResult.movesNeeded == 16", result.gamesInTotal == 16); 
		assertTrue("gameResult.movesNeeded == 11", result.gamesWon == 11); 
		assertTrue("gameResult.movesNeeded == 5", result.gamesLost == 5); 
		assertEquals("gameResult.movesNeeded == 9.0", result.average, 9, 0.00001);
	}
	
	
	private void assertInputVector1(MasterMind masterMind) {
		INDArray inputVector = masterMind.getComputationUnit().computeInputVector(masterMind.getBoard().getListOfMoves());
		System.out.println("input vector assetInputVector: \n" + inputVector.toString());

		assertEquals("input vector", inputVector.get(0, 0), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 1), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 2), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 3), 0, 0.00000000001);
		
		assertEquals("input vector", inputVector.get(0, 4), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 5), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 6), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 7), 0, 0.00000000001);

		assertEquals("input vector", inputVector.get(0, 8), 1, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 9), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,10), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,11), 0, 0.00000000001);

		assertEquals("input vector", inputVector.get(0,12), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,13), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,14), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,15), 0, 0.00000000001);

		assertEquals("input vector", inputVector.get(0,16), 1, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,17), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,18), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,19), 0, 0.00000000001);

		assertEquals("input vector", inputVector.get(0,20), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,21), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,22), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,23), 0, 0.00000000001);
	}
	
	
	private void assertInputVector(MasterMind masterMind) {
		INDArray inputVector = masterMind.getComputationUnit().computeInputVector(masterMind.getBoard().getListOfMoves());
		System.out.println("input vector assetInputVector: \n" + inputVector.toString());

		assertEquals("input vector", inputVector.get(0, 0), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 1), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 2), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 3), 0, 0.00000000001);
		
		assertEquals("input vector", inputVector.get(0, 4), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 5), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 6), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 7), 0, 0.00000000001);

		assertEquals("input vector", inputVector.get(0, 8), 1, 0.00000000001);
		assertEquals("input vector", inputVector.get(0, 9), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,10), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,11), 0, 0.00000000001);

		assertEquals("input vector", inputVector.get(0,12), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,13), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,14), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,15), 0, 0.00000000001);

		assertEquals("input vector", inputVector.get(0,16), 1, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,17), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,18), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,19), 0, 0.00000000001);

		assertEquals("input vector", inputVector.get(0,20), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,21), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,22), 0, 0.00000000001);
		assertEquals("input vector", inputVector.get(0,23), 0, 0.00000000001);
	}
	

	
	@Test
	public void test40_ThreeGuessWithLearning() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 1);
		System.out.println("getCountInputNeurons: " + masterMind.getComputationUnit().getCountInputNeurons());
		System.out.println("getCountOutputNeurons: " + masterMind.getComputationUnit().getCountOutputNeurons());
		
		String code = "32";
		masterMind.getBoard().setCodeToFind(code);
		
		//-------------------------------------------------------------------------------------------------------
		double sum0 = masterMind.getComputationUnit().getSumInputVector(masterMind.getBoard().getListOfMoves());
		System.out.println("sum0 input vector: " + sum0);
		assertEquals("sum0 input vector: " + sum0, 110, masterMind.getComputationUnit().getSumInputVector(masterMind.getBoard().getListOfMoves()), 0.000001);
		
		//-------------------------------------------------------------------------------------------------------
		Guess guess1 = masterMind.getComputationUnit().getGuessAlgoNN(masterMind.getBoard().getListOfMoves(), true);
		Move move1 = masterMind.getBoard().setGuessOnBoard(guess1);
		System.out.println("code to find: " + code + "  guess: " + guess1.code + "  rating black: " + move1.rating.countBlack + "  white: " + move1.rating.countWhite);
		masterMind.getComputationUnit().learnNN(masterMind.getBoard(), false);

		double sum1 = masterMind.getComputationUnit().getSumInputVector(masterMind.getBoard().getListOfMoves());
		System.out.println("sum1 input vector: " + sum1);
		assertEquals("sum1 input vector: " + sum1, 102, masterMind.getComputationUnit().getSumInputVector(masterMind.getBoard().getListOfMoves()), 0.000001);

		//-------------------------------------------------------------------------------------------------------
		Guess guess2 = masterMind.getComputationUnit().getGuessAlgoNN(masterMind.getBoard().getListOfMoves(), true);
		Move move2 = masterMind.getBoard().setGuessOnBoard(guess2);
		System.out.println("code to find: " + code + "  guess: " + guess2.code + "  rating black: " + move2.rating.countBlack + "  white: " + move2.rating.countWhite);
		masterMind.getComputationUnit().learnNN(masterMind.getBoard(), false); //
	
		double sum2 = masterMind.getComputationUnit().getSumInputVector(masterMind.getBoard().getListOfMoves());
		System.out.println("sum2 input vector: " + sum2);
		assertEquals("sum2 input vector: " + sum2, 95, masterMind.getComputationUnit().getSumInputVector(masterMind.getBoard().getListOfMoves()), 0.000001);
		assertInputVector(masterMind);

		System.out.println("isCurrentMoveRatingBetterLastMoveRating: " + masterMind.getComputationUnit().ratingCompare(masterMind.getBoard().getLastMove().rating, masterMind.getBoard().getSecondLastMove().rating));
		assertTrue("isCurrentMoveRatingBetterLastMoveRating ", 1 ==  masterMind.getComputationUnit().ratingCompare(masterMind.getBoard().getLastMove().rating, masterMind.getBoard().getSecondLastMove().rating)); 
		
		//-------------------------------------------------------------------------------------------------------
		Guess guess3 = masterMind.getComputationUnit().getGuessAlgoNN(masterMind.getBoard().getListOfMoves(), true);
		Move move3 = masterMind.getBoard().setGuessOnBoard(guess3);
		System.out.println("code to find: " + code + "  guess: " + guess3.code + "  rating black: " + move3.rating.countBlack + "  white: " + move3.rating.countWhite);
		masterMind.getComputationUnit().learnNN(masterMind.getBoard(), false); //!
		
		double sum3 = masterMind.getComputationUnit().getSumInputVector(masterMind.getBoard().getListOfMoves());
		System.out.println("sum3 input vector: " + sum3);
		assertEquals("sum3 input vector: " + sum3, 87, masterMind.getComputationUnit().getSumInputVector(masterMind.getBoard().getListOfMoves()), 0.000001);

		System.out.println("isCurrentMoveRatingBetterLastMoveRating: " + masterMind.getComputationUnit().ratingCompare(masterMind.getBoard().getLastMove().rating, masterMind.getBoard().getSecondLastMove().rating));
		assertTrue("isCurrentMoveRatingBetterLastMoveRating ", 0 !=  masterMind.getComputationUnit().ratingCompare(masterMind.getBoard().getLastMove().rating, masterMind.getBoard().getSecondLastMove().rating)); 
		
	}
	
	
	@Test
	public void test50_OneGameWithLearning_ALGO_NN() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 1);
		ResultOneGame gameResult = masterMind.playMachineVsHuman(ComputationUnit.ALGO_NN, "00", true, true);
		gameResult.print();
		assertEquals("" + gameResult.movesNeeded, gameResult.movesNeeded, 16);
	
		ResultOneGame gameResult2 = masterMind.playMachineVsHuman(ComputationUnit.ALGO_NN, "32", true, true);
		gameResult2.print();
		assertEquals(gameResult2.movesNeeded, 9);
	}

	
	@Test
	public void test51_AllCombinatoinsWithoutLearning_ALGO_NN() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 1);
		ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, false, false);
		result.print();

		assertTrue("gameResult.gamesInTotal == 16", result.gamesInTotal == 16); 
		assertTrue("gameResult.gamesWon " + result.gamesWon, result.gamesWon == 12); 
		assertTrue("gameResult.gamesLost == 4", result.gamesLost == 4); 
		assertEquals("gameResult.max == 16", result.maxMoves, 16, 0.00001);
		assertEquals("gameResult.average == 9.0", result.average, 9, 0.00001);
	}


	@Test
	public void test52_AllCombinatoinsWithLearning_ALGO_NN() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 1);
	
		for(int i=0; i<1; i++) {
			System.out.println("round: " + i);
			ResultAllGames resultAfterTraining = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, true, false);
			resultAfterTraining.print();
		}

		ResultAllGames resultAfterTraining = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, true, true);
		resultAfterTraining.print();
	}

	
	/*
	@Test
	public void test41_OneGameWithLearning() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 1);
		GameResult gameResult = masterMind.playMachineVsHuman(ComputationUnit.ALGO_NN, "13", true, true);
		gameResult.print();
	
	}
	
	@Test
	public void test42_AllCombinatoinsWithLearning() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 1);
		ResultAllGameCombinations result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, true, false);
		
		result.print();
	
		assertTrue("gameResult.movesNeeded == 16", result.gamesInTotal == 16); 
		assertTrue("gameResult.movesNeeded == 12", result.gamesWon == 12); 
		assertTrue("gameResult.movesNeeded == 4", result.gamesLost == 4); 
		assertEquals("gameResult.movesNeeded == 8.0", result.average, 8, 0.00001);
	}


	@Test
	public void test43_AllCombinatoinsWithLearningMultipleTimes() {
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 1);
		
		for(int i=0; i<1000; i++) {
			ResultAllGameCombinations result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, true, false);
			System.out.println("\nrun number: " + i);
			result.print();
		}
	
	}

	
	@Test
	public void test44_AllCombinatoinsWithLearningMultipleTimes() {
		MasterMind masterMind = new MasterMind(6, 4, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 1);
		
		for(int i=0; i<1000; i++) {
			ResultAllGameCombinations result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, true, false);
			System.out.println("\nrun number: " + i);
			result.print();
		}
	
	}
	
	
	
	@Test
	public void testReferencenInJava() {
		INDArray outputVector = Nd4j.create(1, 5); 
		m.set(0, 0, 1);
		m.set(0, 1, 1);
		m.set(0, 2, 1);
		m.set(0, 3, 1);
		m.set(0, 4, 1);
		
		foo(m);
		
		System.out.println("" + m.get(0, 2));
		assertEquals("copy", 1.0,  m.get(0, 2), 0.000001);
	}

	
	@Test
	public void test70_OneGuessRL() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 444444);
		MathSge.setSgeRandomSeed(2453451);		
		
		String codeToFind = "32";
		masterMind.getBoard().setCodeToFind(codeToFind);
		
		Guess guessPlaying = masterMind.getComputationUnit().getGuessAlgoRLPlaying(masterMind.getBoard().getListOfMoves(), true);
		System.out.println("code to find: " + codeToFind + "  guess: " + guessPlaying.code);
		assertTrue("guessPlaying == " + guessPlaying.code, "31".contentEquals(guessPlaying.code)); 
		masterMind.getBoard().setGuessOnBoard(guessPlaying);

		Guess guessTraining = masterMind.getComputationUnit().getGuessAlgoRLTraining(masterMind.getBoard().getListOfMoves(), true);
		System.out.println("code to find: " + codeToFind + "  guess: " + guessTraining);
		assertTrue("guessTraining == " + guessTraining.code, "03".contentEquals(guessTraining.code)); 
		
		Rating rating = masterMind.getBoard().setGuessOnBoard(guessTraining).rating;
		System.out.println("code to find: " + codeToFind + "  guess: " + guessTraining + "  rating black: " + rating.countBlack + "  white: " + rating.countWhite);
	}
	
	
	@Test
	public void test80_OneGameWithLearning_ALGO_RL() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);

		MathSge.setSgeRandomSeed(1);		
		ResultOneGame gameResult1 = masterMind.playMachineVsHuman(ComputationUnit.ALGO_RL,  0, "01", false, false);
		gameResult1.print();
		assertEquals("" + gameResult1.movesNeeded, gameResult1.movesNeeded, 16);
		assertTrue(!gameResult1.codeFound);

		MathSge.setSgeRandomSeed(1);		
		ResultOneGame gameResult2 = masterMind.playMachineVsHuman(ComputationUnit.ALGO_RL,  0, "01", false, false);
		gameResult2.print();
		assertEquals(gameResult2.movesNeeded, 16);
		assertTrue(!gameResult2.codeFound);
		
		MathSge.setSgeRandomSeed(1);		
		ResultOneGame gameResult3 = null;
		for(int trainingRuns=0; trainingRuns<10; trainingRuns++) {
			gameResult3 = masterMind.playMachineVsHuman(ComputationUnit.ALGO_RL,  0, "01", true, false);
		}
		gameResult3.print();
		assertEquals(gameResult3.movesNeeded, 3);
		assertTrue(gameResult3.codeFound);

		MathSge.setSgeRandomSeed(1);		
		ResultOneGame gameResult4 = masterMind.playMachineVsHuman(ComputationUnit.ALGO_RL,  0, "01", false, false);
		gameResult4.print();
		assertEquals(gameResult4.movesNeeded, 16);
		assertTrue(!gameResult4.codeFound);
	}

	
	//@Test
	public void test90_AllCombinatoinsWithRL() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);
		MathSge.setSgeRandomSeed(1);
		ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_RL,  0, true, false);
		
		result.print();
	
		assertTrue("gameResult.movesNeeded == 16", result.gamesInTotal == 16); 
		assertTrue("gameResult.movesNeeded == 13", result.gamesWon == 13); 
		assertTrue("gameResult.gamesLost == 3", result.gamesLost == 3); 
		assertEquals("result.average == 7.0", result.average, 7, 0.00001);
	}

	
	//@Test
	public void test100_AllCombinatoinsWithLearningMultipleTimes() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);
		masterMind.getComputationUnit().getNn().load();
		
		for(int i=0; i<1000; i++) {
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_RL,  0, true, false);
			System.out.println("\nrun number: " + i);
			result.print();
		}
	
		System.out.println("result of learning");
		ResultAllGames result4 = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_RL,  0, false, false);
		result4.print();
		masterMind.getComputationUnit().getNn().save();
	}
	
	*/
}
