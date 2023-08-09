package org.neuralnetwork.org.sge.mm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.RandomUtils;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.neuralnetworkbasic.ActivationFunction;
import org.sge.math.MathSge;
import org.sge.mm.Board;
import org.sge.mm.ComputationUnit;
import org.sge.mm.GlobalSge;
import org.sge.mm.Guess;
import org.sge.mm.LearnStatisticAllGames;
import org.sge.mm.MasterMind;
import org.sge.mm.Move;
import org.sge.mm.Rating;
import org.sge.mm.ResultAllGames;
import org.sge.mm.ResultOneGame;
import java.sql.Timestamp;

public class AlgoQTest {

	@Test
	public void test05_OneVector() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ComputationUnit computationUnit = masterMind.getComputationUnit();


		INDArray inputVector0 = Nd4j.zeros(1, computationUnit.getCountInputNeurons());
		INDArray inputVector1 = Nd4j.ones(1, computationUnit.getCountInputNeurons());
		INDArray inputVector2 = Nd4j.zeros(1, computationUnit.getCountInputNeurons());
		inputVector2.put(0, 0, 1.0);

		
    	INDArray trainingVector0 = computationUnit.feedForward(1, inputVector0);	
    	INDArray trainingVector1 = computationUnit.feedForward(1, inputVector1);	
    	INDArray trainingVector2 = computationUnit.feedForward(1, inputVector2);	
    	System.out.println("output0: " + trainingVector0.toString());
    	System.out.println("output1: " + trainingVector1.toString());
    	System.out.println("output2: " + trainingVector2.toString());

    	
    	computationUnit.getNN().fit(inputVector0, trainingVector0);
    	computationUnit.getNN().fit(inputVector1, trainingVector1);
    	
    	
		INDArray outputVector2 = Nd4j.zeros(1, computationUnit.getCountOutputNeurons());
		outputVector2.put(0, 0, 1.0);
    	computationUnit.getNN().fit(inputVector2, outputVector2);
		
    	trainingVector0 = computationUnit.feedForward(1, inputVector0);	
    	trainingVector1 = computationUnit.feedForward(1, inputVector1);	
    	trainingVector2 = computationUnit.feedForward(1, inputVector2);	
    	System.out.println("output0: " + trainingVector0.toString());
    	System.out.println("output1: " + trainingVector1.toString());
    	System.out.println("output2: " + trainingVector2.toString());
	}
	


	@Test
	public void test10_OneGameWithLearning_ALGO_Q() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ResultOneGame gameResult1 = masterMind.playMachineVsHuman(ComputationUnit.ALGO_Q, 0, "10", true, true);
		gameResult1.print();
	}


	@Test
	public void test12_TwoGamesWithLearning_ALGO_Q() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ResultOneGame gameResult1 = masterMind.playMachineVsHuman(ComputationUnit.ALGO_Q, 0, "33", true, true);
		gameResult1.print();
		ResultOneGame gameResult2 = masterMind.playMachineVsHuman(ComputationUnit.ALGO_Q, 0, "33", true, true);
		gameResult2.print();
	}

	
	@Test
	public void test15_OneGameRepeatWithLearning_ALGO_Q() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		
		System.out.println("==========================================================================================");
		System.out.println(" find code 00");
		for(int i=0; i<100000; i++) {
			masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(50, "00", ComputationUnit.ALGO_Q, 0.5, true, false);
			ResultAllGames result = masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(50, "00", ComputationUnit.ALGO_Q, 0.0, false, false);
			System.out.println("\nrun number: " + i);
			result.print();
			if(result.gamesWon >= result.gamesInTotal) break;
		}
	}

	
	@Test
	public void test15_OneGameRepeatWithLearningTrainedNet_ALGO_Q() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(50, "33", ComputationUnit.ALGO_Q, 0, true, true);
		
		System.out.println("==========================================================================================");
		System.out.println(" find code 00");
		for(int i=0; i<10000000; i++) {
			masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(1, "00", ComputationUnit.ALGO_Q, 0.5, true, false);
			ResultAllGames result = masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(1, "00", ComputationUnit.ALGO_Q, 0, false, false);
			System.out.println("\nrun number: " + i);
			result.print();
			if(result.gamesWon >= result.gamesInTotal*0.9) break;
		}
	}

	
	@Test
	public void test15_OneGameRepeatWithLearningTwoCombinations_ALGO_Q() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(1, "33", ComputationUnit.ALGO_Q, 0, true, true);
		
		System.out.println("==========================================================================================");
		System.out.println(" find code 00");
		for(int i=0; i<100000; i++) {
			masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(50, "00", ComputationUnit.ALGO_Q, 0.5, true, false);
			ResultAllGames result = masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(50, "00", ComputationUnit.ALGO_Q, 0.0, false, false);
			System.out.println("\nrun number: " + i);
			result.print();
			if(result.gamesWon >= result.gamesInTotal*0.9) break;
		}
		
		System.out.println(" code 00 done ");
		System.out.println("==========================================================================================");
		System.out.println(" find code 01");
		
		for(int i=0; i<100000; i++) {
			masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(50, "01", ComputationUnit.ALGO_Q, 0.5, true, false);
			ResultAllGames result = masterMind.playMachineVsHumanOneFixCombinationMultipleTimes(50, "01", ComputationUnit.ALGO_Q, 1.0, false, false);
			System.out.println("\nrun number: " + i);
			result.print();
			if(result.gamesWon >= result.gamesInTotal*0.9) break;
		}
		
		System.out.println(" code 01 done ");
		System.out.println("==========================================================================================");
		
	}

	
	
	
	
	@Test
	public void test40_CheckOutputVector0() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ComputationUnit computationUnit = masterMind.getComputationUnit();

		
		INDArray inputVector = computationUnit.computeInputVector("01", 0, 0);
		System.out.println("inputVector: " + inputVector);
		// [[         0,    1.0000,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0]]
		// for(int i=0; i<10; i++) computationUnit.getNn().fit(inputVector, computationUnit.feedForward(1, inputVector));
		

    	INDArray outputVector = computationUnit.feedForward(1, inputVector);	
		System.out.println(outputVector);
		// [[    0.0849,    0.0274,    0.1166,    0.0481,    0.0447,    0.0631,    0.0224,    0.0194,    0.0138,    0.0844,    0.0152,    0.2030,    0.0787,    0.1006,    0.0165,    0.0611]]
		//          00      01         02         03         10         11         12         13         20         21         22         23         30         31         32         33
		//          0.0     0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.25       0.25       0.0        0.0        0.25       0.25   
		

		// possible solutions
		// ArrayList<String> codesToTest = new ArrayList<String>(Arrays.asList("22", "23", "32", "33"));

		double s1 = 0;
		double s2 = 0;
		for(int i=0; i<100000; i++) { 
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			
			System.out.println("\nrun number: " + i);
			
			INDArray op = computationUnit.feedForward(1, inputVector);
			System.out.println(op);
			s1 = op.getDouble(0,2) + op.getDouble(0,3);
			s2 = op.getDouble(0,6) + op.getDouble(0,7);
			if(s1 > 0.9 && s2>0.9 && i>100) break;
		}		

		assertTrue(s1 > 0.9 && s2>0.9);
	}	
	

	@Test
	public void test40_CheckOutputVector2() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ComputationUnit computationUnit = masterMind.getComputationUnit();
		INDArray inputVector1 = computationUnit.computeInputVector("01", 0, 0);
		System.out.println("inputVector1: " + inputVector1);


		List<Move> moveList = new ArrayList<Move>();
		Move move0 = new Move();
		moveList.add(move0);
		Guess guess0 = new Guess();
		Rating rating0 = new Rating();

		move0.guess = guess0;
		move0.rating = rating0;
		
		guess0.code = new Guess().code = "01";
		
		Move move1 = new Move();
		moveList.add(move1);
		
		Guess guess1 = new Guess();
		Rating rating1 = new Rating();

		move1.guess = guess1;
		move1.rating = rating1;
		
		guess1.code = new Guess().code = "33";
		rating1.countWhite = 0;
		rating1.countBlack = 0;
		
		INDArray inputVector2 = computationUnit.computeInputVector(moveList, 2);
		System.out.println("inputVector2: " + inputVector2);

		// possible solutions
		ArrayList<String> codesToTest = new ArrayList<String>(Arrays.asList("22", "23", "32", "33"));

		
		for(int i=0; i<100; i++) {
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			// masterMind.playMachineVsHumanSetOfCombinationMultipleTimes(1, codesToTest, ComputationUnit.ALGO_Q, 1.0, true, false);
		}


		double winrate  = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double explorationRate = 1.0;
		double s1 = 0.0;
		double s2 = 0.0;
		double t1 = 0.0;
		double t2 = 0.0;
		for(int i=0; i<200000; i++) { 
			// masterMind.playMachineVsHumanSetOfCombinationMultipleTimes(1, codesToTest, ComputationUnit.ALGO_Q, explorationRate, true, false);
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, explorationRate, true, false);

			if(i%100 == 0) {
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, 0.0, false, false);
				winrate = (double)result.gamesWon / (double)result.gamesInTotal;

				winrate1 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
				winrate2 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
				winrate3 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
			}

			explorationRate = explorationRate - 0.0001;
			if(explorationRate < 0.0) explorationRate = 0.0;
			
			
			System.out.println("\nrun number: " + i + "   explorationRate: " + explorationRate + "   winrate: " + winrate + "   winrate1: " + winrate1 + "   winrate2: " + (winrate2 + winrate1) +"   winrate3: " + (winrate3+ winrate2 + winrate1));
			// For comparison: Algo Exclude: winrate1: 0.0625   winrate2: 0.25   winrate3: 0.4375

			INDArray op1 = computationUnit.feedForward(1, inputVector1);
			System.out.println(op1);
			
			INDArray op2 = computationUnit.feedForward(2, inputVector2);
			System.out.println(op2);
			
			s1 = op1.getDouble(0,2) + op1.getDouble(0,3);
			s2 = op1.getDouble(0,6) + op1.getDouble(0,7);
			
			t1 = op2.getDouble(0, 2);
			t2 = op2.getDouble(0, 6);

			if(s1 > 0.9 && s2>0.9 && t1 > 0.9 && t2>0.9 &&  i>100) break;
		}		

		assertTrue(s1 > 0.9 && s2>0.9 && t1 > 0.9 && t2>0.9);
	}
	
	
	@Test
	public void test40_CheckOutputVector_33() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ComputationUnit computationUnit = masterMind.getComputationUnit();
		
		INDArray inputVector = computationUnit.computeInputVector("01", 0, 0);
		System.out.println(inputVector);
		// [[         0,    1.0000,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0]]
		for(int i=0; i<10; i++) computationUnit.getNN().fit(inputVector, computationUnit.feedForward(1, inputVector));
		

    	INDArray outputVector = computationUnit.feedForward(1, inputVector);	
		System.out.println(outputVector);
		// [[    0.0849,    0.0274,    0.1166,    0.0481,    0.0447,    0.0631,    0.0224,    0.0194,    0.0138,    0.0844,    0.0152,    0.2030,    0.0787,    0.1006,    0.0165,    0.0611]]
		//          00      01         02         03         10         11         12         13         20         21         22         23         30         31         32         33
		//          0.0     0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.25       0.25       0.0        0.0        0.25       0.25   
		
		ArrayList<String> codesToTest = new ArrayList<String>(Arrays.asList("33"));

		double s1 = 0.0;
		double s2 = 0.0;
		for(int i=0; i<10000; i++) {
			// ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0, true, false);
			masterMind.playMachineVsHumanSetOfCombinationMultipleTimes(1, codesToTest, ComputationUnit.ALGO_Q, 0.5, true, false);
			
			System.out.println("\nrun number: " + i);
			
			INDArray op = computationUnit.feedForward(1, inputVector);
			System.out.println(op);
			s1 = op.getDouble(0, 3);
			s2 = op.getDouble(0, 7);
			if((s1 > 0.9) && (s2 > 0.9)) break;
		}		

		assertTrue((s1 > 0.9) && (s2 > 0.9));
	}	
		
	
	@Test
	public void test42_CheckOutputVector() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ComputationUnit computationUnit = masterMind.getComputationUnit();
		
		INDArray inputVector = computationUnit.computeInputVector("01", 0, 0);
		System.out.println(inputVector);
		// [[         0,    1.0000,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0]]
		for(int i=0; i<10; i++) computationUnit.getNN().fit(inputVector, computationUnit.feedForward(1, inputVector));
		

    	INDArray outputVector = computationUnit.feedForward(1, inputVector);	
		System.out.println(outputVector);
		// [[    0.0849,    0.0274,    0.1166,    0.0481,    0.0447,    0.0631,    0.0224,    0.0194,    0.0138,    0.0844,    0.0152,    0.2030,    0.0787,    0.1006,    0.0165,    0.0611]]
		//          00      01         02         03         10         11         12         13         20         21         22         23         30         31         32         33
		//          0.0     0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.25       0.25       0.0        0.0        0.25       0.25   
		
		ArrayList<String> codesToTest = new ArrayList<String>(Arrays.asList("01", "33"));

		double s1=0.0;
		double s2=0.0;
		for(int i=0; i<10000; i++) {
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			
			System.out.println("\nrun number: " + i);
			System.out.println(computationUnit.feedForward(1, inputVector));
			
			INDArray op = computationUnit.feedForward(1, inputVector);
			s1 = op.getDouble(0,2) + op.getDouble(0,3);
			s2 = op.getDouble(0,6) + op.getDouble(0,7);
			if((s1 > 0.9) && (s2>0.9) && (i>1000)) break;
		}		

		assertTrue(s1 > 0.8 && s2>0.8);
	}	

	
	public final static double WINRATE_ALGO_RANDOM = 0.12109375;
	
	@Test
	public void test00_Zero_Test() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		double winrate = 0.0;
		ResultAllGames overAllResult = new ResultAllGames();
		while(winrate < WINRATE_ALGO_RANDOM)
		for(int i=0; i<1000; i++) {
			System.out.println("\nrun number: " + i);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_RANDOM, 0.0, false, false);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate);
		}
		
		assertTrue(winrate >= WINRATE_ALGO_RANDOM);		
	}	
	
	
	@Test
	public void test00_Zero_Test_2_2() {
		MasterMind masterMind = new MasterMind(2, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ResultAllGames overAllResult = new ResultAllGames();
		for(int i=0; i<1000; i++) {
			System.out.println("\nrun number: " + i);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_RANDOM, 0.0, false, false);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate);
			double d = winrate - WINRATE_ALGO_RANDOM;
		}
		
		double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
        assertTrue(winrate >= 0.40);		
	}		


	@Test
	public void test00_Zero_Test_2_3() {
		MasterMind masterMind = new MasterMind(2, 3, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ResultAllGames overAllResult = new ResultAllGames();
		for(int i=0; i<1000; i++) {
			System.out.println("\nrun number: " + i);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_RANDOM, 0.0, false, false);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate);
			double d = winrate - WINRATE_ALGO_RANDOM;
		}
		
		double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
        assertTrue(winrate >= 0.22);		
	}		


	@Test
	public void test00_Zero_Test_6_4() {
		MasterMind masterMind = new MasterMind(6, 4, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ResultAllGames overAllResult = new ResultAllGames();
		for(int i=0; i<10; i++) {
			System.out.println("\nrun number: " + i);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_RANDOM, 0.0, false, false);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate);
			double d = winrate - WINRATE_ALGO_RANDOM;
		}
		
		double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
		// 0.00148269
	}		


	@Test
	public void test00_Zero_Test_ALGO_EXCLUDE_4_2() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ResultAllGames overAllResult = new ResultAllGames();
		double overAllwinrate = 0.0;
		double winrate  = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		int i=0;
		while(overAllwinrate < 0.4375) {
			System.out.println("\nrun number: " + i++);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, 0.0, false, false);
			
			winrate = (double)result.gamesWon / (double)result.gamesInTotal;

			winrate1 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
			winrate2 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
			winrate3 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
			
			System.out.println("\nrun number: " + i + "   winrate: " + winrate + "   winrate1: " + winrate1 + "   winrate2: " + (winrate2) +"   winrate3: " + (winrate3));
			
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			overAllwinrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + overAllwinrate); // winrate: 0.75   winrate1: 0.0625   winrate2: 0.25   winrate3: 0.4375
		}
		
		assertTrue(winrate >= 0.4375);		
	}	
	

	@Test
	public void test00_Zero_Test_ALGO_EXCLUDE_2_2() {
		MasterMind masterMind = new MasterMind(2, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ResultAllGames overAllResult = new ResultAllGames();
		for(int i=0; i<1; i++) {
			System.out.println("\nrun number: " + i);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, 0.0, false, false);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate);
			double d = winrate - 0.75;
			System.out.println("d: " + Math.abs(d));		
		}
		
		double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
		assertTrue(winrate >= 0.74);		
	}		
	

	
	@Test
	public void test00_Zero_Test_ALGO_EXCLUDE_2_3() {
		MasterMind masterMind = new MasterMind(2, 3, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ResultAllGames overAllResult = new ResultAllGames();
		for(int i=0; i<1; i++) {
			System.out.println("\nrun number: " + i);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, 0.0, false, true);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate);
			double d = winrate - 0.61;
			System.out.println("d: " + Math.abs(d));		
		}
		
		double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
		assertTrue(winrate >= 0.5);		
	}		
	

	@Test
	public void test00_Zero_Test_ALGO_EXCLUDE_3_2() {
		MasterMind masterMind = new MasterMind(3, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ResultAllGames overAllResult = new ResultAllGames();
		for(int i=0; i<1; i++) {
			System.out.println("\nrun number: " + i);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, 0.0, false, false);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate);
		}
		
		double winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
		assertTrue(winrate >= 0.44);		
	}		

	
	@Test
	public void test00_Zero_Test_ALGO_EXCLUDE_4_3() {
		MasterMind masterMind = new MasterMind(4, 3, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		double winrate = 0.0;
		ResultAllGames overAllResult = new ResultAllGames();
		for(int i=0; i<1; i++) {
			System.out.println("\nrun number: " + i++);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, 0.0, false, false);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate);
		}
		
		assertTrue(winrate >=  0.005);	// 0.001 random
	}		

	
	@Test
	public void test00_Zero_Test_ALGO_EXCLUDE_6_4() {
		MasterMind masterMind = new MasterMind(6, 4, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		double winrate = 0.0;
		ResultAllGames overAllResult = new ResultAllGames();
		for(int i=0; i<1; i++) {
			System.out.println("\nrun number: " + i++);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, 0.0, false, false);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate);
		}
		
		assertTrue(winrate >=  0.005);	// 0.001 random
	}		

	
	@Test
	public void test00_Zero_Test_ALGO_Q() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		
		double explo = 1.0;
		double winrate = 0.0;
		int l=0;
		while((winrate < 0.40) || (l<10)) {
			l++;
			
			for(int i=0; i<100; i++) {
				masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			}
			
			explo = explo - 0.01;
			if(explo < 0) explo = 0.0;
			
			ResultAllGames overAllResult = new ResultAllGames();
			for(int i=0; i<100; i++) {
				System.out.println("\nrun number eval: " + i + "  loop: " + l);
				
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0.0, false, false);
				overAllResult = overAllResult.add(result); 
				// overAllResult.print();
				
				winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
				System.out.println("winrate: " + winrate + "  explo: " + explo);
			}
		}
		assert(winrate >= 0.40);
	}		


	@Test
	public void test00_Zero_Test_ALGO_Q_2_2() {
		MasterMind masterMind = new MasterMind(2, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		
		double explo = 1.0;
		double winrate = 0.0;
		for(int l=0; l<200; l++) {
			
			for(int i=0; i<100; i++) {
				masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			}
			
			explo = explo - 0.01;
			if(explo < 0) explo = 0.0;
			
			ResultAllGames overAllResult = new ResultAllGames();
			for(int i=0; i<100; i++) {
				System.out.println("\nrun number eval: " + i + "  loop: " + l);
				
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0.0, false, false);
				overAllResult = overAllResult.add(result); 
				overAllResult.print();
				
				winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			}
			System.out.println("winrate: " + winrate + "  explo: " + explo);
			if(winrate >= 0.75) break;
			
		}
		assert(winrate >= 0.75);
	}		

	
	@Test
	public void test00_Zero_Test_ALGO_Q_2_3() {
		MasterMind masterMind = new MasterMind(2, 3, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		
		double explo = 1.0;
		double winrate = 0.0;
		for(int l=0; l<200; l++) {

			for(int i=0; i<100; i++) {
				masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			}
			
			explo = explo - 0.01;
			if(explo < 0) explo = 0.0;
			
			ResultAllGames overAllResult = new ResultAllGames();
			for(int i=0; i<100; i++) {
				System.out.println("\nrun number eval: " + i + "  loop: " + l);
				
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0.0, false, false);
				overAllResult = overAllResult.add(result); 
				overAllResult.print();
				
				winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			}
			System.out.println("winrate: " + winrate + "  explo: " + explo);
			if(winrate >= 0.5) break;
			
		}
		assertTrue(winrate >= 0.5);
	}		


	@Test
	public void test00_Zero_Test_ALGO_Q_3_2() {
		MasterMind masterMind = new MasterMind(3, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		masterMind.getComputationUnit().setFirstMoveFixNatural();  
		
		double explo = 1.0;
		double winrate = 0.0;
		long loop = -1;
		while(winrate < 0.55) {
			loop++;
			for(int i=0; i<100; i++) {
				masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, explo, true, false);
			}
			
			explo = explo - 0.01;
			if(explo < 0) explo = 0.0;
			
			ResultAllGames overAllResult = new ResultAllGames();
			for(int i=0; i<100; i++) {
				System.out.println("\nrun number eval: " + i + "  loop: " + loop);
				
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0.0, false, false);
				overAllResult = overAllResult.add(result); 
				overAllResult.print();
				
			}
			winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate + "  explo: " + explo);
			if(winrate >= 0.4) break;
		}
		
		assertTrue(winrate >= 0.4);
	}		


	@Test
	public void test00_Zero_Test_ALGO_Q_4_2() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		masterMind.getComputationUnit().setFirstMoveFixNatural();  
		
		for(int i=0; i<1000; i++) {
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
		}

		double winrate = 0.0;
		long loop = -1;
		while(winrate < 0.28) {
			loop++;
			for(int i=0; i<1000; i++) {
				masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0.0, true, false);
			}
			
			if(loop % 10 == 0) {
				ResultAllGames overAllResult = new ResultAllGames();
				for(int i=0; i<100; i++) {
					ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0.0, false, false);
					overAllResult = overAllResult.add(result); 
				}
				
				System.out.println("\nrun number eval: " + loop);
				overAllResult.print();
				winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			}
		}
		assertTrue(winrate >= 0.28);
	}
	

	/*
	@Test
	public void test00_Zero_Test_ALGO_Q_6_4() {
		// 6, 4 random: // 0.0015
		// 6, 4 exclude // 0.0050 
		// 6, 4 algoQ   // 0.0046
		                
		
		MasterMind masterMind = new MasterMind(6, 4, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		masterMind.getComputationUnit().setFirstMoveFixNatural();  
		
		double explo = 1.0;
		double winrate = 0.0;
		ResultAllGames overAllResult = new ResultAllGames();
		long loop = -1;
		while(winrate < 0.0045) {
			loop++;
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0.5, true, false);
			
			explo = explo - 0.01;
			if(explo < 0) explo = 0.0;
			
			System.out.println("\nrun number eval: " + loop);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0.0, false, false);
			overAllResult = overAllResult.add(result); 
			overAllResult.print();
			
			winrate = (double)overAllResult.gamesWon / (double)overAllResult.gamesInTotal;
			System.out.println("winrate: " + winrate + "  explo: " + explo);
		}
	}
	*/
	
	
	@Test
	public void test60_Zero_Test_ALGO_Q() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		for(int i=0; i<1000; i++) {
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
		}
		
		
		// ResultAllGames overAllResult = new ResultAllGames();
		double winrate = 0.0;
		int i=0;
		while(winrate < 0.4375 || i<1000) {
			System.out.println("\nrun number: " + i++);
			
			ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 0.0, false, false);
			result.print();
			// overAllResult = overAllResult.add(result); 
			// overAllResult.print();
			
			winrate = (double)result.gamesWon / (double)result.gamesInTotal;
			// System.out.println("winrate: " + winrate);
		}
		
		// assertTrue(winrate >= 0.30);		
	}
	
	
	@Test
	public void test80_FirstLayerCheck() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		ComputationUnit computationUnit = masterMind.getComputationUnit();
		
		INDArray inputVector1 = computationUnit.computeInputVector("01", 0, 0); // 22, 23, 32, 33
		INDArray inputVector2 = computationUnit.computeInputVector("01", 0, 1); // 00, 02, 03, 11, 21, 31     
		INDArray inputVector3 = computationUnit.computeInputVector("01", 1, 0);
		INDArray inputVector4 = computationUnit.computeInputVector("01", 2, 0);

		System.out.println("inputVector1: " + inputVector1);
		System.out.println("inputVector2: " + inputVector2);
		System.out.println("inputVector3: " + inputVector3);
		System.out.println("inputVector4: " + inputVector4);
		
		// [[         0,    1.0000,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0,         0]]
		// for(int i=0; i<10; i++) computationUnit.getNn().fit(inputVector, computationUnit.feedForward(1, inputVector));
		

    	INDArray outputVector = computationUnit.feedForward(1, inputVector1);	
		System.out.println(outputVector);
		
		//          00      01         02         03         04         05         06         07         08         09         10         11         12         13         14         15
		//          00      01         02         03         10         11         12         13         20         21         22         23         30         31         32         33
		
		// ip1      0.0     0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.0        0.25       0.25       0.0        0.0        0.25       0.25   
		// ip2      y       n          y          y          n          y          n          n          n          y          n          n          n          y          n          n
		// ip3      n       n          n          n          n          n          y          y          y          n          n          n          y          n          n          n
		// ip4      n       n          n          n          y          n          n          n          n          n          n          n          n          n          n          n

		// possible solutions
		ArrayList<String> resultCodes1 = new ArrayList<String>(Arrays.asList("22", "23", "32", "33"));
		ArrayList<String> resultCodes2 = new ArrayList<String>(Arrays.asList("00", "02", "03", "11", "21", "31"));
		ArrayList<String> resultCodes3 = new ArrayList<String>(Arrays.asList("12", "13", "20", "30"));
		ArrayList<String> resultCodes4 = new ArrayList<String>(Arrays.asList("10"));

		
		
		double op1_sum = 0;
		boolean op1_condition = false; 
		
		double op2_sum = 0;
		boolean op2_condition = false; 
		
		double op3_sum = 0;
		boolean op3_condition = false; 
		
		double op4_sum = 0;
		boolean op4_condition = false; 
		
		double winrate = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		
		for(int i=0; i<100000; i++) { 
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			
			INDArray op1 = computationUnit.feedForward(1, inputVector1);
			INDArray op2 = computationUnit.feedForward(1, inputVector2);
			INDArray op3 = computationUnit.feedForward(1, inputVector3);
			INDArray op4 = computationUnit.feedForward(1, inputVector4);
			
			System.out.println(op1 + "  " + computationUnit.getWinnerCodeFromOutputVectorMaxValueMethod(op1)  + " " + op1_sum + "  winrate:  "  + winrate);
			System.out.println(op2 + "  " + computationUnit.getWinnerCodeFromOutputVectorMaxValueMethod(op2)  + " " + op2_sum + "  winrate0: " + winrate0);
			System.out.println(op3 + "  " + computationUnit.getWinnerCodeFromOutputVectorMaxValueMethod(op3)  + " " + op3_sum + "  winrate1: " + winrate1);
			System.out.println(op4 + "  " + computationUnit.getWinnerCodeFromOutputVectorMaxValueMethod(op4)  + " " + op4_sum + "  winrate2: " + winrate2);
			
			op1_sum = op1.getDouble(0, 10) + op1.getDouble(0, 11) + op1.getDouble(0, 14) + op1.getDouble(0, 15);
			op1_condition = op1_sum > 0.9;

			op2_sum = op2.getDouble(0, 0) + op2.getDouble(0, 2) + op2.getDouble(0, 3) + op2.getDouble(0, 5) + op2.getDouble(0, 9) + op2.getDouble(0, 13);
			op2_condition = op2_sum > 0.9;

			op3_sum = op3.getDouble(0, 6) + op3.getDouble(0, 7) + op3.getDouble(0, 8) + op3.getDouble(0, 12); 
			op3_condition = op2_sum > 0.9;

			op4_sum = op4.getDouble(0, 4); 
			op4_condition = op2_sum > 0.9;

			if(op1_condition && op2_condition && op3_condition && op4_condition && i>100) break;
			
			
			if(i%100 == 0) {
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN_WITH_DUPLICATES, 0.0, false, false);
				winrate = (double)result.gamesWon / (double)result.gamesInTotal;

				winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
				winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
				winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
			}
			
			System.out.println("run number: " + i);
			System.out.println();
			
		}		

		assertTrue(op1_condition && op2_condition && op3_condition && op4_condition);
		
		
		computationUnit.saveLayer1();
		computationUnit.saveLayer2();
		
		assertTrue(resultCodes1.contains(computationUnit.getWinnerCodeFromOutputVectorMaxValueMethod(computationUnit.feedForward(1, inputVector1))));
		assertTrue(resultCodes2.contains(computationUnit.getWinnerCodeFromOutputVectorMaxValueMethod(computationUnit.feedForward(1, inputVector2))));
		assertTrue(resultCodes3.contains(computationUnit.getWinnerCodeFromOutputVectorMaxValueMethod(computationUnit.feedForward(1, inputVector3))));
		assertTrue(resultCodes4.contains(computationUnit.getWinnerCodeFromOutputVectorMaxValueMethod(computationUnit.feedForward(1, inputVector4))));
	}	
	
	
	@Test
	public void test80_MM_4_2_WinrateCheck() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		// computationUnit.loadLayer1();
		// computationUnit.setLayer1InReadModus();
		// computationUnit.loadLayer2();
		// computationUnit.setLayer2InReadModus();
		
		double winrate  = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double winrate4 = 0.0;
		
		for(int i=0; i<100000; i++) { 
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			
			System.out.println("  winrate:  " + winrate);
			System.out.println("  winrate0: " + winrate0);
			System.out.println("  winrate1: " + winrate1);
			System.out.println("  winrate2: " + winrate2);
			System.out.println("  winrate3: " + winrate3);
			System.out.println("  winrate4: " + winrate4);
			
			// if((winrate1 >= 0.25) && (winrate2 >= 0.4375)) break;
			if(winrate >= 1.0) break;
			// if(winrate2 >= 0.5) break;
			
			int cdw = GlobalSge.countWarningsDuplicateMoves;
			if(i%100 == 0) {
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN_WITH_DUPLICATES, 0.0, false, false);
				winrate = (double)result.gamesWon / (double)result.gamesInTotal;

				winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
				winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
				winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
				winrate3 = (double)result.gamesWonInFourMoves  / (double)result.gamesInTotal;
				winrate4 = (double)result.gamesWonInFiveMoves  / (double)result.gamesInTotal;
			}
			
			System.out.println("run number: " + i + "  countWarningsDuplicateMoves: " + (GlobalSge.countWarningsDuplicateMoves - cdw));
			System.out.println();
		}		

		
		// computationUnit.saveLayer1();
		// computationUnit.saveLayer2();
	}	
	
	
	@Test
	public void test80_MM_4_3_WinrateCheck_Exclude() {
		MasterMind masterMind = new MasterMind(4, 3, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		double winrate  = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double winrate4 = 0.0;
		
		
		ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, 0.0, false, false);
		winrate = (double)result.gamesWon / (double)result.gamesInTotal;

		winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
		winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
		winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
		winrate3 = (double)result.gamesWonInFourMoves  / (double)result.gamesInTotal;
		winrate4 = (double)result.gamesWonInFiveMoves  / (double)result.gamesInTotal;
		
		System.out.println("  winrate:  " + winrate);
		System.out.println("  winrate0: " + winrate0);
		System.out.println("  winrate1: " + winrate1); // 0.125 
		System.out.println("  winrate2: " + winrate2);
		System.out.println("  winrate3: " + winrate3);
		System.out.println("  winrate4: " + winrate4);
	}	
	
	
	@Test
	public void test80_MM_4_3_WinrateCheck() {
		MasterMind masterMind = new MasterMind(4, 3, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		// computationUnit.loadLayer1();
		// computationUnit.setLayer1InReadModus();
		// computationUnit.loadLayer2();
		// computationUnit.setLayer2InReadModus();
		
		double winrate  = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double winrate4 = 0.0;
		
		for(int i=0; i<100000; i++) { 
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			
			System.out.println("  winrate:  " + winrate);
			System.out.println("  winrate0: " + winrate0);
			System.out.println("  winrate1: " + winrate1);  // 0.125 run: 2910 or run: 8330
			System.out.println("  winrate2: " + winrate2);
			System.out.println("  winrate3: " + winrate3);
			System.out.println("  winrate4: " + winrate4);
			
			if(winrate1 >= 0.125) break;
			
			int cdw = GlobalSge.countWarningsDuplicateMoves;
			if(i%10 == 0) {
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN_WITH_DUPLICATES, 0.0, false, false);
				winrate = (double)result.gamesWon / (double)result.gamesInTotal;

				winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
				winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
				winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
				winrate3 = (double)result.gamesWonInFourMoves  / (double)result.gamesInTotal;
				winrate4 = (double)result.gamesWonInFiveMoves  / (double)result.gamesInTotal;
			}
			
			System.out.println("run number: " + i + "  countWarningsDuplicateMoves: " + (GlobalSge.countWarningsDuplicateMoves - cdw));
			System.out.println();
		}		

		// masterMind.getComputationUnit().saveLayer1();
		// computationUnit.saveLayer2();
	}	
	@Test
	public void test80_MM_5_4_WinrateCheck_Exclude() {
		MasterMind masterMind = new MasterMind(5, 4, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		double winrate  = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double winrate4 = 0.0;
		
		
		ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, 0.0, false, false);
		winrate = (double)result.gamesWon / (double)result.gamesInTotal;

		winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
		winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
		winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
		winrate3 = (double)result.gamesWonInFourMoves  / (double)result.gamesInTotal;
		winrate4 = (double)result.gamesWonInFiveMoves  / (double)result.gamesInTotal;
		
		System.out.println("  winrate:  " + winrate);
		System.out.println("  winrate0: " + winrate0);
		System.out.println("  winrate1: " + winrate1);  // 0.0208
		System.out.println("  winrate2: " + winrate2);
		System.out.println("  winrate3: " + winrate3);
		System.out.println("  winrate4: " + winrate4);
	}	
	
	
	@Test
	public void test80_MM_5_4_WinrateCheck() {
		MasterMind masterMind = new MasterMind(5, 4, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		// computationUnit.loadLayer1();
		// computationUnit.setLayer1InReadModus();
		// computationUnit.loadLayer2();
		// computationUnit.setLayer2InReadModus();
		
		double winrate  = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double winrate4 = 0.0;
		
		for(int i=0; i<100000; i++) { 
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			
			System.out.println("  winrate:  " + winrate);
			System.out.println("  winrate0: " + winrate0);
			System.out.println("  winrate1: " + winrate1);  // 0.0208  
			System.out.println("  winrate2: " + winrate2);
			System.out.println("  winrate3: " + winrate3);
			System.out.println("  winrate4: " + winrate4);
			
			if(winrate1 >= 0.0208) break;
			
			int cdw = GlobalSge.countWarningsDuplicateMoves;
			if(i%10 == 0) {
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN_WITH_DUPLICATES, 0.0, false, false);
				winrate = (double)result.gamesWon / (double)result.gamesInTotal;

				winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
				winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
				winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
				winrate3 = (double)result.gamesWonInFourMoves  / (double)result.gamesInTotal;
				winrate4 = (double)result.gamesWonInFiveMoves  / (double)result.gamesInTotal;
			}
			
			System.out.println("run number: " + i + "  countWarningsDuplicateMoves: " + (GlobalSge.countWarningsDuplicateMoves - cdw));
			System.out.println();
		}		

		
		masterMind.getComputationUnit().saveLayer1();
	}	
	
	
	@Test
	public void test80_MM_4_4_WinrateCheck() {
		MasterMind masterMind = new MasterMind(4, 4, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		// computationUnit.loadLayer1();
		// computationUnit.setLayer1InReadModus();
		// computationUnit.loadLayer2();
		// computationUnit.setLayer2InReadModus();
		
		double winrate  = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double winrate4 = 0.0;
		
		for(int i=0; i<100000; i++) { 
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			
			System.out.println("  winrate:  " + winrate);
			System.out.println("  winrate0: " + winrate0);
			System.out.println("  winrate1: " + winrate1);  // 0.04296875 runs 16040 only positives, runs also negatives  
			System.out.println("  winrate2: " + winrate2);
			System.out.println("  winrate3: " + winrate3);
			System.out.println("  winrate4: " + winrate4);
			
			if(winrate1 >= 0.04296875) break;
			
			int cdw = GlobalSge.countWarningsDuplicateMoves;
			if(i%10 == 0) {
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN_WITH_DUPLICATES, 0.0, false, false);
				winrate = (double)result.gamesWon / (double)result.gamesInTotal;

				winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
				winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
				winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
				winrate3 = (double)result.gamesWonInFourMoves  / (double)result.gamesInTotal;
				winrate4 = (double)result.gamesWonInFiveMoves  / (double)result.gamesInTotal;
			}
			
			System.out.println("run number: " + i + "  countWarningsDuplicateMoves: " + (GlobalSge.countWarningsDuplicateMoves - cdw));
			System.out.println();
		}		

		
		// masterMind.getComputationUnit().saveLayer1();
	}	
	
	
	@Test
	public void test80_MM_6_4_WinrateCheck_Exclude() {
		MasterMind masterMind = new MasterMind(6, 4, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		double winrate  = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double winrate4 = 0.0;
		double winrate5 = 0.0;
		
		
		ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, 0.0, false, false);
		winrate = (double)result.gamesWon / (double)result.gamesInTotal;

		winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
		winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal;
		winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal;
		winrate3 = (double)result.gamesWonInFourMoves  / (double)result.gamesInTotal;
		winrate4 = (double)result.gamesWonInFiveMoves  / (double)result.gamesInTotal;
		winrate5 = (double)result.gamesWonInSixMoves   / (double)result.gamesInTotal;
		
		System.out.println("  winrate:  " + winrate);   // 0.8950617283950617
		System.out.println("  winrate0: " + winrate0);  // 7.716049382716049E-4
		System.out.println("  winrate1: " + winrate1);  // 0.010030864197530864
		System.out.println("  winrate2: " + winrate2);  // 0.05555555555555555
		System.out.println("  winrate3: " + winrate3);  // 0.19367283950617284
		System.out.println("  winrate4: " + winrate4);  // 0.34953703703703703
		System.out.println("  winrate5: " + winrate5);  // 0.2854938271604938
	}	
	
	
	
	@Test
	public void test80_MM_4_2_WinrateCheckPlus() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		// masterMind.getComputationUnit().loadLayer1();
		// masterMind.getComputationUnit().setLayer1InReadModus();
		
		// masterMind.getComputationUnit().loadLayer2();
		// masterMind.getComputationUnit().setLayer2InReadModus();

		// masterMind.getComputationUnit().loadLayer3();
		// masterMind.getComputationUnit().setLayer3InReadModus();

		// masterMind.getComputationUnit().loadLayer4();
		// masterMind.getComputationUnit().setLayer4InReadModus();
		
		// masterMind.getComputationUnit().loadLayer5();
		// masterMind.getComputationUnit().setLayer5InReadModus();
		
		
		double averageMoves = 0.0;
		double winrate  = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double winrate4 = 0.0;
		double winrate5 = 0.0;
		
		int dm = 0;
		
		ArrayList<String> codeListToGuess = new ArrayList<String>(); 
		
		for(int i=0; i<100000; i++) { 
			// masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			masterMind.playMachineVsHumanSetOfCombinationMultipleTimes(1, codeListToGuess, ComputationUnit.ALGO_Q, 1.0, true, false);
			
			int cdw = GlobalSge.countWarningsDuplicateMoves;
			if(i%10 == 0) {
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN_WITH_DUPLICATES, 0.0, false, false);
				codeListToGuess = result.notFoundCodeList;

				winrate = (double)result.gamesWon / (double)result.gamesInTotal;

				averageMoves = result.average;
				winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
				winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal; 
				winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal; 
				winrate3 = (double)result.gamesWonInFourMoves  / (double)result.gamesInTotal;
				winrate4 = (double)result.gamesWonInFiveMoves  / (double)result.gamesInTotal;
				winrate5 = (double)result.gamesWonInSixMoves   / (double)result.gamesInTotal;
				
				dm = (GlobalSge.countWarningsDuplicateMoves - cdw);
			}

			System.out.println("  winrate:  " + winrate + "  average moves: " + averageMoves);
			System.out.println("  winrate0: " + winrate0);
			System.out.println("  winrate1: " + winrate1);  
			System.out.println("  winrate2: " + winrate2);  
			System.out.println("  winrate3: " + winrate3);
			System.out.println("  winrate4: " + winrate4);
			System.out.println("  winrate5: " + winrate5);
			
			if(winrate >= 1.0) break;
			
			if(i%1000 == 999) {
				masterMind.getComputationUnit().saveLayer5();
			}
			
			System.out.println("run number: " + i + "  countWarningsDuplicateMoves: " + dm);
			System.out.println();
		}
		/*
		run number: 1759  countWarningsDuplicateMoves: 1

		  winrate:  1.0  average moves: 3.0
		  winrate0: 0.0625
		  winrate1: 0.125
		  winrate2: 0.1875
		  winrate3: 0.3125
		  winrate4: 0.125
		  winrate5: 0.1875		
		*/
	}
	
	
	@Test
	public void test80_MM_6_4_WinrateCheck() {
		MasterMind masterMind = new MasterMind(6, 4, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);

		masterMind.getComputationUnit().loadLayer1();
		masterMind.getComputationUnit().setLayer1InReadModus();
		
		masterMind.getComputationUnit().loadLayer2();
		masterMind.getComputationUnit().setLayer2InReadModus();

		masterMind.getComputationUnit().loadLayer3();
		masterMind.getComputationUnit().setLayer3InReadModus();

		masterMind.getComputationUnit().loadLayer4();
		masterMind.getComputationUnit().setLayer4InReadModus();
		
		masterMind.getComputationUnit().loadLayer5();
		masterMind.getComputationUnit().setLayer5InReadModus();
		
		masterMind.getComputationUnit().loadLayer6();
		masterMind.getComputationUnit().setLayer5InReadModus();
		
		double averageMoves = 0.0;
		double winrate  = 0.0;
		double winrate0 = 0.0;
		double winrate1 = 0.0;
		double winrate2 = 0.0;
		double winrate3 = 0.0;
		double winrate4 = 0.0;
		double winrate5 = 0.0;
		double winrate6 = 0.0;
		
		int dm = 0;
		
		// ArrayList<String> codeListToGuess = new ArrayList<String>(); 
		// List<String> codeListToGuess = Stream.of("0230", "0303", "0323", "0500", "1405", "2003", "2112", "2154", "2253", "2302", "2420", "3133", "3314", "3420", "3440", "3523", "4315", "4335", "4400", "5043", "5113", "5221", "5235", "5420", "5435").collect(Collectors.toList());
		
		for(int i=0; i<100000; i++) { 
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			// masterMind.playMachineVsHumanSetOfCombinationMultipleTimes(1, codeListToGuess, ComputationUnit.ALGO_Q, 1.0, true, false);
			
			int cdw = GlobalSge.countWarningsDuplicateMoves;
			if(i%10 == 0) {
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN_WITH_DUPLICATES, 0.0, false, false);
				// codeListToGuess = result.notFoundCodeList;
				// System.out.println("" + codeListToGuess.toString());

				winrate = (double)result.gamesWon / (double)result.gamesInTotal;

				averageMoves = result.average;
				winrate0 = (double)result.gamesWonInOneMove    / (double)result.gamesInTotal;
				winrate1 = (double)result.gamesWonInTwoMoves   / (double)result.gamesInTotal; 
				winrate2 = (double)result.gamesWonInThreeMoves / (double)result.gamesInTotal; 
				winrate3 = (double)result.gamesWonInFourMoves  / (double)result.gamesInTotal;
				winrate4 = (double)result.gamesWonInFiveMoves  / (double)result.gamesInTotal;
				winrate5 = (double)result.gamesWonInSixMoves   / (double)result.gamesInTotal;
				winrate6 = (double)result.gamesWonInSevenMoves / (double)result.gamesInTotal;
				
				dm = (GlobalSge.countWarningsDuplicateMoves - cdw);
			}

			System.out.println("  winrate:  " + winrate + "  average moves: " + averageMoves);
			System.out.println("  winrate0: " + winrate0);
			System.out.println("  winrate1: " + winrate1);  
			System.out.println("  winrate2: " + winrate2);  
			System.out.println("  winrate3: " + winrate3);
			System.out.println("  winrate4: " + winrate4);
			System.out.println("  winrate5: " + winrate5);
			System.out.println("  winrate6: " + winrate6);
			
			if(winrate >= 1.0) break;
			
			if(i%1000 == 999) {
				masterMind.getComputationUnit().saveLayer6();
			}
			
			System.out.println("run number: " + i + "  countWarningsDuplicateMoves: " + dm);
			System.out.println();
		}		
		masterMind.getComputationUnit().saveLayer6();
	}
	

/*

  winrate:  0.9768518518518519  average moves: 34.0
  winrate0: 7.716049382716049E-4
  winrate1: 0.010030864197530864
  winrate2: 0.06867283950617284
  winrate3: 0.30787037037037035
  winrate4: 0.44367283950617287
  winrate5: 0.14583333333333334
run number: 30645  countWarningsDuplicateMoves: 0

*/
	
	
}	