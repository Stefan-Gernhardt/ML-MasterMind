package org.sge.mm;

import java.util.ArrayList;

import org.neuralnetworkbasic.ActivationFunction;
import org.sge.math.MathSge;

public class MasterMind {
	private ComputationUnit computationUnit = null;
	Board board = null;
	
	
	public Board getBoard() {
		return board;
	}


	public void setBoard(Board board) {
		this.board = board;
	}


	public MasterMind(int countColors, int countDigits, int mode, double constantWeight, long seed) {
		this(countColors, countDigits, ActivationFunction.SIGMOID, mode, constantWeight, seed);
	}
	

	public MasterMind(int countColors, int countDigits, ActivationFunction activationFunction, int mode, double constantWeight, long seed) {
		computationUnit = new ComputationUnit(countColors, countDigits, mode, constantWeight, seed);
		board = new Board(countColors, countDigits);
		
		computationUnit.setFirstMoveFixNatural();  
	}
	

	public void playMachineVsHuman(boolean verbose) {
		System.out.println("Write your combination down");
		
		Guess guess = computationUnit.getGuessAlgoNN(board.getListOfMoves(), board.getTurnNumber(), verbose);
		if(verbose) System.out.println(guess);
	}
	

	public Guess getGuess(int moveNumber, int algo, double explorationRate, boolean learn, boolean verbose) {
		if(board.getListOfMoves().isEmpty()) return computationUnit.getFirstMove(verbose);

		if(algo == ComputationUnit.ALGO_RANDOM) { 
			return computationUnit.getGuessRandom(board.getListOfMoves(), moveNumber, verbose);
		}
		
		if(algo == ComputationUnit.ALGO_EXCLUDE) { 
			return computationUnit.getGuessAlgoExclude(board.getListOfMoves(), verbose);
		}
		
		if(algo == ComputationUnit.ALGO_Q) { 
			return computationUnit.getGuessAlgoQ(board.getListOfMoves(), moveNumber, explorationRate, verbose);
		}
		
		return computationUnit.getGuessAlgoNN(board.getListOfMoves(), moveNumber, verbose);
	}
	
	
	private void playMachineVsHumanLearnRL(String code, boolean verbose) {
		board.reset();
		board.setCodeToFind(code);

		ResultOneGame gameResult = new ResultOneGame();
		LearnStatisticOneGame learnStatisticOneGame = new LearnStatisticOneGame();
		gameResult.learnStatisticOneGame = learnStatisticOneGame;
		
		while(!board.codeFound() && !board.maxAttemptsReached()) {
			if(verbose) System.out.println("---------------------------------------------------------------------------------------");
			if(verbose) System.out.println("turn number: " + board.getTurnNumber());
			
			Guess guess = computationUnit.getGuessAlgoRLTraining(board.getListOfMoves(), board.getTurnNumber(), verbose);
			Move move = board.setGuessOnBoard(guess, verbose);

			if(verbose) System.out.println("code to find: " + code + "  guess: " + guess + "  rating black: " + move.rating.countBlack + "  white: " + move.rating.countWhite + "  score: " + move.rating.score);
		}
	}

	
	public ResultOneGame playMachineVsHuman(int algo, double explorationRate, String code, boolean learn, boolean verbose) {
		if(verbose) System.out.println("Code to find: " + code + "  ");
		board.reset();
		board.setCodeToFind(code);

		ResultOneGame gameResult = new ResultOneGame();
		LearnStatisticOneGame learnStatisticOneGame = new LearnStatisticOneGame();
		gameResult.learnStatisticOneGame = learnStatisticOneGame;
		
		while(!board.codeFound() && !board.maxAttemptsReached()) {
			if(verbose) System.out.println("---------------------------------------------------------------------------------------");
			if(verbose) System.out.println("turn number: " + board.getTurnNumber());
			
			Guess guess = getGuess(board.getTurnNumber(), algo, explorationRate, learn, verbose);
			Move move = board.setGuessOnBoard(guess, verbose);

			if(verbose) System.out.println("code to find: " + code + "  guess: " + guess.code + "  rating black: " + move.rating.countBlack + "  white: " + move.rating.countWhite + "  score: " + move.rating.score);
			if(learn) {
				if(algo == ComputationUnit.ALGO_NN) { 
					learnStatisticOneGame.addLearnResultOneTraining(computationUnit.learnNN(board, verbose));
				}
				
				if(algo == ComputationUnit.ALGO_Q) {
					computationUnit.trainAlgoQ(board, board.getTurnNumber(), verbose);
				}
			}
		}
		gameResult.codeFound = board.codeFound();
		
		if(board.codeFound()) {
			if(verbose) System.out.println("*** Code found *** in moves: " + board.getTurnNumber());
			gameResult.movesNeeded = board.getTurnNumber();
		}
		else {
			if(verbose) System.out.println("*** Code not found *** after moves: " + board.getTurnNumber());
			gameResult.movesNeeded = board.getNumberPossibleCombinations();
		}
		
		
		return gameResult;
	}


	public ComputationUnit getComputationUnit() {
		return computationUnit;
	}


	public ResultAllGames playMachineVsHumanAllCombinations(int algo, double explorationRate, boolean withLearning, boolean verbose) {
		ResultAllGames resultAllGameCombinations = new ResultAllGames();
		resultAllGameCombinations.learnStatisticAllGames = new LearnStatisticAllGames();
		
		int maxMoves = -1;
		int sumCountMoves = 0;
		int sumGamesWithCodeFound = 0;
		int sumGamesWithCodeNotFound = 0;
		for(int combination=0; combination<board.getNumberPossibleCombinations(); combination++) {
			if(verbose) System.out.println("********************************************************************");
			if(verbose) System.out.print("" + combination + "  ");
			ResultOneGame gameResult = playMachineVsHuman(algo, explorationRate, MathSge.convertDecTo(board.getCountColors(), combination, board.getCountDigits()), withLearning, verbose);
			resultAllGameCombinations.learnStatisticAllGames.addLearnResultOneGame(gameResult.learnStatisticOneGame);
			if(verbose) gameResult.print(); 
			
			if(gameResult.movesNeeded > maxMoves) maxMoves = gameResult.movesNeeded;
			
			if(gameResult.codeFound) {
				sumCountMoves = sumCountMoves + gameResult.movesNeeded;
				sumGamesWithCodeFound++;
			}
			else {
				sumCountMoves = sumCountMoves + board.getNumberPossibleCombinations(); 
				sumGamesWithCodeNotFound++;
			}
			
			if(gameResult.movesNeeded == 1) resultAllGameCombinations.gamesWonInOneMove++;
			if(gameResult.movesNeeded == 2) resultAllGameCombinations.gamesWonInTwoMoves++;
			if(gameResult.movesNeeded == 3) resultAllGameCombinations.gamesWonInThreeMoves++;
		}
		
		double averageCountMoves = sumCountMoves / board.getNumberPossibleCombinations(); 
		
		if(verbose) System.out.println("Number of games with code found: " + sumGamesWithCodeFound);
		if(verbose) System.out.println("Number of games with code not found: " + sumGamesWithCodeNotFound);
		
		resultAllGameCombinations.gamesInTotal = board.getNumberPossibleCombinations();
		resultAllGameCombinations.gamesWon 	   = sumGamesWithCodeFound;
		resultAllGameCombinations.gamesLost    = sumGamesWithCodeNotFound;
		resultAllGameCombinations.average      = averageCountMoves; 
		resultAllGameCombinations.maxMoves     = maxMoves; 
		
		return resultAllGameCombinations;
	}
	

	public ResultAllGames playMachineVsHumanOneFixCombinationMultipleTimes(int lotsize, String codeToGuess, int algo, double explorationRate, boolean withLearning, boolean verbose) {
		ResultAllGames resultAllGames = new ResultAllGames();
		resultAllGames.learnStatisticAllGames = new LearnStatisticAllGames();
		
		int maxMoves = -1;
		int sumCountMoves = 0;
		int sumGamesWithCodeFound = 0;
		int sumGamesWithCodeNotFound = 0;
		for(int i=0; i<lotsize; i++) {
			if(verbose) System.out.println("********************************************************************");
			if(verbose) System.out.print("round: " + i + "  ");
			ResultOneGame gameResult = playMachineVsHuman(algo, explorationRate, codeToGuess, withLearning, verbose);
			resultAllGames.learnStatisticAllGames.addLearnResultOneGame(gameResult.learnStatisticOneGame);
			if(verbose) gameResult.print(); 
			if(gameResult.movesNeeded > maxMoves) maxMoves = gameResult.movesNeeded;
			if(gameResult.codeFound) {
				sumCountMoves = sumCountMoves + gameResult.movesNeeded;
				sumGamesWithCodeFound++;
			}
			else {
				sumCountMoves = sumCountMoves + board.getNumberPossibleCombinations(); 
				sumGamesWithCodeNotFound++;
			}
		}
		
		double averageCountMoves = 1.0*sumCountMoves / 1.0*lotsize; 
		
		if(verbose) System.out.println("Number of games with code found: " + sumGamesWithCodeFound);
		if(verbose) System.out.println("Number of games with code not found: " + sumGamesWithCodeNotFound);
		
		resultAllGames.gamesInTotal = lotsize;
		resultAllGames.gamesWon 	= sumGamesWithCodeFound;
		resultAllGames.gamesLost    = sumGamesWithCodeNotFound;
		resultAllGames.average      = averageCountMoves; 
		resultAllGames.maxMoves     = maxMoves; 
		
		return resultAllGames;
	}


	public ResultAllGames playMachineVsHumanSetOfCombinationMultipleTimes(int lotsize, ArrayList<String> codeListToGuess, int algo, double explorationRate, boolean withLearning, boolean verbose) {
		ResultAllGames resultAllGames = new ResultAllGames();
		resultAllGames.learnStatisticAllGames = new LearnStatisticAllGames();
		
		int maxMoves = -1;
		int sumCountMoves = 0;
		int sumGamesWithCodeFound = 0;
		int sumGamesWithCodeNotFound = 0;
		for(int i=0; i<lotsize; i++) {
			for(int j=0; j<codeListToGuess.size(); j++) {
				if(verbose) System.out.println("********************************************************************");
				if(verbose) System.out.print("round: " + i + "  ");
				ResultOneGame gameResult = playMachineVsHuman(algo, explorationRate, codeListToGuess.get(j), withLearning, verbose);
				resultAllGames.learnStatisticAllGames.addLearnResultOneGame(gameResult.learnStatisticOneGame);
				if(verbose) gameResult.print(); 
				if(gameResult.movesNeeded > maxMoves) maxMoves = gameResult.movesNeeded;
				if(gameResult.codeFound) {
					sumCountMoves = sumCountMoves + gameResult.movesNeeded;
					sumGamesWithCodeFound++;
				}
				else {
					sumCountMoves = sumCountMoves + board.getNumberPossibleCombinations(); 
					sumGamesWithCodeNotFound++;
				}
			}
		}
		
		double averageCountMoves = 1.0*sumCountMoves / (1.0*lotsize*codeListToGuess.size()); 
		
		if(verbose) System.out.println("Number of games with code found: " + sumGamesWithCodeFound);
		if(verbose) System.out.println("Number of games with code not found: " + sumGamesWithCodeNotFound);
		
		resultAllGames.gamesInTotal = lotsize*codeListToGuess.size();
		resultAllGames.gamesWon 	= sumGamesWithCodeFound;
		resultAllGames.gamesLost    = sumGamesWithCodeNotFound;
		resultAllGames.sumMoves     = sumCountMoves; 
		resultAllGames.average      = averageCountMoves; 
		resultAllGames.maxMoves     = maxMoves; 
		
		return resultAllGames;
	}


	public void learn() {
		ResultAllGames result = playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, 0, false, false);
		result.print();
		
		playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN, 0, true, false);
		
	}

	
	private ResultAllGames learnMachineVsHumanOneCombinations(String code, int algo, double explorationRate, boolean withLearning, boolean verbose) {
		int countTrainingGames = 1;
		int maxMoves = -1;
		int sumCountMoves = 0;
		int sumGamesWithCodeFound = 0;
		int sumGamesWithCodeNotFound = 0;
		for(int gameNumber=0; gameNumber<countTrainingGames; gameNumber++) {
			if(verbose) System.out.println("********************************************************************");
			if(verbose) System.out.print("" + gameNumber + "  ");
			ResultOneGame gameResult = playMachineVsHuman(algo, explorationRate, code, withLearning, verbose);
			if(gameResult.movesNeeded > maxMoves) maxMoves = gameResult.movesNeeded;
			if(gameResult.codeFound) {
				sumCountMoves = sumCountMoves + gameResult.movesNeeded;
				sumGamesWithCodeFound++;
			}
			else {
				sumCountMoves = sumCountMoves + board.getNumberPossibleCombinations(); 
				sumGamesWithCodeNotFound++;
			}
			if(verbose) System.out.println("gameResult.movesNeeded: " + gameResult.movesNeeded);
		}
		
		double averageCountMoves = (double)sumCountMoves / (double)countTrainingGames; 
		
		if(verbose) System.out.println("Number of games with code found: " + sumGamesWithCodeFound);
		if(verbose) System.out.println("Number of games with code not found: " + sumGamesWithCodeNotFound);
		
		ResultAllGames resultAllGameCombinations = new ResultAllGames();
		
		resultAllGameCombinations.gamesInTotal = countTrainingGames;
		resultAllGameCombinations.gamesWon 	   = sumGamesWithCodeFound;
		resultAllGameCombinations.gamesLost    = sumGamesWithCodeNotFound;
		resultAllGameCombinations.average      = averageCountMoves; 
		resultAllGameCombinations.maxMoves     = maxMoves; 
		
		return resultAllGameCombinations;
	}
	
	
	public ResultAllGames learnOneSpecificCode(String code) {
		return learnMachineVsHumanOneCombinations(code, ComputationUnit.ALGO_NN, 0, true, true);
	}

	

}
