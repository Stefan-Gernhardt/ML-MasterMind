package org.sge.mm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Board {
	private String codeTofind = "";
	private List<Move> listOfMoves = null;
	 

	private static int countColors=0;
	private static int countDigits=0;

	private int turnNumber = 0;
	public static final int MAX_COUNT_GUESSES = 5-3;
	
	public int getCountColors() {
		return countColors;
	}
	
	
	static public int getCountDigits() {
		return countDigits;
	}

	
	public int getTurnNumber() {
		return turnNumber;
	}

	
	public List<Move> getListOfMoves() {
		return listOfMoves;
	}
	
	
	public Move getLastMove() {
		if(turnNumber < 1) return null;
		
		return listOfMoves.get(turnNumber-1);
	}


	public Move getSecondLastMove() {
		if(turnNumber < 2) return null;
		
		return listOfMoves.get(turnNumber-2);
	}


	public Board(int countColors, int countDigits) {
		this.countColors = countColors;
		this.countDigits = countDigits;
		this.listOfMoves = new ArrayList<Move>();
	}
		
	
	public boolean validCode(String code) {
		if(code == null) return false;
		if(code.length() != countDigits) return false;
		
		for(int d=0; d<code.length(); d++) {
			int digit = Integer.parseInt("" + code.charAt(d));
			if(digit < 0) return false;
			if(digit >= countColors) return false;
		}
		
		return true;
	}
	
	
	public void setCodeToFind(String code) {
		if(!validCode(code)) {
			throw new RuntimeException("trying to set an invalid code. Number of digits: " + countDigits + "  code: " + code);
		}
		
		codeTofind = code;
	}

	
	public String getCodeToFind() {
		return codeTofind;
	}

	
	public boolean isDuplicateMove(Guess guess) {
		for(int i=0; i<listOfMoves.size(); i++) {
			Move move = listOfMoves.get(i);
			if(move.guess.code.contentEquals(guess.code)) return true;
			// if(move.guess.index ==  guess.index) return true;
		}
		
		return false;
	}

	
	public boolean isMoveComplete(Guess guess) {
		for(int i=0; i<listOfMoves.size(); i++) {
			Move move = listOfMoves.get(i);
			if(move.guess.code.contentEquals("")) return false;
			// if(move.guess.index == -1) return false;
		}
		
		return true;
	}

	
	public Move setGuessOnBoard(Guess guess, boolean verbose) {
		turnNumber++;
		
		Move move = new Move();
		move.guess = guess;
		move.rating = getRating(guess.code);
		
		if(!isMoveComplete(guess)) {
			if(verbose) System.out.println("Warning: Uncomplete Move");
			GlobalSge.countWarningsUncompleteMoves++;
		}
		
		if(isDuplicateMove(guess)) {
			if(verbose) System.out.println("Warning: Duplicate Move " + guess.code);
			GlobalSge.countWarningsDuplicateMoves++;
		}
		
		listOfMoves.add(move);
		return move;
	}

	
	public boolean codeFound() {
		if(turnNumber == 0) return false;
		
		return codeTofind.contentEquals(listOfMoves.get(turnNumber-1).guess.code);
	}


	private static boolean containsColor(int[] array, int c) {
		for(int i=0; i<array.length; i++) {
			if(array[i] == c) return true;
		}

		return false;
	}
	
	
	public static int getMaxScore() {
		return countDigits * countColors;
	}
	
	
	static public double getRatingScore(int countBlack, int countWhite) {
		double maxScore = getMaxScore();
		return (1.0*countBlack*countColors  + countWhite*1.0) / maxScore;
	}
	
	
	static public Rating getRating(String codeGuess, String codeToFind) {
		Rating rating = new Rating();
		
		rating.countBlack = 0;
		rating.countWhite = 0;
		
		int[] codeGuessInt  = new int[codeGuess.length()];
		int[] codeToFindInt = new int[codeToFind.length()];
		
		for(int i=0; i<codeGuess.length(); i++) {
			codeGuessInt[i]  = Integer.parseInt(codeGuess.substring(i, i+1));
			codeToFindInt[i] = Integer.parseInt(codeToFind.substring(i, i+1));
		}
		
		for(int d=0; d<codeGuess.length(); d++) {
			if(codeToFindInt[d] == codeGuessInt[d]) {
				rating.countBlack++;
				codeToFindInt[d]  = -1;
				codeGuessInt [d]  = -2;
			}
		}

		for(int d=0; d<codeGuess.length(); d++) {
			// System.out.println("codeToFindInt: " + Arrays.toString(codeToFindInt) + "   d: " + codeGuessInt[d]);
			if(containsColor(codeToFindInt, codeGuessInt[d])) {
				rating.countWhite++;
				for(int e=0; e<codeToFindInt.length; e++) {
					if(codeToFindInt[e] == codeGuessInt[d]) codeToFindInt[e] = -3; 
				}
				codeGuessInt [d]  = -4;
			}
		}
		
		
		rating.score = getRatingScore(rating.countBlack, rating.countWhite);
		
		return rating;

	}
	

	public Rating getRating(String codeGuess) {
		return getRating(codeGuess, codeTofind);
	}
	

	public Rating getRating(int i) {
		return listOfMoves.get(i).rating;
	}


	public boolean maxAttemptsReached() {
		return turnNumber >= Board.MAX_COUNT_GUESSES;
	}

	
	public int getNumberPossibleCombinations() {
		int number = 1;
		for(int i=0; i<countDigits; i++) {
			number = number * countColors;
		}
		
		return number;
	}


	public void reset() {
		codeTofind = "";
		listOfMoves = new ArrayList<Move>();
		turnNumber = 0;
	}	
}
