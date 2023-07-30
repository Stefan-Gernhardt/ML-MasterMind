package org.sge.mm;

public class ResultOneGame {
	public boolean codeFound = false;
	public int movesNeeded = 0;
	public LearnStatisticOneGame learnStatisticOneGame = null;
	
	public void print() {
		System.out.println();
		System.out.println("Game Result:");
		System.out.println("codeFound: " + codeFound);
		System.out.println("movesNeeded: " + movesNeeded);
		if(learnStatisticOneGame != null) learnStatisticOneGame.print();
	}
}
