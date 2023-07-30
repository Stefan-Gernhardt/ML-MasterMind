package org.sge.mm;

public class LearnStatisticAllGames {
	public int counterTrainGoodMove=0;
	public int counterTrainBadMove=0;
	public int successfulTrainedFromBadToGoodMove=0;
	

	public void addLearnResultOneGame(LearnStatisticOneGame learnResultOneGame) {
		counterTrainGoodMove = counterTrainGoodMove + learnResultOneGame.counterTrainGoodMove;
		counterTrainBadMove = counterTrainBadMove + learnResultOneGame.counterTrainBadMove;
		successfulTrainedFromBadToGoodMove = successfulTrainedFromBadToGoodMove + learnResultOneGame.successfulTrainedFromBadToGoodMove;
	}

	void print() {
		System.out.println("counterTrainGoodMove: " + counterTrainGoodMove);
		System.out.println("counterTrainBadMove: " + counterTrainBadMove);
		System.out.println("successfulTrainedFromBadToGoodMove: " + successfulTrainedFromBadToGoodMove);
	}
}
