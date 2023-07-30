package org.sge.mm;

public class LearnStatisticOneGame {
	public int counterTrainGoodMove=0;
	public int counterTrainBadMove=0;
	public int successfulTrainedFromBadToGoodMove=0;
	

	public void addLearnResultOneTraining(LearnStatisticOneTraining learnResultOneTraining) {
		if(learnResultOneTraining == null) return;
		
		counterTrainGoodMove = counterTrainGoodMove + learnResultOneTraining.counterTrainGoodMove;
		counterTrainBadMove = counterTrainBadMove + learnResultOneTraining.counterTrainBadMove;
		successfulTrainedFromBadToGoodMove = successfulTrainedFromBadToGoodMove + learnResultOneTraining.successfulTrainedFromBadToGoodMove;
	}
	

	void print() {
		System.out.println("counterTrainGoodMove: " + counterTrainGoodMove);
		System.out.println("counterTrainBadMove: " + counterTrainBadMove);
		System.out.println("successfulTrainedFromBadToGoodMove: " + successfulTrainedFromBadToGoodMove);
	}
}
