package org.sge.mm;

public class ResultAllGames {
	public int gamesInTotal = 0;
	public int gamesWon     = 0;
	public int gamesLost    = 0;
	public int sumMoves     = 0;
	public int maxMoves     = 0;
	public double average   = 0;
	public LearnStatisticAllGames learnStatisticAllGames = null;
	
	public int gamesWonInOneMove        = 0;
	public int gamesWonInTwoMoves       = 0;
	public int gamesWonInThreeMoves     = 0;
	public int gamesWonInFourMoves      = 0;
	public int gamesWonInFiveMoves      = 0;
	

	public void print() {
		System.out.println();		
		System.out.println("gamesInTotal:  " + gamesInTotal);
		System.out.println("gamesWon:      " + gamesWon + "  gamesWonInOneMove: " + gamesWonInOneMove + "  gamesWonInTwoMoves: " + gamesWonInTwoMoves + "  gamesWonInThreeMoves: " + gamesWonInThreeMoves + "  gamesWonInFourMoves: " + gamesWonInFourMoves + "  gamesWonInFiveMoves: " + gamesWonInFiveMoves);
		System.out.println("gamesLost:     " + gamesLost);
		System.out.println("win rate:      " + (1.0*gamesWon) / (1.0*gamesInTotal));
		System.out.println("sum moves:     " + sumMoves);
		System.out.println("max moves:     " + maxMoves);
		System.out.println("average moves: " + (1.0*sumMoves) / (1.0*gamesInTotal));
		// if(learnStatisticAllGames != null) learnStatisticAllGames.print();
		System.out.println();		
	}
	

	public ResultAllGames add(ResultAllGames result) {
		this.gamesInTotal = this.gamesInTotal + result.gamesInTotal; 
		this.gamesWon     = this.gamesWon     + result.gamesWon; 
		this.gamesLost    = this.gamesLost    + result.gamesLost; 
		this.sumMoves     = this.sumMoves     + result.sumMoves; 

		this.gamesWonInOneMove    = this.gamesWonInOneMove    + result.gamesWonInOneMove;
		this.gamesWonInTwoMoves   = this.gamesWonInTwoMoves   + result.gamesWonInTwoMoves;
		this.gamesWonInThreeMoves = this.gamesWonInThreeMoves + result.gamesWonInThreeMoves;
		this.gamesWonInFourMoves  = this.gamesWonInFourMoves  + result.gamesWonInFourMoves;
		this.gamesWonInFiveMoves  = this.gamesWonInFiveMoves  + result.gamesWonInFiveMoves;
		
		return this;
	}
}
