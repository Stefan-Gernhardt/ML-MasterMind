package main;

import org.neuralnetworkbasic.ActivationFunction;
import org.sge.mm.ComputationUnit;
import org.sge.mm.GlobalSge;
import org.sge.mm.MasterMind;
import org.sge.mm.ResultAllGames;

public class Main {
	

	public static void main(String[] args) {
		System.out.println("Master Mind");
		
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
		
		
		for(int i=0; i<100000; i++) { 
			masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_Q, 1.0, true, false);
			
			int cdw = GlobalSge.countWarningsDuplicateMoves;
			if(i%10 == 0) {
				ResultAllGames result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_NN_WITH_DUPLICATES, 0.0, false, false);

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
	}
	
}
