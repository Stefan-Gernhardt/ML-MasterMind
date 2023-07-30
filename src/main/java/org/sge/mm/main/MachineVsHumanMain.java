package org.sge.mm.main;

import org.sge.mm.ComputationUnit;
import org.sge.mm.MasterMind;
import org.sge.mm.ResultAllGames;

public class MachineVsHumanMain {

	public static void main(String[] args) {
		// MasterMind masterMind = new MasterMind(4, 2, false);
		// MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);
		// MasterMind masterMind = new MasterMind(6, 4, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);
		
		// masterMind.playMachineVsHuman();
		// masterMind.playMachineVsHuman("32");
		
		// long start = System.currentTimeMillis();
		// ResultAllGameCombinations result = masterMind.playMachineVsHumanAllCombinations(ComputationUnit.ALGO_EXCLUDE, false, false);
		// result.print();
		// long end = System.currentTimeMillis();
		// System.out.println("duration: " + (end - start) + " ms"); // 2465ms
	
		// learn one specific code
		MasterMind masterMind = new MasterMind(4, 2, ComputationUnit.SET_WEIGHTS_WITH_PSEUDO_RANDOM_VALUES, 0, 0);
		long start = System.currentTimeMillis();
		ResultAllGames result = masterMind.learnOneSpecificCode("12");
		result.print();
		long end = System.currentTimeMillis();
		System.out.println("duration: " + (end - start) + " ms"); // 2465ms
		
	}

}
