package org.sge.mm;

public class GlobalSge {
	
	public static int countWarningsDuplicateMoves = 0;
	public static int countWarningsImpossibleMoveTried = 0;
	public static int countWarningsUncompleteMoves = 0;
	public static int countWarningsNoFreePlaceFound = 0;
	
	public static void printErrorWarningReport() {
		System.out.println("countWarningsDuplicateMoves: " + countWarningsDuplicateMoves);
		System.out.println("countWarningsImpossibleMoveTried: " + countWarningsImpossibleMoveTried);
		System.out.println("countWarningsUncompleteMoves: " + countWarningsUncompleteMoves);
		System.out.println("countWarningsNoFreePlaceFound: " + countWarningsNoFreePlaceFound);
	}

}
