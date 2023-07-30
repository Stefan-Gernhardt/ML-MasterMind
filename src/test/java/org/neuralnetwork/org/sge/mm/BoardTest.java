package org.neuralnetwork.org.sge.mm;

import static org.junit.Assert.assertEquals;

import org.sge.math.MathSge;
import org.sge.mm.Board;
import org.sge.mm.ComputationUnit;
import org.sge.mm.Guess;
import org.sge.mm.MasterMind;
import org.sge.mm.Rating;
import org.junit.jupiter.api.Test;
import org.neuralnetworkbasic.ActivationFunction;

public class BoardTest {

	@Test
	public void ratingTest1() {
		Board b = new Board(4, 2);
		b.setCodeToFind("32");
		
		Rating r1 = b.getRating("32");
		assertEquals(r1.countWhite, 0);
		assertEquals(r1.countBlack, 2);
		
		Rating r2 = b.getRating("00");
		assertEquals(r2.countWhite, 0);
		assertEquals(r2.countBlack, 0);
		
		Rating r3 = b.getRating("31");
		assertEquals(r3.countWhite, 0);
		assertEquals(r3.countBlack, 1);
		
		Rating r4 = b.getRating("12");
		assertEquals(r4.countWhite, 0);
		assertEquals(r4.countBlack, 1);
		
		Rating r5 = b.getRating("13");
		assertEquals(r5.countWhite, 1);
		assertEquals(r5.countBlack, 0);
		
		Rating r6 = b.getRating("21");
		assertEquals(r6.countWhite, 1);
		assertEquals(r6.countBlack, 0);
	}
	
	
	@Test
	public void ratingTest2() {
		Board b = new Board(6, 4);
		b.setCodeToFind("1134");
		
		Rating r1 = b.getRating("1111");
		assertEquals(r1.countWhite, 0);
		assertEquals(r1.countBlack, 2);
		
		Rating r2 = b.getRating("2211");
		assertEquals(r2.countWhite, 1);
		assertEquals(r2.countBlack, 0);
		
		Rating r3 = b.getRating("2233");
		assertEquals(r3.countWhite, 0);
		assertEquals(r3.countBlack, 1);
	}
	
	
	@Test
	public void ratingTest3() {
		Board b = new Board(6, 4);
		b.setCodeToFind("1232");
		
		Rating r1 = b.getRating("1243");
		assertEquals(r1.countWhite, 1);
		assertEquals(r1.countBlack, 2);
		
		Rating r2 = b.getRating("2133");
		assertEquals(r2.countWhite, 2);
		assertEquals(r2.countBlack, 1);
		
		Rating r3 = b.getRating("1155");
		assertEquals(r3.countWhite, 0);
		assertEquals(r3.countBlack, 1);
	}
	
	
	@Test
	public void ratingTest4() {
		Board b = new Board(6, 4);
		b.setCodeToFind("1112");
		
		Rating r1 = b.getRating("1444");
		assertEquals(r1.countWhite, 0);
		assertEquals(r1.countBlack, 1);
		
		Rating r2 = b.getRating("1234");
		assertEquals(r2.countWhite, 1);
		assertEquals(r2.countBlack, 1);
		
		Rating r3 = b.getRating("1222");
		assertEquals(r3.countWhite, 0);
		assertEquals(r3.countBlack, 2);
		
		Rating r4 = b.getRating("2111");
		assertEquals(r4.countWhite, 2);
		assertEquals(r4.countBlack, 2);
		
		Rating r5 = b.getRating("0001");
		assertEquals(r5.countWhite, 1);
		assertEquals(r5.countBlack, 0);
	}
	
	
	@Test
	public void ratingTest5() {
		Board b = new Board(6, 4);
		b.setCodeToFind("1234");
		
		Rating r1 = b.getRating("1222");
		assertEquals(r1.countWhite, 0);
		assertEquals(r1.countBlack, 2);
		
		Rating r2 = b.getRating("0220");
		assertEquals(r2.countWhite, 0);
		assertEquals(r2.countBlack, 1);
	}
	
	
	@Test
	public void ratingTest6() {
		Board b = new Board(6, 4);
		b.setCodeToFind        ("1122");
		
		Rating r1 = b.getRating("1112");
		assertEquals(r1.countWhite, 0);
		assertEquals(r1.countBlack, 3);
	}
	
	
	@Test
	public void ratingTest7() {
		Board b = new Board(6, 4);
		b.setCodeToFind        ("1231");
		
		Rating r1 = b.getRating("2021");
		assertEquals(r1.countWhite, 1);
		assertEquals(r1.countBlack, 1);
		
		
		Rating r2 = b.getRating("1102");
		assertEquals(r2.countWhite, 2);
		assertEquals(r2.countBlack, 1);
	}
	
	
	@Test
	public void ratingTest8() {
		Board b = new Board(6, 4);
		b.setCodeToFind        ("1212");
		
		Rating r1 = b.getRating("2121");
		assertEquals(r1.countWhite, 2);
		assertEquals(r1.countBlack, 0);
		
		Rating r2 = b.getRating("2020");
		assertEquals(r2.countWhite, 1);
		assertEquals(r2.countBlack, 0);
		
		Rating r3 = b.getRating("2020");
		assertEquals(r3.countWhite, 1);
		assertEquals(r3.countBlack, 0);
		
		Rating r4 = b.getRating("2200");
		assertEquals(r4.countWhite, 1);
		assertEquals(r4.countBlack, 1);
		
		Rating r5 = b.getRating("0200");
		assertEquals(r5.countWhite, 0);
		assertEquals(r5.countBlack, 1);
	}	

	
	@Test
	public void computeProbabilities() {
		MasterMind masterMind = new MasterMind(4, 2, ActivationFunction.SIGMOID, ComputationUnit.SET_RANDOMVALUES_FOR_WEIGHTS_SET_BY_NN, 0, 0);
		ComputationUnit computationUnit = masterMind.getComputationUnit();
		Board b = masterMind.getBoard();
		
		
		for(int i=0; i<computationUnit.getCountOutputNeurons(); i++) {
			// System.out.println("code: " + MathSge.convertDecTo(computationUnit.getCountColors(), i, computationUnit.getCountDigits()));
			System.out.print("" + MathSge.convertDecTo(computationUnit.getCountColors(), i, computationUnit.getCountDigits()) + "  ");
		}
		System.out.println();
		
		
		for(int i=0; i<computationUnit.getCountOutputNeurons(); i++) {
			b.reset();
			b.setCodeToFind(MathSge.convertDecTo(computationUnit.getCountColors(), i, computationUnit.getCountDigits()));
			
			Guess guess = new Guess();
			guess.code  = "01";
			
			b.setGuessOnBoard(guess, true);
			Rating r = b.getRating(0);
			int black = r.countBlack;
			int white = r.countWhite;
			
			if((black == 0) && (white == 2)) {
				System.out.print("y   ");
			}
			else {
				System.out.print("n   ");
			}
		}
	}
}
	