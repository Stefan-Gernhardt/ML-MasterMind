package org.neuralnetwork.org.sge.math;

import org.junit.jupiter.api.Test;
import org.sge.math.MathSge;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MathSgeTest {

	@Test
	public void test1() {
		assertEquals(MathSge.convertDecTo(3, 5), "12");
	}

	
	@Test
	public void test2() {
		assertEquals(MathSge.convertDecTo(2, 4), "100");
	}
	

	@Test
	public void test3() {
		assertEquals(MathSge.convertDecTo(5, 5), "10");
	}
	
	
	@Test
	public void test4() {
		assertEquals(MathSge.convertDecTo(3, 5, 4), "0012");
	}

	
	@Test
	public void test5() {
		assertEquals(MathSge.convertDecTo(2, 4, 1), "100");
	}
	

	@Test
	public void test6() {
		assertEquals(MathSge.convertDecTo(5, 5, 2), "10");
	}
}
