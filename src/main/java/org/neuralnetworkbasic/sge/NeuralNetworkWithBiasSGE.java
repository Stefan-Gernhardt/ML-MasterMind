package org.neuralnetworkbasic.sge;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import java.util.function.DoubleUnaryOperator;

import org.neuralnetworkbasic.NeuralNetworkWithBias;
import org.neuralnetworkbasic.la.Matrix;

public class NeuralNetworkWithBiasSGE extends NeuralNetworkWithBias {
	public static final String lineSeperator = "----------";
	public static final String fileSuffix = ".txt";

	private String netName = "";

	public NeuralNetworkWithBiasSGE(String netNameParameter, int numberOfInputs, int[] layerSizes,
			DoubleUnaryOperator[] activationFunctions, DoubleUnaryOperator[] activationFunctionPrimes) {
		super(numberOfInputs, layerSizes, activationFunctions, activationFunctionPrimes);
		netName = netNameParameter;
	}
	

	public NeuralNetworkWithBiasSGE(int numberOfInputs, int[] layerSizes,
			DoubleUnaryOperator[] activationFunctions, DoubleUnaryOperator[] activationFunctionPrimes) {
		super(numberOfInputs, layerSizes, activationFunctions, activationFunctionPrimes);
		
		// "Input8Hidden8Output4"
		netName = "Input" + numberOfInputs + "Hidden" + layerSizes.length + "Output" + layerSizes[layerSizes.length-1];
	}
	

	public void save() {
		BufferedWriter writer = null;
		try
		{
		    writer = new BufferedWriter( new FileWriter(netName + fileSuffix));
		    System.out.println("save to file: " + netName + fileSuffix);
		    int countLayers = getNumberOfLayers();
		    writer.write("" + countLayers);		
		    writer.newLine();
		    
			for(int layerNumber=0; layerNumber<countLayers; layerNumber++) {
				Matrix layer = getWeights(layerNumber);

				writer.write("" + layer.getHeight());		
			    writer.newLine();

				writer.write("" + layer.getWidth());		
			    writer.newLine();

			    for(int row=0; row<layer.getHeight(); row++) {
					for(int col=0; col<layer.getWidth(); col++) {
						// System.out.print(layer.get(row, col) + ",");
					    writer.write(layer.get(row, col) + ",");
					}
					// System.out.println();
					writer.newLine();
				}
				// System.out.println(lineSeperator);
				writer.write(lineSeperator);
				writer.newLine();
			}
		}
		catch ( IOException e)
		{
		}
		finally
		{
		    try
		    {
		        if ( writer != null)
		        writer.close( );
		    }
		    catch ( IOException e)
		    {
		    }
		}			
	}
	
	
	public void load() {
		Scanner sc=null;
		try {
			sc = new Scanner(new BufferedReader(new FileReader(netName + fileSuffix)));
		    System.out.println("load from file: " + netName + fileSuffix);
			String countLayersString = sc.nextLine();
			int countLayers = Integer.parseInt(countLayersString);
			// System.out.println("Count Layers: " + countLayers);
			
			for(int layer=0; layer<countLayers; layer++) {
				Matrix layerMatrix = getWeights(layer);

				String heightString = sc.nextLine();
				int height = Integer.parseInt(heightString);
				// System.out.println("height: " + height);
				
				String widthString = sc.nextLine();
				int width = Integer.parseInt(widthString);
				// System.out.println("width: " + width);
				
				int rows = height;
				int columns = width;
				double[][] myArray = new double[rows][columns];
				for (int i = 0; i < myArray.length; i++) {
					String[] line = sc.nextLine().trim().split(",");
					for (int j = 0; j < line.length; j++) {
						double value = Double.parseDouble(line[j]);
						layerMatrix.set(i, j, value);
						// myArray[i][j] = Double.parseDouble(line[j]);
						// System.out.print("" + myArray[i][j] + ",");
					}
					// System.out.println();
				}
				
				sc.nextLine(); // read line separator
			}
		} catch (FileNotFoundException e) {
		} finally {
			if(sc != null) sc.close();
		}
	}

	
	public void printWeights() {
		System.out.println("print weights");
		for(int layer=0; layer<this.getNumberOfLayers(); layer++) {
			Matrix layerMatrix = getWeights(layer);
			System.out.println("weights: \n" + layerMatrix.toString());
			
			Matrix biasMatrix = getBiases(layer);
			System.out.println("bias: \n" + biasMatrix.toString());
		}
		
	}
	
	
	public void setRandomWeights(long seed) {
	    Random generator = new Random(seed);
	    
	    for(int layer=0; layer<getNumberOfLayers(); layer++) {
			Matrix weigthMatrix = getWeights(layer);
			int w = weigthMatrix.getWidth();
			int h = weigthMatrix.getHeight();
			
			for(int row=0; row<h; row++) {
				for(int col=0; col<w; col++) {
					double randowmWeight = (generator.nextDouble() * 2) - 1;
					weigthMatrix.set(row, col, randowmWeight);
				}
			}
			
			Matrix biases = getBiases(layer);
			for(int col=0; col<w; col++) {
				double value = (generator.nextDouble() * 2) - 1;
				biases.set(0, col, value);
			}
		}
	}		
}
