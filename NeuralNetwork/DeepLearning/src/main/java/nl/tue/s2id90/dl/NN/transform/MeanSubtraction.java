/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.transform;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;

/**
 *
 * @author Administrator
 */
public class MeanSubtraction implements DataTransform{
    Float mean;
    public MeanSubtraction() {}
    @Override public void fit(List<TensorPair> data) {
        if (data.isEmpty()){
            throw new IllegalArgumentException("Empty dataset");           
        }                
        System.out.println("Initializing mean subtractions...");
        // Initialize the sum of means of the inputs per data pairs
        float sumMean = 0;
        for (TensorPair pair: data){
            // Accumulate the sum of means
            sumMean += pair.model_input.getValues().meanNumber().floatValue();            
        }
        // Compute the mean over all of the inputs per data pairs in the training set
        mean = sumMean / data.size();
    }
    @Override public void transform(List<TensorPair> data){
        for (TensorPair pair: data){
            // Subtract the mean from each data pair's input
            pair.model_input.getValues().subi(mean);
        }
        System.out.println("Mean subtractions in gray channel done...");
    }
    
}
