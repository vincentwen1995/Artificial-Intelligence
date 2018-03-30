/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.transform;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author Administrator
 */
public class MeanSubtractionRGB implements DataTransform{
    Float mean_R, mean_G, mean_B;
    public MeanSubtractionRGB() {}
    @Override public void fit(List<TensorPair> data) {
        if (data.isEmpty()){
            throw new IllegalArgumentException("Empty dataset");           
        }                
        System.out.println("Initializing mean subtractions...");
        float sumMean_R = 0, sumMean_G = 0, sumMean_B = 0;
        for (TensorPair pair: data){
            sumMean_R += pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()).meanNumber().floatValue();
            sumMean_G += pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).meanNumber().floatValue();
            sumMean_B += pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all()).meanNumber().floatValue();
        }
        mean_R = sumMean_R / data.size();
        mean_G = sumMean_G / data.size();
        mean_B = sumMean_B / data.size();
    }
    @Override public void transform(List<TensorPair> data){
        for (TensorPair pair: data){
            pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()).subi(mean_R);
            pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).subi(mean_G);
            pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all()).subi(mean_B);
        }
        System.out.println("Mean subtractions in RGB channels done...");
    }
    
}
