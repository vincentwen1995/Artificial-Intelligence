/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;

/**
 *
 * @author Administrator
 */
public class MeanSubtraction implements DataTransform{
    Float mean;
    MeanSubtraction() {}
    @Override public void fit(List<TensorPair> data) {
        if (data.isEmpty()){
            throw new IllegalArgumentException("Empty dataset");           
        }                
        float sumMean = 0;
        for (TensorPair pair: data){
            sumMean += pair.model_input.getValues().meanNumber().floatValue();            
        }
        mean = sumMean / data.size();
    }
    @Override public void transform(List<TensorPair> data){
        for (TensorPair pair: data){
            pair.model_input.getValues().subi(mean);
        }
    }
    
}
