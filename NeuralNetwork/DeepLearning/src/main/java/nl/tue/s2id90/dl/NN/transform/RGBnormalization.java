/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.transform;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author Administrator
 */
public class RGBnormalization implements DataTransform{
    INDArray sum;
    public RGBnormalization() {}
    @Override public void fit(List<TensorPair> data) {
        if (data.isEmpty()){
            throw new IllegalArgumentException("Empty dataset");           
        }
        if (sum == null) {
            sum = Nd4j.zeros(data.size(), data.get(0).model_input.getValues().size(1), data.get(0).model_input.getValues().size(2));
        }
        for (TensorPair pair: data){
            int[] indices = {data.indexOf(pair), pair.model_input.getValues().size(1), pair.model_input.getValues().size(2)};
            sum.put(indices, pair.model_input.getValues().sum(1));            
        }        
    }
    @Override public void transform(List<TensorPair> data){
        for (TensorPair pair: data){
            for (int i = 0; i < 3; i++) {
                pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all())
                        .divi(sum.get(NDArrayIndex.point(data.indexOf(pair)), NDArrayIndex.all(), NDArrayIndex.all()))
                        .muli(255.0);                
            }
        }
    }
}
