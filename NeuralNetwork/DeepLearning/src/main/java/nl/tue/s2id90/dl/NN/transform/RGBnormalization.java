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
import org.nd4j.linalg.indexing.INDArrayIndex;
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
            sum = Nd4j.zeros(data.size(), data.get(0).model_input.getValues().size(2), data.get(0).model_input.getValues().size(3));
        }
        System.out.println("Initializing RGB normalization...");
        for (TensorPair pair: data){
//            int[] indices = {data.indexOf(pair), pair.model_input.getValues().size(2), pair.model_input.getValues().size(3)};
            INDArrayIndex[] indices = {NDArrayIndex.point(data.indexOf(pair)), NDArrayIndex.all(), NDArrayIndex.all()};
//            System.out.println(Arrays.toString(indices));            
            sum.put(indices, pair.model_input.getValues().sum(1));            
//            sum.put(indices, pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())
//                    .add(pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()))
//                    .add(pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all())));
        }
//        System.out.println("Sum of three channels in first image of Training Data (Computed): ");
//        System.out.println(data.get(0).model_input.getValues().sum(1));
//        System.out.println("Sum of three channels in first image of Training Data (Assigned): ");
//        System.out.println(sum.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()));
    }
    @Override public void transform(List<TensorPair> data){
        for (TensorPair pair: data){
            (pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()))
                    .divi(sum.get(NDArrayIndex.point(data.indexOf(pair)), NDArrayIndex.all(), NDArrayIndex.all()))
                    .muli(255.0f);                            
            (pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()))
                    .divi(sum.get(NDArrayIndex.point(data.indexOf(pair)), NDArrayIndex.all(), NDArrayIndex.all()))
                    .muli(255.0f);
            (pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all()))
                    .divi(sum.get(NDArrayIndex.point(data.indexOf(pair)), NDArrayIndex.all(), NDArrayIndex.all()))
                    .muli(255.0f);
        }
//        System.out.println("RGB channels in first image of Training Data divided by sum: ");
//        System.out.println(data.get(0).model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()).div(sum.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())));
//        System.out.println(data.get(0).model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).div(sum.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())));
//        System.out.println(data.get(0).model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all()).div(sum.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())));
        System.out.println("RGB normalization done...");
    }
}
