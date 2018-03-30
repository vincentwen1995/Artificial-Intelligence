/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.optimizer.update;

import java.util.function.Supplier;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Administrator
 */
public class L2Decay implements UpdateFunction{
    float decay; 
    UpdateFunction f;
    INDArray update;
    public L2Decay(Supplier<UpdateFunction> supplier, float decay) {
        this.decay = decay;
        this.f = supplier.get();        
    }
    
    @Override
    public void update(INDArray value, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        // Only apply L2Decay to weights matrix
        if (!isBias) {
            // Add the penalized weights to the gradient: gradient <- gradient + decay * value(weights)
            gradient.addi(value.mul(decay));
        }
        f.update(value, isBias, learningRate, batchSize, gradient);
    }
}
