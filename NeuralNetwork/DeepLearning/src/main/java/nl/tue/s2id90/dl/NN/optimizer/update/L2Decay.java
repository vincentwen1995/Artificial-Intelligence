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
    public L2Decay(Supplier<UpdateFunction> supplier, float decay) {
        this.decay = decay;
        this.f = supplier.get();
    }
    
    @Override
    public void update(INDArray value, boolean isBias, float learningRate, int batchSize, INDArray gradient) {                
        // First apply GD with Momentum
        f.update(value, isBias, learningRate, batchSize, gradient);
        // Only apply L2Decay to weights matrix
        if (!isBias) {
            float factor = -(learningRate/batchSize);
            // Scale down the weights matrix by (1 - decay) * factor
            Nd4j.getBlasWrapper().level1().scal( value.length(), (1 - decay) * factor, value);
            gradient.assign(0);
        }
    }
}
