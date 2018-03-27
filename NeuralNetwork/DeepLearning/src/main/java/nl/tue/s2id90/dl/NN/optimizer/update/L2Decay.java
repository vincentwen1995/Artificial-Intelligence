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
        // Make a copy of gradient as it is reset after gradient descent momentum update
//        INDArray value_copy = Nd4j.zeros(value.shape());
//        Nd4j.getBlasWrapper().level1().copy(value, value_copy);
        // First apply GD (with Momentum ?) NB: To cope with GD_Momentum, the update from GD_Momentum needs to subtract(add) decay * factor * value(original)
//        f.update(value, isBias, learningRate, batchSize, gradient);     // value(updated) <- value(original) + factor * gradient
        
        // Only apply L2Decay to weights matrix
        if (!isBias) {
//            float factor = -(learningRate/batchSize);
            
            // Subtract the decay from the updated value ()
//            Nd4j.getBlasWrapper().level1().axpy( value.length(), decay * factor, value_copy, value );   // value <- value(updated) + factor * decay * value(original)
//            gradient.assign(0);
            // Update the gradient with the L2 penalty: gradient = gradient + decay * value
//            Nd4j.getBlasWrapper().level1().axpy( gradient.length(), decay, value, gradient );
            // Using higher-level method of ND4J: gradient = gradient + decay * value
//            INDArray penalty = ;
            gradient.addi(value.mul(decay));
        }
        f.update(value, isBias, learningRate, batchSize, gradient);
    }
}
