/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

/**
 *
 * @author Administrator
 */
public class Adadelta implements UpdateFunction{
    float rho;          //Decay rate
    float epsilon;      
    INDArray accum_g2;  //Accumulation variable/running average of squared gradients
    INDArray accum_dx2; //Accumulation variable/running average of squared updates
    INDArray update;
    
    public Adadelta(float rho, float epsilon){
        this.rho = rho;
        this.epsilon = epsilon;
    }
    
    @Override
    public void update(INDArray value, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        // Initialize the INDArray variables
        if (update == null) {
            accum_g2 = gradient.dup('f').assign(0);
            accum_dx2 = value.dup('f').assign(0);
            update = value.dup('f').assign(0);
        }      
        // Accumulate Gradient: accum_g2 <- rho * accum_g2 + (1 - rho) * gradient * gradient
        accum_g2.muli(rho).addi((gradient.mul(gradient)).muli(1 - rho));
        // Compute Update: update <- -1 * sqrt(accum_dx2 + epsilon) / sqrt(accum_g2 + epsilon) * gradient
        update = ((sqrt(accum_dx2.add(epsilon)).muli(-1.0f)).divi(sqrt(accum_g2.add(epsilon)))).muli(gradient);
        // Accumulate Updates: accum_dx2 <- rho * accum_dx2 + (1 - rho) * update * update
        accum_dx2.muli(rho).addi((update.mul(update)).muli(1 - rho));
        // Apply Update: value <- value + update
        value.addi(update);
        gradient.assign(0);
    }
}
