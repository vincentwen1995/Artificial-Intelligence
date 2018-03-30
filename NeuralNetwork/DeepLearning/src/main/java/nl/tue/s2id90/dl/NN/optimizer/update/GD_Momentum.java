/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.optimizer.update;

import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Administrator
 */
public class GD_Momentum implements UpdateFunction{    
    INDArray update;    
    float beta;     //Momentum factor
        
    public GD_Momentum(float beta){
        this.beta = beta;
    }
    @Override
    public void update(INDArray value, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        if (update == null) update = gradient.dup('f').assign(0);
        
        float factor = -(learningRate/batchSize);                
        
        // Method: CS231N 
        // Update the momentum: v <- beta * v + factor * gradient
        update.muli(beta).addi(gradient.muli(factor));
        // Update the value(weights): value <- value + update
        value.addi(update);
        
        gradient.assign(0);
    }
}
