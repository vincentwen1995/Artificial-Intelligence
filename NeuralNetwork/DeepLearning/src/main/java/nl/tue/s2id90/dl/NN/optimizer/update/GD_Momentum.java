/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Administrator
 */
public class GD_Momentum implements UpdateFunction{
    INDArray update;
    float beta;

    public GD_Momentum() {
        this.beta = 0.9f;
    }
//    public GD_Momentum(float beta){
//        this.beta = beta;
//    }
    @Override
    public void update(INDArray value, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        if (update == null) update = gradient.dup('f').assign(0);
        
        float factor = -(learningRate/batchSize);
        
        // Method1: Slides
        
        // First scale the momentum with the hyperparameter beta
        Nd4j.getBlasWrapper().level1().scal( update.length(), beta, update);
        // Then update the momentum with the scaled gradient
        Nd4j.getBlasWrapper().level1().axpy( update.length(), (1 - beta), gradient, update );
        // Finally update the weight with the scaled momentum
        Nd4j.getBlasWrapper().level1().axpy( value.length(), factor, update, value );
        
        
        //Method2: CS231N
        /*
        // First scale the momentum with the hyperparameter beta
        Nd4j.getBlasWrapper().level1().scal( update.length(), beta, update);
        // Then update the momentum with the scaled gradient
        Nd4j.getBlasWrapper().level1().axpy( update.length(), factor, gradient, update );
        // Finally update the weight with the momentum
        Nd4j.getBlasWrapper().level1().axpy( value.length(), 1.0f, update, value );
        */
        gradient.assign(0);
    }
}
