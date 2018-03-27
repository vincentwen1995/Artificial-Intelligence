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
    float beta;
    float epsilon;
    INDArray accum_g2;
    INDArray accum_dx2;
//    INDArray g2;
//    INDArray dx2;
    INDArray update;
    
    public Adadelta(float beta, float epsilon){
        this.beta = beta;
        this.epsilon = epsilon;
    }
    
    @Override
    public void update(INDArray value, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        // Initialize the INDArray variables
        if (update == null) {
            accum_g2 = gradient.dup('f').assign(0);
            accum_dx2 = value.dup('f').assign(0);
//            g2 = gradient.dup('f').assign(0);
//            dx2 = value.dup('f').assign(0);
            update = value.dup('f').assign(0);
        }      
//        // Compute squares of gradients
//        g2 = gradient.dup();        
//        g2.muli(gradient);                
//        // First scale accum_g2 with beta
//        Nd4j.getBlasWrapper().level1().scal( accum_g2.length(), beta, accum_g2);
//        // Then update accum_g2 with the scaled gradients squared
//        Nd4j.getBlasWrapper().level1().axpy( accum_g2.length(), (1 - beta), g2, accum_g2 );
//        // Compute the RMS ratio and compute the update                
//        update = sqrt(accum_dx2.add(epsilon)).divi(sqrt(accum_g2.add(epsilon))).muli(-1.0f).muli(gradient);
//        dx2 = update.dup();
//        dx2.muli(update);
//        // Scale accum_dx2 with beta
//        Nd4j.getBlasWrapper().level1().scal( accum_dx2.length(), beta, accum_dx2);
//        // Update accum_dx2 with the scaled updates squared        
//        Nd4j.getBlasWrapper().level1().axpy( accum_dx2.length(), (1 - beta), dx2, accum_dx2 );
//        // Apply update
//        Nd4j.getBlasWrapper().level1().axpy( value.length(), 1.0f, update, value );

        // Using higher-level Nd4j methods:
        // Accumulate Gradient: accum_g2 = beta * accum_g2 + (1 - beta) * gradient * gradient
        accum_g2.muli(beta).addi((gradient.mul(gradient)).muli(1 - beta));
        // Compute Update: update = -1 * sqrt(accum_dx2 + epsilon) / sqrt(accum_g2 + epsilon) * gradient
        update = ((sqrt(accum_dx2.add(epsilon)).muli(-1.0f)).divi(sqrt(accum_g2.add(epsilon)))).muli(gradient);
        // Accumulate Updates: accum_dx2 = beta * accum_dx2 + (1 - beta) * update * update
        accum_dx2.muli(beta).addi((update.mul(update)).muli(1 - beta));
        // Apply Update: value = value + update
        value.addi(update);
        gradient.assign(0);
    }
}
