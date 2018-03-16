/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;
import nl.tue.s2id90.dl.experiment.Experiment;
import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Regression;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;

/**
 *
 * @author Administrator
 */
public class FunctionExperiment extends Experiment {
    // (hyper) parameters
    // ...
    int batchSize = 32;
    int epochs = 10;
    float learningRate = 0.01f;
    
    public void go() throws IOException {
    // read input and print some information on the data
        InputReader reader = GenerateFunctionData.THREE_VALUED_FUNCTION(batchSize);
        System.out.println("Reader info:\n" + reader.toString());
        
        int inputs = reader.getInputShape().getNeuronCount();
        int outputs = reader.getOutputShape().getNeuronCount();
        
        Model model = createModel(inputs, outputs);
        
        Optimizer sgd = SGD.builder()
                .model(model)
                .validator(new Regression())
                .learningRate(learningRate)
                .build();
        
        trainModel(model, reader, sgd, epochs, 0);
    }
    
    Model createModel(int inputs, int outputs) {
        Model model = new Model(new InputLayer("In", new TensorShape(inputs), true));
        model.addLayer(new SimpleOutput("Out", new TensorShape(inputs), outputs, new MSE(), true));
        model.initialize(new Gaussian());
        System.out.println(model);
        return model;
    }
    
    public static void main(String[] args) throws IOException {
        new FunctionExperiment().go();
    }
    
}
