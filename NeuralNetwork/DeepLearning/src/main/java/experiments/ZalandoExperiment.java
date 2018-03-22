/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.Experiment;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.MNISTReader;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;

/**
 *
 * @author Administrator
 */
public class ZalandoExperiment extends Experiment {
    int batchSize = 32;
    int epochs = 5;
    float learningRate = 0.01f;
    int layers = 10;
    int layerSize = 10;
    
    String[] labels = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        };
        
    ZalandoExperiment(){ super(true); }
    
    
    public void go() throws IOException {
    // read input and print some information on the data
        InputReader reader = MNISTReader.fashion(batchSize);
        System.out.println("Reader info:\n" + reader.toString());
        
        ShowCase showCase = new ShowCase(i -> labels[i]);
        FXGUI.getSingleton().addTab("show case", showCase.getNode());
        showCase.setItems(reader.getValidationData(100));
        
        int inputs = reader.getInputShape().getNeuronCount();
        int outputs = reader.getOutputShape().getNeuronCount();
        
        Model model = createModel(inputs, outputs);
        
        Optimizer sgd = SGD.builder()
                .model(model)
                .validator(new Classification())
                .learningRate(learningRate)
//                .updateFunction(GradientDescent::new)
                .build();
        
        trainModel(model, reader, sgd, epochs, 0);
    }
    
    Model createModel(int inputs, int outputs) {

        Model model = new Model(new InputLayer("In", new TensorShape(inputs), true));
        model.addLayer(new Flatten("Flatten", new TensorShape(inputs)));
        model.addLayer(new OutputSoftmax("Out", new TensorShape(28 * 28), labels.length, new CrossEntropy()));
//        model.addLayer(new FullyConnected("fc1", new TensorShape(inputs), layerSize, new RELU()));
//        for (int i = 1; i < layers; i++) {
//            model.addLayer(new FullyConnected("fc"+Integer.toString(i + 1), new TensorShape(layerSize), layerSize, new RELU()));
//        }
//        model.addLayer(new SimpleOutput("Out", new TensorShape(layerSize), outputs, new MSE(), true));
        model.initialize(new Gaussian());
        System.out.println(model);
        return model;
    }
    
    public static void main(String[] args) throws IOException {
        new ZalandoExperiment().go();
    }
        
    
}
