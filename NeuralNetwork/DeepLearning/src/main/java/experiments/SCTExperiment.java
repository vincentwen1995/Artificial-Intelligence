/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import nl.tue.s2id90.dl.NN.transform.MeanSubtraction;
import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Convolution2D;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.layer.PoolMax2D;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.optimizer.update.Adadelta;
import nl.tue.s2id90.dl.NN.optimizer.update.GD_Momentum;
import nl.tue.s2id90.dl.NN.optimizer.update.L2Decay;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.Experiment;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.PrimitivesDataGenerator;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;

/**
 *
 * @author Administrator
 */
public class SCTExperiment extends Experiment {    
    int batchSize = 16;
    int epochs = 5;
    float learningRate = 0.01f;
    float beta = 0.9f;
    float epsilon = (float) 1e-6;
    float lambda = 0.0001f;
    int kernelSize = 3;
    int kernels = 8;
    int convStride = 1;
    int poolStride = 2;
    
    String[] labels = {
            "Square", "Circle", "Triangle"
        };
        
    SCTExperiment(){ super(true); }
    
    
    public void go() throws IOException {
        // read input and print some information on the data
        int seed = 11081961, trainingDataSize = 1200, testDataSize = 200;
        InputReader reader = new PrimitivesDataGenerator(batchSize, seed, trainingDataSize, testDataSize);
        System.out.println("Reader info:\n" + reader.toString());
        
        ShowCase showCase = new ShowCase(i -> labels[i]);
        FXGUI.getSingleton().addTab("show case", showCase.getNode());
        showCase.setItems(reader.getValidationData(100));
        
        int input_width = 28;
        int input_height = 28;
        int input_depth = 1;
        int outputs = labels.length;
        
        Model model = createModel(input_width, input_height, input_depth, outputs);
        
        Optimizer sgd = SGD.builder()
                .model(model)
                .validator(new Classification())
                .learningRate(learningRate)
//                .updateFunction(() -> new GD_Momentum(beta))
//                .updateFunction(() -> new L2Decay(() -> new GD_Momentum(beta), lambda))
//                .updateFunction(() -> new L2Decay(GradientDescent::new, lambda))        //Best performance
//                .updateFunction(() -> new Adadelta(beta, epsilon))
                .build();
        // Data Preprocessing
        MeanSubtraction data_pre = new MeanSubtraction();
        data_pre.fit((reader.getTrainingData()));
        data_pre.transform(reader.getTrainingData());
        data_pre.transform(reader.getValidationData());
        trainModel(model, reader, sgd, epochs, 250);
    }
    
    Model createModel(int input_width, int input_height, int input_depth, int outputs) {

        Model model = new Model(new InputLayer("In", new TensorShape(input_width, input_height, input_depth), true));        
        model.addLayer(new Convolution2D("Conv", new TensorShape(input_width, input_height, input_depth), kernelSize, kernels, new RELU()));
        model.addLayer(new PoolMax2D("Pool", new TensorShape(input_width, input_height, kernels), poolStride));
        model.addLayer(new Flatten("Flatten", new TensorShape(input_width / poolStride, input_height / poolStride, kernels)));
        model.addLayer(new OutputSoftmax("Out", new TensorShape((input_width / poolStride) * (input_height / poolStride) * kernels), labels.length, new CrossEntropy()));
        model.initialize(new Gaussian());
        System.out.println(model);
        return model;
    }
    
    public static void main(String[] args) throws IOException {
        new SCTExperiment().go();
    }
        
    
}
