
public class NeuralNetwork {
    public int layers; //internal layers
    public int inputSize;
    public int outputSize;
    public int layerSize;
    public int stochastic;
    public double dropoutFactor;
    public double l2Factor;
    private boolean softmax;
    private double[][][] weights;
    private double[][] biases;
    /*
    Input layer is counted as layer -1
    weights[2][6][0] represents the weight from neuron 6 of internal layer 1 to neuron 0 of internal layer 2
    */

    public NeuralNetwork(int l, int i, int o, int s, boolean f) {
        layers = l; inputSize = i; outputSize = o; layerSize = s; softmax = f;
        stochastic = 1;
        dropoutFactor = 0;
        l2Factor = 0;
        weights = new double[layers + 1][][];
        weights[0] = new double[inputSize][layerSize];
        for (int j = 0; j < inputSize; j++) {
            for (int k = 0; k < layerSize; k++) {
                weights[0][j][k] = (Math.random() * 2 - 1) * Math.sqrt(2.0 / inputSize); //He initialization
            }
        }
        for (int j = 1; j < layers; j++) {
            weights[j] = new double[layerSize][layerSize];
            for (int k = 0; k < layerSize; k++) {
                for (int m = 0; m < layerSize; m++) {
                    weights[j][k][m] = (Math.random() * 2 - 1) * Math.sqrt(2.0 / layerSize);
                }
            }
        }
        weights[layers] = new double[layerSize][outputSize];
        for (int j = 0; j < layerSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                weights[layers][j][k] =(Math.random() * 2 - 1) * Math.sqrt(2.0 / layerSize);
            }
        }
        biases = new double[layers + 1][];
        for (int j = 0; j < layers; j++) {
            biases[j] = new double[layerSize];
            for (int k = 0; k < layerSize; k++) {
                biases[j][k] = (Math.random() * 2 - 1);
            }
        }
        biases[layers] = new double[outputSize];
        for (int j = 0; j < outputSize; j++) {
            biases[layers][j] = (Math.random() * 2 - 1);
        }
    }

    //Returns the output of the nerual network for a given input
    public double[] think(double[] input) {
        return getNetwork(input, null)[layers];
    }
    
    //Runs backpropagation on a series of given testcases with corresponding expected outputs
    public void learn(int testcases, double[][] inputs, double[][] outputs) {
        //Apply stochastic training
        for (int s = 0; s < stochastic; s++) {
            boolean[][] dropout = new boolean[layers][layerSize];
            for (int i = 0; i < layers; i++) {
                for (int j = 0; j < layerSize; j++) {
                    dropout[i][j] = Math.random() < dropoutFactor; //Randomly remove neurons from training to reduce overfitting
                }
            }
            double[][][] dWeight = new double[layers + 1][][];
            double[][] dBias = new double[layers + 1][];
            dWeight[layers] = new double[layerSize][outputSize];
            for (int i = 0; i < layerSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    dWeight[layers][i][j] += weights[layers][i][j] * l2Factor; //L2 regularization
                }
            }
            dBias[layers] = new double[outputSize];
            for (int i = 1; i < layers; i++) {
                dWeight[i] = new double[layerSize][layerSize];
                for (int j = 0; j < layerSize; j++) {
                    for (int k = 0; k < outputSize; k++) {
                        dWeight[i][j][k] += weights[i][j][k] * l2Factor;
                    }
                }
                dBias[i] = new double[layerSize];
            }
            dWeight[0] = new double[inputSize][layerSize];
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < layerSize; j++) {
                    dWeight[0][i][j] += weights[0][i][j] * l2Factor;
                }
            }
            dBias[0] = new double[layerSize];
            double gradSum = 0;
            double cost = 0;
            for (int t = 0; t < testcases; t += stochastic) {
                double[] in = inputs[t];
                double[] target = outputs[t];
                double[][] network = getNetwork(in, null);
                //Output to internal layer
                double[] dcda = new double[outputSize];
                double[] nextdcda = new double[layerSize];
                //Compute gradient of cost function with respect to expected vs output
                if (softmax) {
                    //softmax output follows special derivative formulas
                    int correct = 0;
                    double[] softmaxed = softmax(network[layers]);
                    for (int i = 0; i < outputSize; i++) {
                        if (target[i] > 0) {
                            correct = i;
                            cost -= Math.log(softmaxed[i]);
                            break;
                        }
                    }
                    for (int i = 0; i < outputSize; i++) {
                        if (i == correct) {
                            dcda[i] = softmaxed[i] - 1;
                        }
                        else {
                            dcda[i] = softmaxed[i];
                        }
                    }
                }
                else {
                    for (int i = 0; i < outputSize; i++) {
                        dcda[i] = network[layers][i] - target[i];
                        cost += dcda[i] * dcda[i];
                    }
                }
                for (int i = 0; i < outputSize; i++) {
                    //Compute gradient of cost function with respect to current layer's biases
                    dBias[layers][i] += dcda[i] * squishDeriv(network[layers][i]);
                    gradSum += dBias[layers][i] * dBias[layers][i];
                    //Compute gradient of cost function with respect to current layer's weights
                    for (int j = 0; j < layerSize; j++) {
                        if (dropout[layers - 1][j]) {
                            continue;
                        }
                        dWeight[layers][j][i] += dcda[i] * squishDeriv(network[layers][i]) * network[layers - 1][j];
                        gradSum += dWeight[layers][j][i] * dWeight[layers][j][i];
                    }
                    //Compute gradient of cost function with respect to previous layer's activations
                    for (int j = 0; j < layerSize; j++) {
                        if (dropout[layers - 1][j]) {
                            continue;
                        }
                        nextdcda[j] += dcda[i] * squishDeriv(network[layers][i]) * weights[layers][j][i];
                    }
                }
                //Internal to internal layer
                for (int l = layers - 1; l > 0; l--) {
                    //Backpropagate the gradient of cost function with respect to layer activations
                    dcda = nextdcda;
                    nextdcda = new double[layerSize];
                    for (int i = 0; i < layerSize; i++) {
                        if (dropout[l][i]) {
                            continue;
                        }
                        //Compute gradient of cost function with respect to current layer's biases
                        dBias[l][i] += dcda[i] * squishDeriv(network[l][i]);
                        gradSum += dBias[l][i] * dBias[l][i];
                        //Compute gradient of cost function with respect to current layer's weights
                        for (int j = 0; j < layerSize; j++) {
                            if (dropout[l][j]) {
                                continue;
                            }
                            dWeight[l][j][i] += dcda[i] * squishDeriv(network[l][i]) * network[l - 1][j];
                            gradSum += dWeight[l][j][i] * dWeight[l][j][i];
                        }
                        //Compute gradient of cost function with respect to previous layer's activations
                        for (int j = 0; j < layerSize; j++) {
                            if (dropout[l][j]) {
                                continue;
                            }
                            nextdcda[j] += dcda[i] * squishDeriv(network[l][i]) * weights[l][j][i];
                        }
                    }
                }
                //Internal to input layer
                //Backpropagate the gradient of cost function with respect to layer activations
                dcda = nextdcda;
                for (int i = 0; i < layerSize; i++) {
                    if (dropout[0][i]) {
                        continue;
                    }
                    //Compute gradient of cost function with respect to current layer's biases
                    dBias[0][i] += dcda[i] * squishDeriv(network[0][i]);
                    gradSum += dBias[0][i] * dBias[0][i];
                    //Compute gradient of cost function with respect to current layer's weights
                    for (int j = 0; j < inputSize; j++) {
                        dWeight[0][j][i] += dcda[i] * squishDeriv(network[0][i]) * in[j];
                        gradSum += dWeight[0][j][i] * dWeight[0][j][i];
                    }
                }
            }
            gradSum = Math.min(-1000, Math.sqrt(gradSum) * -1);
            //apply gradient descent to weights and biases
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < layerSize; j++) {
                    weights[0][i][j] += dWeight[0][i][j] / gradSum;
                }
            }
            for (int i = 1; i < layers; i++) {
                for (int j = 0; j < layerSize; j++) {
                    for (int k = 0; k < layerSize; k++) {
                        weights[i][j][k] += dWeight[i][j][k] / gradSum;
                    }
                }
            }
            for (int i = 0; i < layerSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weights[layers][i][j] += dWeight[layers][i][j] / gradSum;
                }
            }
            for (int i = 0; i < layers; i++) {
                for (int j = 0; j < layerSize; j++) {
                    biases[i][j] += dBias[i][j] / gradSum;
                }
            }
            for (int i = 0; i < outputSize; i++) {
                biases[layers][i] += dBias[layers][i] / gradSum;
            }
        }
    }

    //Returns the derivative of each output neuron with respect to each input neuron
    //Work in progress
    public double[][] explain(double[] input) {
        double[][] deriv = new double[outputSize][inputSize];
        double[][] network = getNetwork(input, null);
        double[][] innerDeriv = new double[layerSize][layerSize];
        return deriv;
    }

    //Returns the activations of the entire network for a given input
    private double[][] getNetwork(double[] input, boolean[][] dropout) {
        double[][] network = new double[layers + 1][];
        network[0] = new double[layerSize];
        for (int i = 0; i < layerSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                network[0][i] += input[j] * weights[0][j][i];
            }
            network[0][i] += biases[0][i];
            network[0][i] = squish(network[0][i]);
        }
        for (int i = 1; i < layers; i++) {
            network[i] = new double[layerSize];
            for (int j = 0; j < layerSize; j++) {
                for (int k = 0; k < layerSize; k++) {
                    if (dropout != null && dropout[i - 1][k]) {
                        continue;
                    }
                    network[i][j] += network[i - 1][k] * weights[i][k][j];
                }
                network[i][j] += biases[i][j];
                network[i][j] = squish(network[i][j]);
            }
        }
        network[layers] = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < layerSize; j++) {
                if (dropout != null && dropout[layers - 1][j]) {
                    continue;
                }
                network[layers][i] += network[layers - 1][j] * weights[layers][j][i];
            }
            network[layers][i] += biases[layers][i];
            network[layers][i] = squish(network[layers][i]);
        }
        return network;
    }

    //Applies the softmax function to an array
    public static double[] softmax(double[] values) {
        double[] softmaxed = new double[values.length];
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += Math.pow(Math.E, values[i]);
        }
        for (int i = 0; i < values.length; i++) {
            softmaxed[i] = Math.pow(Math.E, values[i]) / sum;
        }
        return softmaxed;
    }

    //The squish function
    private double squish(double value) {
        return value > 0 ? value : value * 0.01;
        //return Math.tanh(value);
    }

    //Calculates the derivative of the squish function for a given output value
    private double squishDeriv(double value) {
        return value > 0 ? 1 : 0.01;
        //double x = 0.5 * Math.log((1 + value) / (1 - value));
        //return 4 / Math.pow(Math.pow(Math.E, x) + Math.pow(Math.E, x * -1), 2);
    }
}
