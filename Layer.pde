class Layer {
  int size;
  Neuron[] neurons;
  double[] outputs;
  int functionType= ActF.SIGMOID;
  double LEARNING_RATIO=0.5;

  Layer(int _size) {
    size=_size;
    neurons = new Neuron[size];
    outputs= new double[size];
  }

  void setInputs(double[] inputs) {
    for (int i=0; i<size; i++) {
      if (neurons[i]==null)
        neurons[i] = new Neuron(inputs.length);
      neurons[i].functionType= functionType;
      neurons[i].LEARNING_RATIO=LEARNING_RATIO;
      neurons[i].setInputs(inputs);
    }
  }

  double[] getOutput() {
    for (int i=0; i<neurons.length; i++)
      outputs[i]=neurons[i].getOutput();
    return outputs;
  }
}