class Neuron {

  double LEARNING_RATIO = 0.5;
  double biasWeight;
  double delta=0;
  double[] inputs;
  double[] weights;
  double output;
  double function_input;
  int functionType= ActF.SIGMOID;

  public Neuron(int inputSize) {
    this.inputs = new double[inputSize];
    this.weights = new double[inputSize];
    for (int i = 0; i < inputSize; i++)
      weights[i]=Math.random();
    this.biasWeight = Math.random();
    function_input=0;
  }  

  void setInputs(double[] _inputs) {
    inputs=_inputs;
  }

  void updateWeight(double delta) {
    for (int i = 0; i < weights.length; i++) {
      weights[i]+= LEARNING_RATIO * delta * inputs[i];
    }
  }

  double getOutput() {
    double sum = 0;
    for (int i = 0; i < inputs.length; i++) {
      sum += inputs[i] * weights[i];
    }
    sum += biasWeight;
    function_input = sum;
    output = activation(sum,functionType);
    return output;
  }


}