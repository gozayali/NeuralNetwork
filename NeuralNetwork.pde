class NeuralNetwork { //<>//
  int[][] test_table;
  double sqError;
  Layer[] layers;
  double[] target;
  double[] inputs;
  double[] outputs;
  int functionType = ActF.SIGMOID;
  double LEARNING_RATIO=0.5;

  NeuralNetwork(Layer[] _layers) {
    layers=_layers;
  }

  NeuralNetwork(int[] neuronPerLayer) {
    layers=new Layer[neuronPerLayer.length];
    for (int i=0; i<layers.length; i++)
      layers[i]=new Layer(neuronPerLayer[i]);
  }

  void setFunctionType(int type) {
    functionType= type;
    for (int i=0; i<layers.length; i++)
      layers[i].functionType = functionType;
  }

  void setLearningRatio(double LR) {
    LEARNING_RATIO= LR;
    for (int i=0; i<layers.length; i++)
      layers[i].LEARNING_RATIO= LR;
  }

  void setInputs(double[] _inputs) {
    inputs=_inputs;
  }

  void setTarget(double[] _target) {
    target=_target;
    outputs= new double[target.length];
  }

  void forward() {
    layers[0].setInputs(inputs);
    for (int i=1; i<layers.length; i++) {
      layers[i].setInputs(layers[i-1].getOutput());
    }
    outputs =layers[layers.length-1].getOutput();
    sqError=0;
    for (int i=0; i<outputs.length; i++)
      sqError+= pow((float)(target[i]-outputs[i]), 2f);
  }

  void backward() {
    deltalarHesapla();
    for (int i=0; i < layers.length; i++)       
      for (int j=0; j<layers[i].neurons.length; j++)
        layers[i].neurons[j].updateWeight(layers[i].neurons[j].delta);
  }

  void deltalarHesapla() {
    for (int i=layers.length-1; i >= 0; i--) {
      if (i==layers.length-1) {
        for (int j=0; j<layers[i].neurons.length; j++) {
          layers[i].neurons[j].delta = (target[j] - layers[i].neurons[j].output) * 
            derivative(layers[i].neurons[j].output, functionType, layers[i].neurons[j].function_input);
        }
      } else {
        for (int j=0; j<layers[i].neurons.length; j++) {
          double errorSum=0;
          for (int k=0; k<layers[i+1].neurons.length; k++)
            errorSum += layers[i+1].neurons[k].delta * layers[i+1].neurons[k].weights[j];

          layers[i].neurons[j].delta = errorSum * derivative(layers[i].neurons[j].output, functionType, layers[i].neurons[j].function_input);
        }
      }
    }
  }

  void train(Data data, int iteration) {
    int outSize= data.resultList.size();
    double[] inp= new double[data.train_data[0].length-outSize];
    double[] tar=new double[outSize];
    int lastPart = 0;
    print("| Training process below! |\n =");
    for (int cnt=0; cnt<iteration; cnt++) {
      for (int i=0; i<data.train_data.length; i++) {
        for (int j=0; j<data.train_data[0].length-outSize; j++) {
          inp[j]=data.train_data[i][j];
        }
        for (int j=0; j<outSize; j++) {
          tar[j]=data.train_data[i][data.train_data[0].length-outSize+j];
        }
        setInputs(inp);
        setTarget(tar);
        forward();
        backward();
      }
      int part=cnt*25/iteration;
      if (part>lastPart) {
        print("=");
        lastPart=part;
      }
    }
    println("\nTraining completed");
  }

  String[][] test(Data data) {
    println("\nTest started...");
    int score=0;
    int resSize=data.resultList.size();
    double[] inp= new double[data.train_data[0].length-resSize];

    test_table = new int [resSize][resSize];
    for (int i=0; i<resSize; i++)
      for (int j=0; j<resSize; j++)
        test_table[i][j]=0;

    String[][] out = new String[data.test_data.length+1][2];

    for (int i=0; i<data.test_data.length; i++) {
      inp= new double[data.train_data[0].length-resSize];
      for (int j=0; j<inp.length; j++) {
        inp[j]=data.test_data[i][j];
      }
      setInputs(inp); 
      forward();

      int index=0;
      int gercekIndex=0;
      double maxRes = outputs[0];
      for (int j=0; j<outputs.length; j++) {
        if (outputs[j]>maxRes) {
          index=j;
          maxRes =  outputs[j];
        }
        if (data.test_data[i][(data.test_data[i].length-resSize+j)]==1)
          gercekIndex=j;
      }

      //      println("\nOUTS: "+i);
      //      printArray(outputs);

      out[i][0]=data.resultList.get(index);
      out[i][1]=data.resultList.get(gercekIndex);
      test_table[gercekIndex][index]++;

      if (index==gercekIndex)
        score++;
    }
    out[out.length-1][0]="ACC";
    out[out.length-1][1]=(score*100)/data.test_data.length + "% ("+score+"/"+data.test_data.length+")";
    println( "Test Completed!\nACC: "+ (double)score/data.test_data.length + " ("+score+"/"+data.test_data.length+")\nMSE: "+sqError);

    print_test_result_table(data);
    return out;
  }

  void print_test_result_table(Data data) {
    print("\nR / P");
    int resSize=data.resultList.size();
    for (int i=0; i<resSize; i++)
      print("\t["+data.resultList.get(i)+"]");
    println("\tACC");
    for (int i=0; i<resSize; i++) {
      print("["+data.resultList.get(i)+"]\t");
      int tot=0;
      for (int j=0; j<resSize; j++) {
        print(test_table[i][j]+"\t");
        tot+=  test_table[i][j];
      }
      println(nf((float)test_table[i][i]/tot));
    }
  }

  String[][] predict(double[][] table_to_predict, Data trained_data) {    
    String[][] predictResults= new String[table_to_predict.length][table_to_predict[0].length+1];
    double[] inp= new double[table_to_predict[0].length];

    for (int i=0; i<table_to_predict.length; i++)
      for (int j=0; j<table_to_predict[0].length; j++) {
        predictResults[i][j] = String.format("%.2f", table_to_predict[i][j]);
        table_to_predict[i][j]= map((float)table_to_predict[i][j], (float) trained_data.maxMin[j][1], (float)trained_data.maxMin[j][0], 0, 1);
      }

    for (int i=0; i<table_to_predict.length; i++) {
      for (int j=0; j<inp.length; j++) {
        inp[j]=table_to_predict[i][j];
      }
      setInputs(inp); 
      forward();

      int index=0;
      double maxRes = outputs[0];
      for (int j=1; j<outputs.length; j++) {
        if (outputs[j]>maxRes) {
          index=j;
          maxRes =  outputs[j];
        }
      }
      predictResults[i][table_to_predict[0].length]=trained_data.resultList.get(index);
    }

    return predictResults;
  }

  String getNetworkJSON() {
    String json="{\"LR\":"+LEARNING_RATIO+",\"FUNC\":"+( functionType==0 ? "\"SIGMOID\"":"OTHER" )+",\"Layers\": [";
    for (int i=0; i<layers.length; i++) {
      json+="{ \"Level\":"+i+",\"Neurons\": [";
      for (int j=0; j<layers[i].neurons.length; j++) {
        json+= "{\"Index\": "+j+",\"Bias\":"+ layers[i].neurons[j].biasWeight+",\"Weights\": [";
        for (int k=0; k<layers[i].neurons[j].weights.length; k++) {
          json+= layers[i].neurons[j].weights[k];
          if (k<layers[i].neurons[j].weights.length-1)
            json+=",";
        }        
        json+="]}";
        if (j<layers[i].neurons.length-1)
          json+=",";
      }
      json+="]}";
      if (i<layers.length-1)
        json+=",";
    }
    json+="]}";
    return json;
  }
}
