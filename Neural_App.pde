NeuralNetwork net; //<>// //<>// //<>// //<>//
Data data;
String txtName="data/iris";
String[][] result;

void setup() {
  size(640, 480);
  textSize(12);
  background(0);
  long ms=millis();
  int iteration=5000;
  double train_oran=0.9;

  data= new Data(txtName+".txt");
  data.parameters_scale();
  data.splitData(train_oran);

  net = new NeuralNetwork(new int[]{16,16, data.resultList.size()});  //32,6 red_wine
  net.setFunctionType(ActF.SIGMOID);
  net.setLearningRatio(0.5);  //0.2 red_wine

  net.train(data, iteration); 
  long ms2=millis(); //<>// //<>//
  result = net.test(data);
  ms2=millis()-ms2;

  background(0);
  String netString=net.layers.length+ " layerlı ( ";
  for (int i=0; i<net.layers.length; i++) {
    netString+= net.layers[i].size;
    if (i<net.layers.length-1)
      netString+="-";
  }
  netString+=" ) network için; ";
  text(netString, 30, 80);
  text(iteration+" Train Tamamlandı", 30, 100);

  text(String.format("Train data oranı: %.2f", train_oran), 30, 140);
  text(String.format("SqError: %.20f", net.sqError), 30, 160); 
  text(result[result.length-1][1], 30, 180);

  text("Geçen Süre: "+(millis()-ms)+" ms", 30, 220);
  text("Test Süre: "+ms2+" ms", 30, 240);
}

void draw() {
}

void mousePressed() {

   println(net.getNetworkJSON());
  
  /*
  // son kayıtların inputlarını dene
  int kayitSayisi=1000;
  String[] hepsi = loadStrings("rcm_ok.txt");
//  String[] basliklar = split(loadStrings(txtName+"_headers.txt")[0], ",");
  String[] row = split(hepsi[0], ",");    
  int colSize = row.length;
  double[][] test= new double[kayitSayisi][colSize-1];

  for (int i=hepsi.length-kayitSayisi; i<hepsi.length; i++) {
    row = split(hepsi[i], ",");    
    for (int j=0; j<colSize-1; j++)
      test[i-(hepsi.length-kayitSayisi)][j]= (Double.parseDouble(row[j]));
  }

  long mil=millis();
  String[][] res = net.predict(test, data);
  println("\n\n"+(millis()-mil)+" ms");
  int nokSayisi= 0;

//  for (int j=0; j<res[0].length; j++) 
//    print(basliklar[j]+"\t");
  println();
  for (int i=0; i<res.length; i++) {
    for (int j=0; j<res[0].length; j++) {
      print(res[i][j]+"\t");
    }
    println();
    if(res[i][res[i].length-1].charAt(1)=='N') //<>// //<>//
      nokSayisi++;
  }
  println("\nNOK: "+ nokSayisi);
  */
}
