class Data {
  double[][] all_data, train_data, test_data, maxMin;
  StringList resultList;

  Data(String path) {
    resultList= new StringList();
    String[] hepsi = loadStrings(path);
    int rowSize = hepsi.length;
    String[] row = split(hepsi[0], ",");    
    int colSize = row.length;
    double[][] temp_data= new double[rowSize][colSize];

    for (int i=0; i<hepsi.length; i++) {
      if (hepsi[i].replace(" ", "").length()>0) {
        row = split(hepsi[i], ",");    
        for (int j=0; j<colSize-1; j++)
          temp_data[i][j]= (Double.parseDouble(row[j]));
        temp_data[i][colSize-1]=getIndexInResultList(row[colSize-1]);
      }
    }

    all_data = new double[rowSize][colSize+resultList.size()-1];
    for (int i=0; i<all_data.length; i++) {   
      for (int j=0; j<all_data[0].length; j++) {
        if (j<colSize-1) {
          all_data[i][j]=temp_data[i][j];
        } else {
          int index=(int)temp_data[i][colSize-1];
          all_data[i][j]=0;
          if (colSize-1+index==j)
            all_data[i][j]=1;
        }
      }
    }    
    temp_data=null;
  }

  private int getIndexInResultList(String str) {
    for (int i=0; i<resultList.size(); i++)
      if (str.equals(resultList.get(i)))
        return i;
    resultList.append(str);
    return resultList.size()-1;
  }

  void shuffleData(double[][] d) {
    for (int i=0; i<d.length*20; i++) {
      int x1= (int)random(d.length);
      int x2= (int)random(d.length);
      double[] tempRow = new double[d[0].length];
      for (int j=0; j<tempRow.length; j++) {
        tempRow[j]=d[x1][j];
        d[x1][j]=d[x2][j];
        d[x2][j]=tempRow[j];
      }
    }
  }

  void parameters_scale() {
    int resSize = resultList.size();
    maxMin = new double[all_data[0].length-resSize][2];
    for (int i=0; i<maxMin.length; i++) {
      maxMin[i][0]=-1;
      maxMin[i][1]=100000;
    }

    for (int i=0; i<all_data.length; i++) {
      for (int j=0; j<all_data[0].length-resSize; j++) {
        if (all_data[i][j]>maxMin[j][0])
          maxMin[j][0]=all_data[i][j];
        if (all_data[i][j]<maxMin[j][1])
          maxMin[j][1]=all_data[i][j];
      }
    }
    for (int i=0; i<all_data.length; i++) {
      for (int j=0; j<all_data[0].length-resSize; j++) {
        all_data[i][j]= map((float)all_data[i][j], (float) maxMin[j][1], (float)maxMin[j][0], 0, 1) ;
      }
    }
  }

  void splitData_(double train_test_oran) {    
    shuffleData(all_data);
    int trainSize = (int)(all_data.length*train_test_oran);
    int testSize = all_data.length-trainSize;
    train_data= new double[trainSize][all_data[0].length];
    test_data= new double[testSize][all_data[0].length];

    for (int i=0; i<all_data.length; i++) {
      for (int j=0; j<all_data[0].length; j++) {
        if (i<trainSize)
          train_data[i][j]=all_data[i][j];
        else
          test_data[i-trainSize][j]=all_data[i][j];
      }
    }
  }

  void splitData(double train_test_oran) {
    int resSize = resultList.size();
    ArrayList<ArrayList<double[]>> data_by_results = new ArrayList<ArrayList<double[]>>();

    for (int i=0; i<resSize; i++) {
      data_by_results.add(new ArrayList<double[]>());
    }

    for (int i=0; i<all_data.length; i++) {
      int index=resSize-1;
      for (int j=0; j<resSize; j++) {
        if (all_data[i][all_data[i].length-1-j]==1) {
          index=resSize-1-j;
        }
      }
      data_by_results.get(index).add(all_data[i]);
    }

    int[][] splitted = new int[resSize][2];
    int minSize=100000000;
    int total_training=0;
    for (int i=0; i<resSize; i++) {
      splitted[i][1]=data_by_results.get(i).size();
      splitted[i][0]=(int)(splitted[i][1]*train_test_oran);
      total_training+=splitted[i][0];
      if (minSize>splitted[i][0])
        minSize=splitted[i][0];
    }

    int trainSize = total_training;
    int testSize = all_data.length-total_training;

    train_data= new double[trainSize][all_data[0].length];
    test_data= new double[testSize][all_data[0].length];

    int tri=0;
    int tei=0;
    for (int k=0; k<resSize; k++)
      if (data_by_results.get(k)!=null)
        for (int i=0; i<data_by_results.get(k).size(); i++) {
          boolean trainData=false;
          for (int j=0; j<all_data[0].length; j++) {
            if (i< (int)(data_by_results.get(k).size()*train_test_oran)) {
              train_data[tri][j]=data_by_results.get(k).get(i)[j];
              trainData=true;
            } else {
              test_data[tei][j]=data_by_results.get(k).get(i)[j];
            }
          }
          if (trainData)
            tri++;
          else
            tei++;
        }

    shuffleData(train_data);
    println("NAME\tCNT\tTRAIN\tTEST");
    for (int i=0; i<splitted.length; i++) {
      println("["+resultList.get(i)+"]\t"+ splitted[i][1]+"\t"+ splitted[i][0]+"\t"+(splitted[i][1]-splitted[i][0]) );
    }
    println("TOTAL\t"+all_data.length+"\t"+train_data.length+"\t"+test_data.length+"\n");
    
  }
  
}
