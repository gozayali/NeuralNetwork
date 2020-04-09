static abstract class ActF {
  static final int SIGMOID = 0;
  static final int ReLU = 1;
  static final int TANH = 2;
  static final int ARC_TAN = 3;
  static final int SOFT_PLUS = 4;
  static final int IDENTITY = 5;
}

public static double activation(double val, int type) {
  double result= 1 / (1 + Math.exp(-val));

  if (type==ActF.SIGMOID) {
    result= (1 / (1 + Math.exp(-val)));
  } else if (type==ActF.ReLU) {
    result= (val>=0 ? val:0);
  } else if (type==ActF.TANH) {
    result= (2/(1+Math.exp(-2*val))-1);
  } else if (type==ActF.ARC_TAN) {
    result= Math.atan(val);
  } else if (type==ActF.SOFT_PLUS) {
    result= Math.log(1+Math.exp(val));
  } else if (type==ActF.IDENTITY) {
    result= val;
  }

  if (result==0) 
    result = pow(10, -10);
  return result;
}

public static double derivative(double val, int type, double func_input) {
  double result= val * (1-val);

  if (type==ActF.SIGMOID) {
    result= val * (1-val);
  } else if (type==ActF.ReLU) {
    result= (func_input>=0 ? 1:0);
  }else if (type==ActF.TANH) {
    result= 1-val*val;
  } else if (type==ActF.ARC_TAN) {
    result= (1/(Math.pow(func_input,2)+1));
  } else if (type==ActF.SOFT_PLUS) {
    result= 1/(1+Math.exp(-func_input));
  }  else if (type==ActF.IDENTITY) {
    result= 1;
  }
  if (result==0) 
    result = pow(10, -10);
  return result;
}