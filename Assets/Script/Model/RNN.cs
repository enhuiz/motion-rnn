using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System;

// Details of the model can be found in an open course of Stanford
// CS231n Winter 2016: Lecture 10: Recurrent Neural Networks, Image Captioning, LSTM
// https://www.youtube.com/watch?v=yCC09vCHzF8

// A minimal implement of rnn in python from scratch given by the above lecture:
// https://gist.github.com/karpathy/d4dee566867f8291f086


namespace Model {
    public class RNN {
        int inputDim;
        int outputDim;
        int hiddenDim;

        Matrix<double> Wxh;
        Matrix<double> Whh;
        Matrix<double> Why;

        Matrix<double> bh;
        Matrix<double> by;

        Matrix<double> RandomMatrix(int row, int column) {
            Random rand = new Random();
            Matrix<double> ret = Matrix<double>.Build.Dense(row, column);
            for (int r = 0; r < row; r++) {
                for (int c = 0; c < column; c++) {
                    ret[r, c] = rand.NextDouble();
                }
            }
            return ret;
        }

        public RNN(int inputDim, int hiddenDim, int outputDim) {
            this.inputDim = inputDim;
            this.outputDim = outputDim;
            this.hiddenDim = hiddenDim;

            Wxh = RandomMatrix(inputDim, hiddenDim);
            Whh = RandomMatrix(hiddenDim, hiddenDim);
            Why = RandomMatrix(hiddenDim, outputDim);

            bh = Matrix<double>.Build.Dense(1, hiddenDim, 0.1);
            by = Matrix<double>.Build.Dense(1, outputDim, 0.1);
        }

        public void Predict(List<Matrix<double>> xs,
                                out List<Matrix<double>> ys) {
            List<Matrix<double>> hs = new List<Matrix<double>>();
            ys = new List<Matrix<double>>();

            for (int t = 0; t < xs.Count; t++) {
                if (t == 0) {
                    hs.Add(Sigmoid(xs[t] * Wxh + bh));
                } else {
                    hs.Add(Sigmoid(xs[t] * Wxh + hs[t - 1] * Whh + bh));
                }
                ys.Add(Sigmoid(hs[t] * Why + by));
            }
        }

        public double FeedForward(List<Matrix<double>> xs,
                                List<Matrix<double>> y_s,
                                out List<Matrix<double>> hs,
                                out List<Matrix<double>> ys) {
            hs = new List<Matrix<double>>();
            ys = new List<Matrix<double>>();
            double loss = 0;

            for (int t = 0; t < xs.Count; t++) {
                if (t == 0) {
                    hs.Add(Sigmoid(xs[t] * Wxh + bh));
                } else {
                    hs.Add(Sigmoid(xs[t] * Wxh + hs[t - 1] * Whh + bh));
                }
                ys.Add(Sigmoid(hs[t] * Why + by));
                loss += ReduceSum((ys[t] - y_s[t]).PointwisePower(2));
            }
            return loss;
        }

        public double BackPropagate(List<Matrix<double>> xs,
                                  List<Matrix<double>> y_s,
                                  out Matrix<double> dWxh,
                                  out Matrix<double> dWhh,
                                  out Matrix<double> dWhy,
                                  out Matrix<double> dbh,
                                  out Matrix<double> dby) {
            dWxh = Matrix<double>.Build.Dense(inputDim, hiddenDim);
            dWhh = Matrix<double>.Build.Dense(hiddenDim, hiddenDim);
            dWhy = Matrix<double>.Build.Dense(hiddenDim, outputDim);

            dbh = Matrix<double>.Build.Dense(1, hiddenDim);
            dby = Matrix<double>.Build.Dense(1, outputDim);

            Matrix<double> dhnext = Matrix<double>.Build.Dense(1, hiddenDim);

            List<Matrix<double>> hs;
            List<Matrix<double>> ys;
            double loss = FeedForward(xs, y_s, out hs, out ys);
            for (int t = xs.Count - 1; t >= 0; t--) {
                Matrix<double> d = ((1 - y_s[t]).PointwiseDivide((1 - ys[t])) - y_s[t].PointwiseDivide(ys[t])).PointwiseMultiply(ys[t].PointwiseMultiply(1 - ys[t]));
                // Matrix<double> d = ys[t] - y_s[t];
                dby += d;
                dWhy += hs[t].Transpose() * d;
                
                d = (hs[t].PointwiseMultiply(1-hs[t])).PointwiseMultiply(d * Why.Transpose() + dhnext);

                dbh += d;
                dWxh += xs[t].Transpose() * d;
                if (t != 0) {
                    dWhh += hs[t - 1].Transpose() * d;
                }
                dhnext = d * Whh;
            }
            return loss;
        }

        public double Train(List<Matrix<double>> xs, List<Matrix<double>> y_s, double eta) {
            Matrix<double> dWxh;
            Matrix<double> dWhh;
            Matrix<double> dWhy;
            Matrix<double> dbh;
            Matrix<double> dby;
            double loss = BackPropagate(xs, y_s, out dWxh, out dWhh, out dWhy, out dbh, out dby);
            Wxh -= eta * dWxh;
            Whh -= eta * dWhh;
            Why -= eta * dWhy;
            bh -= eta * dbh;
            by -= eta * dby;
            return loss;
        }

        double ReduceSum(Matrix<double> m) {
            double ret = 0;
            for (int r = 0; r < m.RowCount; r++) {
                for (int c = 0; c < m.ColumnCount; c++) {
                    ret += m[r, c];
                }
            }
            return ret;
        }

        Matrix<double> Sigmoid(Matrix<double> m) {
            Matrix<double> tmp = -m;
            tmp = tmp.PointwiseExp();
            return 1.0 / (1.0 + tmp);
        }

    
    }
}
