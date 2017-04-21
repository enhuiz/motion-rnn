using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

namespace Model {
    public class GRU {
        int inputDim;
        int outputDim;
        int hiddenDim;

        Matrix<double> Wxc;
        Matrix<double> Wxr;
        Matrix<double> Wxz;

        Matrix<double> Whc;
        Matrix<double> Whr;
        Matrix<double> Whz;

        Matrix<double> Why;

        Matrix<double> bc;
        Matrix<double> br;
        Matrix<double> bz;

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

        public GRU(int inputDim, int hiddenDim, int outputDim) {
            this.inputDim = inputDim;
            this.outputDim = outputDim;
            this.hiddenDim = hiddenDim;


            Wxc = RandomMatrix(inputDim, hiddenDim);
            Wxr = RandomMatrix(inputDim, hiddenDim);
            Wxz = RandomMatrix(inputDim, hiddenDim);

            Whc = RandomMatrix(hiddenDim, hiddenDim);
            Whr = RandomMatrix(hiddenDim, hiddenDim);
            Whz = RandomMatrix(hiddenDim, hiddenDim);

            Why = RandomMatrix(hiddenDim, outputDim);

            bc = Matrix<double>.Build.Dense(1, hiddenDim, 0.1);
            br = Matrix<double>.Build.Dense(1, hiddenDim, 0.1);
            bz = Matrix<double>.Build.Dense(1, hiddenDim, 0.1);

            by = Matrix<double>.Build.Dense(1, outputDim, 0.1);
        }

        public void Predict(List<Matrix<double>> xs,
                                out List<Matrix<double>> ys) {
            List<Matrix<double>> cs = new List<Matrix<double>>();
            List<Matrix<double>> rs = new List<Matrix<double>>();
            List<Matrix<double>> zs = new List<Matrix<double>>();
            List<Matrix<double>> hs = new List<Matrix<double>>();
            ys = new List<Matrix<double>>();

            for (int t = 0; t < xs.Count; t++) {
                if (t == 0) {
                    rs.Add(Sigmoid(xs[t] * Wxr + br));
                    zs.Add(Sigmoid(xs[t] * Wxz + bz));
                    cs.Add(Tanh(xs[t] * Wxc + bc));
                    hs.Add(cs[t].PointwiseMultiply(zs[t]));
                } else {
                    rs.Add(Sigmoid(xs[t] * Wxr + hs[t - 1] * Whr + br));
                    zs.Add(Sigmoid(xs[t] * Wxz + hs[t - 1] * Whz + bz));
                    cs.Add(Tanh(xs[t] * Wxc + rs[t - 1].PointwiseMultiply(hs[t - 1]) * Whc + bc));
                    hs.Add(cs[t].PointwiseMultiply(zs[t]) + hs[t - 1].PointwiseMultiply(1 - zs[t]));
                }
                ys.Add(Sigmoid(hs[t] * Why + by));
            }
        }

        public double FeedForward(List<Matrix<double>> xs,
                                List<Matrix<double>> y_s,
                                out List<Matrix<double>> cs,
                                out List<Matrix<double>> rs,
                                out List<Matrix<double>> zs,
                                out List<Matrix<double>> hs,
                                out List<Matrix<double>> ys) {

            cs = new List<Matrix<double>>();
            rs = new List<Matrix<double>>();
            zs = new List<Matrix<double>>();
            hs = new List<Matrix<double>>();
            ys = new List<Matrix<double>>();

            double loss = 0;

            for (int t = 0; t < xs.Count; t++) {
                if (t == 0) {
                    rs.Add(Sigmoid(xs[t] * Wxr + br));
                    zs.Add(Sigmoid(xs[t] * Wxz + bz));
                    cs.Add(Tanh(xs[t] * Wxc + bc));
                    hs.Add(cs[t].PointwiseMultiply(zs[t]));
                } else {
                    rs.Add(Sigmoid(xs[t] * Wxr + hs[t - 1] * Whr + br));
                    zs.Add(Sigmoid(xs[t] * Wxz + hs[t - 1] * Whz + bz));
                    cs.Add(Tanh(xs[t] * Wxc + rs[t - 1].PointwiseMultiply(hs[t - 1]) * Whc + bc));
                    hs.Add(cs[t].PointwiseMultiply(zs[t]) + hs[t - 1].PointwiseMultiply(1 - zs[t]));
                }
                ys.Add(Sigmoid(hs[t] * Why + by));
                loss += ReduceSum((ys[t] - y_s[t]).PointwisePower(2));
            }
            return loss;
        }

        public double BackPropagate(List<Matrix<double>> xs,
                                  List<Matrix<double>> y_s,
                                  out Matrix<double> dWxc,
                                  out Matrix<double> dWxr,
                                  out Matrix<double> dWxz,
                                  out Matrix<double> dWhc,
                                  out Matrix<double> dWhr,
                                  out Matrix<double> dWhz,
                                  out Matrix<double> dWhy,
                                  out Matrix<double> dbc,
                                  out Matrix<double> dbr,
                                  out Matrix<double> dbz,
                                  out Matrix<double> dby) {

            dWxc = Matrix<double>.Build.Dense(inputDim, hiddenDim);
            dWxr = Matrix<double>.Build.Dense(inputDim, hiddenDim);
            dWxz = Matrix<double>.Build.Dense(inputDim, hiddenDim);

            dWhc = Matrix<double>.Build.Dense(hiddenDim, hiddenDim);
            dWhr = Matrix<double>.Build.Dense(hiddenDim, hiddenDim);
            dWhz = Matrix<double>.Build.Dense(hiddenDim, hiddenDim);

            dWhy = Matrix<double>.Build.Dense(hiddenDim, outputDim);

            dbc = Matrix<double>.Build.Dense(1, hiddenDim);
            dbr = Matrix<double>.Build.Dense(1, hiddenDim);
            dbz = Matrix<double>.Build.Dense(1, hiddenDim);

            dby = Matrix<double>.Build.Dense(1, outputDim);

            Matrix<double> dhnext = Matrix<double>.Build.Dense(1, hiddenDim);
            Matrix<double> dcrawnext = Matrix<double>.Build.Dense(1, hiddenDim);
            Matrix<double> drrawnext = Matrix<double>.Build.Dense(1, hiddenDim);
            Matrix<double> dzrawnext = Matrix<double>.Build.Dense(1, hiddenDim);

            List<Matrix<double>> cs;
            List<Matrix<double>> rs;
            List<Matrix<double>> zs;
            List<Matrix<double>> hs;
            List<Matrix<double>> ys;

            double loss = FeedForward(xs, y_s, out cs, out rs, out zs, out hs, out ys);
            for (int t = xs.Count - 1; t >= 0; t--) {
                Matrix<double> dy = ((1 - y_s[t]).PointwiseDivide((1 - ys[t])) - y_s[t].PointwiseDivide(ys[t])).PointwiseMultiply(ys[t].PointwiseMultiply(1 - ys[t]));
                //Matrix<double> dy = (ys[t] - y_s[t]).PointwiseMultiply(ys[t].PointwiseMultiply(1 - ys[t]));
                //Matrix<double> dy = ys[t] - y_s[t];
                dby += dy;
                dWhy += hs[t].Transpose() * dy;
                Matrix<double> dh;
                Matrix<double> dhc;
                if (t == xs.Count - 1) {
                    dh = dhnext;
                    dhc = dhnext;
                } else {
                    dh = dhnext.PointwiseMultiply(1 - zs[t + 1]);
                    dhc = rs[t + 1].PointwiseMultiply(dcrawnext * Whc);
                }
                Matrix<double> dhr = drrawnext * Whr.Transpose();
                Matrix<double> dhz = dzrawnext * Whz.Transpose();
                Matrix<double> dhy = dy * Why.Transpose();

                dh += dhc + dhr + dhz + dhy;

                Matrix<double> dc = dh.PointwiseMultiply(zs[t]).PointwiseMultiply(1 - cs[t].PointwisePower(2));
                Matrix<double> dr;
                Matrix<double> dz;
                if (t == 0) {
                    dr = Matrix<double>.Build.Dense(1, hiddenDim);
                    dz = Matrix<double>.Build.Dense(1, hiddenDim);
                } else {
                    dr = hs[t - 1].PointwiseMultiply(dc * Whc.Transpose()).PointwiseMultiply(rs[t].PointwiseMultiply((1 - rs[t])));
                    dz = dh.PointwiseMultiply(cs[t] - hs[t - 1]).PointwiseMultiply(zs[t].PointwiseMultiply((1 - zs[t])));
                }

                dWxc += xs[t].Transpose() * dc;
                dWxr += xs[t].Transpose() * dr;
                dWxz += xs[t].Transpose() * dz;

                if (t != 0) {
                    dWhc += hs[t - 1].Transpose() * dc;
                    dWhr += hs[t - 1].Transpose() * dr;
                    dWhz += (hs[t - 1].PointwiseMultiply(rs[t])).Transpose() * dz;
                }

                dbc += dc;
                dbr += dr;
                dbz += dz;

                dhnext = dh;
                dcrawnext = dc;
                drrawnext = dr;
                dzrawnext = dz;
            }
            return loss;
        }

        public double Train(List<Matrix<double>> xs, List<Matrix<double>> y_s, double eta) {
            Matrix<double> dWxc;
            Matrix<double> dWxr;
            Matrix<double> dWxz;

            Matrix<double> dWhc;
            Matrix<double> dWhr;
            Matrix<double> dWhz;

            Matrix<double> dWhy;

            Matrix<double> dbc;
            Matrix<double> dbr;
            Matrix<double> dbz;

            Matrix<double> dby;

            double loss = BackPropagate(xs, y_s,
                                        out dWxc,
                                        out dWxr,
                                        out dWxz,
                                        out dWhc,
                                        out dWhr,
                                        out dWhz,
                                        out dWhy,
                                        out dbc,
                                        out dbr,
                                        out dbz,
                                        out dby);

            Wxc -= eta * dWxc;
            Wxr -= eta * dWxr;
            Wxz -= eta * dWxz;

            Whc -= eta * dWhc;
            Whr -= eta * dWhr;
            Whz -= eta * dWhz;

            Why -= eta * dWhy;

            bc -= eta * dbc;
            br -= eta * dbr;
            bz -= eta * dbz;

            dby -= eta * dby;
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

        Matrix<double> Tanh(Matrix<double> m) {
            Matrix<double> negm = -m;
            return (m.PointwiseExp() - negm.PointwiseExp()).PointwiseDivide((m.PointwiseExp() + negm.PointwiseExp()));
        }
    }
}
