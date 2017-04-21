using UnityEngine;
using System.Collections;

public class IIRFilter {
    float[] alpha;
    float[] beta;

    float[] filtered_data;
    float[] data;

    public IIRFilter(float[] alpha, float[] beta) {
        this.alpha = alpha;
        this.beta = beta;
        this.filtered_data = new float[2];
        this.data = new float[2];
    }

    public float Filter(float x) {
        float filtered_x = alpha[0] * (x * beta[0] +
                                data[1] * beta[1] +
                                data[0] * beta[2] -
                                filtered_data[1] * alpha[1] -
                                filtered_data[0] * alpha[2]);

        data[0] = data[1];
        data[1] = x;
        filtered_data[0] = filtered_data[1];
        filtered_data[1] = filtered_x;
        return filtered_x;
    }
}
