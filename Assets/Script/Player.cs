using UnityEngine;
using System.Collections.Generic;
using Model;
using MathNet.Numerics.LinearAlgebra;

public class Player : MonoBehaviour {
    public static Player Instance { get; set; }

    public GameObject eye;

    public float speed = 5;
    public float orientationRadius = 0.5f;
    public float positionSensitivity;

    public bool Training = false;

    IIRFilter low0Hz0, low0Hz1, low0Hz2, low0Hz3, low0Hz4;
    IIRFilter low5Hz0, low5Hz1, low5Hz2, low5Hz3, low5Hz4;
    //IIRFilter high1Hz;

    void Awake() {
        Instance = this;
    }

    // Use this for initialization
    void Start() {
        if (!Input.gyro.enabled) {
            Input.gyro.enabled = true;
        }


        low0Hz0 = new IIRFilter(new float[] { 1f, -1.979133761292768f, 0.979521463540373f }, new float[] { 0.000086384997973502f, 0.000172769995947004f, 0.000086384997973502f });
        low0Hz1 = new IIRFilter(new float[] { 1f, -1.979133761292768f, 0.979521463540373f }, new float[] { 0.000086384997973502f, 0.000172769995947004f, 0.000086384997973502f });
        low0Hz2 = new IIRFilter(new float[] { 1f, -1.979133761292768f, 0.979521463540373f }, new float[] { 0.000086384997973502f, 0.000172769995947004f, 0.000086384997973502f });
        low0Hz3 = new IIRFilter(new float[] { 1f, -1.979133761292768f, 0.979521463540373f }, new float[] { 0.000086384997973502f, 0.000172769995947004f, 0.000086384997973502f });
        low0Hz4 = new IIRFilter(new float[] { 1f, -1.979133761292768f, 0.979521463540373f }, new float[] { 0.000086384997973502f, 0.000172769995947004f, 0.000086384997973502f });


        low5Hz0 = new IIRFilter(new float[] { 1f, -1.80898117793047f, 0.827224480562408f }, new float[] { 0.095465967120306f, -0.172688631608676f, 0.095465967120306f });
        low5Hz1 = new IIRFilter(new float[] { 1f, -1.80898117793047f, 0.827224480562408f }, new float[] { 0.095465967120306f, -0.172688631608676f, 0.095465967120306f });
        low5Hz2 = new IIRFilter(new float[] { 1f, -1.80898117793047f, 0.827224480562408f }, new float[] { 0.095465967120306f, -0.172688631608676f, 0.095465967120306f });
        low5Hz3 = new IIRFilter(new float[] { 1f, -1.80898117793047f, 0.827224480562408f }, new float[] { 0.095465967120306f, -0.172688631608676f, 0.095465967120306f });
        low5Hz4 = new IIRFilter(new float[] { 1f, -1.80898117793047f, 0.827224480562408f }, new float[] { 0.095465967120306f, -0.172688631608676f, 0.095465967120306f });

        // high1Hz = new IIRFilter(new float[] { 1f, -1.905384612118461f, 0.910092542787947f }, new float[] { 0.953986986993339f, -1.907503180919730f, 0.953986986993339f });
    }

    // Update is called once per frame
    void FixedUpdate() {
        PositionUpdate();
        OrientationUpdate();
    }

    float Normalize(float x, float min, float max) {
        return (x - min) / (max - min);
    }

    Vector2 preTrainDirection = Vector2.one;
    void PositionUpdate() {
        Vector3 userAccleration = eye.transform.rotation * Input.gyro.userAcceleration;
        userAccleration = eye.transform.InverseTransformDirection(userAccleration);
        
        float x = userAccleration.x;
        float y = userAccleration.y;
        float z = userAccleration.z;
        x = x - low0Hz0.Filter(x);
        y = y - low0Hz1.Filter(y);
        z = z - low0Hz2.Filter(z);

        //x = Normalize(x, -0.5f, 0.5f);
        //y = Normalize(y, -0.5f, 0.5f);
        //z = Normalize(z, -0.5f, 0.5f);

        inputs.AddLast((new Vector3(x, y, z)) * 30);

        x = Joystick.Instance.InputDirection.x;
        y = Joystick.Instance.InputDirection.y;

        targets.AddLast(new Vector2(x, y));

        if (inputs.Count > seqLen) {
            inputs.RemoveFirst();
        }
        if (targets.Count > seqLen) {
            targets.RemoveFirst();
        }

        if (Training) {
            if (Joystick.Instance.InputDirection != Vector2.zero) {
                Vector3 direction = new Vector3(-Joystick.Instance.InputDirection.x, 0, Joystick.Instance.InputDirection.y);
                direction = eye.transform.TransformDirection(direction);
                GetComponent<Rigidbody>().transform.position += new Vector3(direction.x, 0, direction.z) * speed;
            }
            Train();
        } else {
            Test(); // move the player automatically
        }
    }

    
    void OrientationUpdate() {
        eye.transform.rotation = Input.gyro.attitude;
        eye.transform.rotation = Quaternion.FromToRotation(new Vector3(0, 0, 1), new Vector3(0, -1, 0)) * eye.transform.rotation;
        Vector3 orientation = eye.transform.rotation * new Vector3(0, 0, 1);
        eye.transform.localPosition = Vector3.Normalize(orientation) * orientationRadius;
    }

    // RNN Parameters, dim for dimension, 
    const int seqLen = 20;
    const int inputDim = 3;
    const int hiddenDim = 15;
    const int outputDim = 2;

    // generate a model 
    GRU rnn = new GRU(inputDim, hiddenDim, outputDim);

    // inputs, MEMS data
    LinkedList<Vector3> inputs = new LinkedList<Vector3>();

    // outputs, joystick
    LinkedList<Vector2> targets = new LinkedList<Vector2>();

    // training loss
    double loss = 0;

    void Train() {
        // build input and output
        List<Matrix<double>> xs = new List<Matrix<double>>();
        List<Matrix<double>> y_s = new List<Matrix<double>>();

        for (LinkedListNode<Vector3> i = inputs.First; i != inputs.Last.Next; i = i.Next) {
            Matrix<double> x = Matrix<double>.Build.Dense(1, inputDim);
            x[0, 0] = i.Value.x;
            x[0, 1] = i.Value.y;
            x[0, 2] = i.Value.z;
            xs.Add(x);
        }

        for (LinkedListNode<Vector2> i = targets.First; i != targets.Last.Next; i = i.Next) {
            Matrix<double> y_ = Matrix<double>.Build.Dense(1, outputDim);
            y_[0, 0] = i.Value.x * 0.5 + 0.5;
            y_[0, 1] = i.Value.y * 0.5 + 0.5;
            y_s.Add(y_);
        }

        // train
        loss = rnn.Train(xs, y_s, 1e-2);
    }

    void Test() {
        // build input
        List<Matrix<double>> xs = new List<Matrix<double>>();

        for (LinkedListNode<Vector3> i = inputs.First; i != inputs.Last.Next; i = i.Next) {
            Matrix<double> x = Matrix<double>.Build.Dense(1, inputDim);
            x[0, 0] = i.Value.x;
            x[0, 1] = i.Value.y;
            x[0, 2] = i.Value.z;
            xs.Add(x);
        }

        // predict
        List<Matrix<double>> ys;
        rnn.Predict(xs, out ys);

        // the least recently result
        Matrix<double> y = ys[ys.Count - 1];

        // this is not only the orientation, magnitude is aslo involved
        Vector2 inputDirection = new Vector2((float)(y[0, 0] - 0.5) * 2, (float)(y[0, 1] - 0.5) * 2);

        // make the predict result visible on the joystick 
        Joystick.Instance.SetDirection(inputDirection);
        
        // if there is a movement, move the player
        if (Joystick.Instance.InputDirection != Vector2.zero) {
            Vector3 direction = new Vector3(-Joystick.Instance.InputDirection.x, 0, Joystick.Instance.InputDirection.y);
            direction = eye.transform.TransformDirection(direction);
            GetComponent<Rigidbody>().transform.position += new Vector3(direction.x, 0, direction.z) * speed;
        }
    }

    void OnGUI() {
        GUIStyle style = new GUIStyle();
        style.normal.textColor = Color.red;
        style.fontSize = 100;
        GUI.Label(new Rect(100, 100, 500, 200), "loss " + loss.ToString("F3"), style);
        GUI.Label(new Rect(100, 250, 500, 200), "real " + inputs.Last.Value.ToString("F1"), style);
        GUI.Label(new Rect(100, 400, 500, 200), "op" + Joystick.Instance.InputDirection.ToString("F1"), style);
    }
}
