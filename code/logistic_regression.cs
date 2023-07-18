// https://jamesmccaffrey.wordpress.com/2023/02/16/logistic-regression-using-raw-c-2/

Console.WriteLine("\nLogistic regression raw C# demo ");

// 0. get ready
Random rnd = new Random(0);

// 1. load train data
Console.WriteLine("\nLoading People data ");
int[] cols = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
double[][] xTrain = Extract(FeedTrainData(), cols[1..]);
double[][] yTrain = Extract(FeedTrainData(), cols[0..1]);
// 1b. load test data
double[][] xTest = Extract(FeedTestData(), cols[1..]);
double[][] yTest = Extract(FeedTestData(), cols[0..1]);

// 2. create model
Console.WriteLine("\nCreating logistic regression model ");
var lrnRate = 0.005;
var maxEpochs = 10000;
double[] wts = InitializeWeights(8, -0.01, 0.01, rnd);

// 3. train model
Console.WriteLine($"\nTraining using SGD with lrnRate = {lrnRate}");
TrainModel(xTrain, yTrain, wts, maxEpochs, lrnRate, rnd);
Console.WriteLine("Done ");

// 4. evaluate model 
Console.WriteLine("\nEvaluating trained model ");
double accTrain = Accuracy(wts, xTrain, yTrain);
Console.WriteLine($"Accuracy on train data: {accTrain:F4}");
double accTest = Accuracy(wts, xTest, yTest);
Console.WriteLine($"Accuracy on test data: {accTest:F4}");

// 5. use model
Console.WriteLine("\n[33, Nebraska, $50,000, moderate]: ");
double[] inpt = new double[] { 0.33, 0, 1, 0, 0.50000, 0, 1, 0 };
double prediction = ComputeOutput(wts, inpt);
Console.WriteLine($"p-val = {prediction:F4}");
Console.WriteLine((prediction < 0.5) ? "class 0 (male) " : "class 1 (female) ");

Console.WriteLine("\nEnd logistic regression C# demo ");
Console.ReadLine();

double[] InitializeWeights(int inputSize, double lo, double hi, Random rnd)
{
    double[] wts = new double[inputSize + 1];
    for (int i = 0; i < inputSize; ++i)
        wts[i] = (hi - lo) * rnd.NextDouble() + lo;
    return wts;
}
void TrainModel(double[][] xTrain, double[][] yTrain, double[] wts, int maxEpochs, double lrnRate, Random rnd)
{
    int[] indices = new int[xTrain.Length];
    for (int i = 0; i < indices.Length; ++i)
        indices[i] = i;

    // Initialize the weights with a specified range
    double lo = -0.01;
    double hi = 0.01;
    for (int i = 0; i < wts.Length - 1; ++i)
        wts[i] = (hi - lo) * rnd.NextDouble() + lo;

    for (int epoch = 0; epoch < maxEpochs; ++epoch)
    {
        Shuffle(indices, rnd);
        for (int ii = 0; ii < indices.Length; ++ii)
        {
            int i = indices[ii];
            double[] x = xTrain[i];
            double y = yTrain[i][0];  // target 0 or 1
            double p = ComputeOutput(wts, x);

            // update all wts and the bias
            for (int j = 0; j < wts.Length - 1; ++j)
                wts[j] += lrnRate * x[j] * (y - p);

            // update the bias
            wts[wts.Length - 1] += lrnRate * (y - p);
        }

        if (epoch % 1000 == 0)
        {
            double loss = MSELoss(wts, xTrain, yTrain);
            Console.WriteLine($"epoch = {epoch,4}  |  loss = {loss:F4}");
        }
    }
}
static void Shuffle(int[] arr, Random rnd)
{
    // Fisher-Yates algorithm
    int n = arr.Length;
    for (int i = 0; i < n; ++i)
    {
        int ri = rnd.Next(i, n);  // random index
        int tmp = arr[ri];
        arr[ri] = arr[i];
        arr[i] = tmp;
    }
}
static double ComputeOutput(double[] w, double[] x)
{
    double z = 0.0;
    for (int i = 0; i < w.Length - 1; ++i)
        z += w[i] * x[i];
    z += w[^1];
    double p = 1.0 / (1.0 + Math.Exp(-z));
    return p;
}
static double Accuracy(double[] w, double[][] dataX, double[][] dataY)
{
    int nCorrect = 0; int nWrong = 0;
    for (int i = 0; i < dataX.Length; ++i)
    {
        double[] x = dataX[i];
        int y = (int)dataY[i][0];
        double p = ComputeOutput(w, x);
        if ((y == 0 && p < 0.5) ||
          (y == 1 && p >= 0.5))
            nCorrect += 1;
        else
            nWrong += 1;
    }
    double acc = (nCorrect * 1.0) / (nCorrect + nWrong);
    return acc;
}
static double MSELoss(double[] w, double[][] dataX, double[][] dataY)
{
    double sum = 0.0;
    for (int i = 0; i < dataX.Length; ++i)
    {
        double[] x = dataX[i];
        double y = dataY[i][0];
        double p = ComputeOutput(w, x);
        sum += (y - p) * (y - p);
    }
    double mse = sum / dataX.Length;
    return mse;
}
static double[][] MatCreate(int rows, int cols)
{
    double[][] result = new double[rows][];
    for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
    return result;
}
static double[][] Extract(double[][] mat, int[] cols)
{
    int nRows = mat.Length;
    int nCols = cols.Length;
    double[][] result = MatCreate(nRows, nCols);
    for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)  // idx into src cols
        {
            int srcCol = cols[j];
            int destCol = j;
            result[i][destCol] = mat[i][srcCol];
        }
    return result;
}
double[][] FeedTrainData()
{
    double[][] trainData = new double[][]
    {
    new double[] {1,0.24,1,0,0,0.2950,0,0,1 },  // 1
    new double[] {0,0.39,0,0,1,0.5120,0,1,0 },  // 2
    new double[] {1,0.63,0,1,0,0.7580,1,0,0 },  // 3
    new double[] {0,0.36,1,0,0,0.4450,0,1,0 },  // 4
    new double[] {1,0.27,0,1,0,0.2860,0,0,1 },  // 5
    new double[] {1,0.50,0,1,0,0.5650,0,1,0 },  // 6
    new double[] {1,0.50,0,0,1,0.5500,0,1,0 },  // 7
    new double[] {0,0.19,0,0,1,0.3270,1,0,0 },  // 8
    new double[] {1,0.22,0,1,0,0.2770,0,1,0 },  // 9
    new double[] {0,0.39,0,0,1,0.4710,0,0,1 },  // 10
    new double[] {1,0.34,1,0,0,0.3940,0,1,0 },  // 11
    new double[] {0,0.22,1,0,0,0.3350,1,0,0 },  // 12
    new double[] {1,0.35,0,0,1,0.3520,0,0,1 },  // 13
    new double[] {0,0.33,0,1,0,0.4640,0,1,0 },  // 14
    new double[] {1,0.45,0,1,0,0.5410,0,1,0 },  // 15
    new double[] {1,0.42,0,1,0,0.5070,0,1,0 },  // 16
    new double[] {0,0.33,0,1,0,0.4680,0,1,0 },  // 17
    new double[] {1,0.25,0,0,1,0.3000,0,1,0 },  // 18
    new double[] {0,0.31,0,1,0,0.4640,1,0,0 },  // 19
    new double[] {1,0.27,1,0,0,0.3250,0,0,1 },  // 20
    new double[] {1,0.48,1,0,0,0.5400,0,1,0 },  // 21
    new double[] {0,0.64,0,1,0,0.7130,0,0,1 },  // 22
    new double[] {1,0.61,0,1,0,0.7240,1,0,0 },  // 23
    new double[] {1,0.54,0,0,1,0.6100,1,0,0 },  // 24
    new double[] {1,0.29,1,0,0,0.3630,1,0,0 },  // 25
    new double[] {1,0.50,0,0,1,0.5500,0,1,0 },  // 26
    new double[] {1,0.55,0,0,1,0.6250,1,0,0 },  // 27
    new double[] {1,0.40,1,0,0,0.5240,1,0,0 },  // 28
    new double[] {1,0.22,1,0,0,0.2360,0,0,1 },  // 29
    new double[] {1,0.68,0,1,0,0.7840,1,0,0 },  // 30
    new double[] {0,0.60,1,0,0,0.7170,0,0,1 },  // 31
    new double[] {0,0.34,0,0,1,0.4650,0,1,0 },  // 32
    new double[] {0,0.25,0,0,1,0.3710,1,0,0 },  // 33
    new double[] {0,0.31,0,1,0,0.4890,0,1,0 },  // 34
    new double[] {1,0.43,0,0,1,0.4800,0,1,0 },  // 35
    new double[] {1,0.58,0,1,0,0.6540,0,0,1 },  // 36
    new double[] {0,0.55,0,1,0,0.6070,0,0,1 },  // 37
    new double[] {0,0.43,0,1,0,0.5110,0,1,0 },  // 38
    new double[] {0,0.43,0,0,1,0.5320,0,1,0 },  // 39
    new double[] {0,0.21,1,0,0,0.3720,1,0,0 },  // 40
    new double[] {1,0.55,0,0,1,0.6460,1,0,0 },  // 41
    new double[] {1,0.64,0,1,0,0.7480,1,0,0 },  // 42
    new double[] {0,0.41,1,0,0,0.5880,0,1,0 },  // 43
    new double[] {1,0.64,0,0,1,0.7270,1,0,0 },  // 44
    new double[] {0,0.56,0,0,1,0.6660,0,0,1 },  // 45
    new double[] {1,0.31,0,0,1,0.3600,0,1,0 },  // 46
    new double[] {0,0.65,0,0,1,0.7010,0,0,1 },  // 47
    new double[] {1,0.55,0,0,1,0.6430,1,0,0 },  // 48
    new double[] {0,0.25,1,0,0,0.4030,1,0,0 },  // 49
    new double[] {1,0.46,0,0,1,0.5100,0,1,0 },  // 50
    new double[] {0,0.36,1,0,0,0.5350,1,0,0 },  // 51
    new double[] {1,0.52,0,1,0,0.5810,0,1,0 },  // 52
    new double[] {1,0.61,0,0,1,0.6790,1,0,0 },  // 53
    new double[] {1,0.57,0,0,1,0.6570,1,0,0 },  // 54
    new double[] {0,0.46,0,1,0,0.5260,0,1,0 },  // 55
    new double[] {0,0.62,1,0,0,0.6680,0,0,1 },  // 56
    new double[] {1,0.55,0,0,1,0.6270,1,0,0 },  // 57
    new double[] {0,0.22,0,0,1,0.2770,0,1,0 },  // 58
    new double[] {0,0.50,1,0,0,0.6290,1,0,0 },  // 59
    new double[] {0,0.32,0,1,0,0.4180,0,1,0 },  // 60
    new double[] {0,0.21,0,0,1,0.3560,1,0,0 },  // 61
    new double[] {1,0.44,0,1,0,0.5200,0,1,0 },  // 62
    new double[] {1,0.46,0,1,0,0.5170,0,1,0 },  // 63
    new double[] {1,0.62,0,1,0,0.6970,1,0,0 },  // 64
    new double[] {1,0.57,0,1,0,0.6640,1,0,0 },  // 65
    new double[] {0,0.67,0,0,1,0.7580,0,0,1 },  // 66
    new double[] {1,0.29,1,0,0,0.3430,0,0,1 },  // 67
    new double[] {1,0.53,1,0,0,0.6010,1,0,0 },  // 68
    new double[] {0,0.44,1,0,0,0.5480,0,1,0 },  // 69
    new double[] {1,0.46,0,1,0,0.5230,0,1,0 },  // 70
    new double[] {0,0.20,0,1,0,0.3010,0,1,0 },  // 71
    new double[] {0,0.38,1,0,0,0.5350,0,1,0 },  // 72
    new double[] {1,0.50,0,1,0,0.5860,0,1,0 },  // 73
    new double[] {1,0.33,0,1,0,0.4250,0,1,0 },  // 74
    new double[] {0,0.33,0,1,0,0.3930,0,1,0 },  // 75
    new double[] {1,0.26,0,1,0,0.4040,1,0,0 },  // 76
    new double[] {1,0.58,1,0,0,0.7070,1,0,0 },  // 77
    new double[] {1,0.43,0,0,1,0.4800,0,1,0 },  // 78
    new double[] {0,0.46,1,0,0,0.6440,1,0,0 },  // 79
    new double[] {1,0.60,1,0,0,0.7170,1,0,0 },  // 80
    new double[] {0,0.42,1,0,0,0.4890,0,1,0 },  // 81
    new double[] {0,0.56,0,0,1,0.5640,0,0,1 },  // 82
    new double[] {0,0.62,0,1,0,0.6630,0,0,1 },  // 83
    new double[] {0,0.50,1,0,0,0.6480,0,1,0 },  // 84
    new double[] {1,0.47,0,0,1,0.5200,0,1,0 },  // 85
    new double[] {0,0.67,0,1,0,0.8040,0,0,1 },  // 86
    new double[] {0,0.40,0,0,1,0.5040,0,1,0 },  // 87
    new double[] {1,0.42,0,1,0,0.4840,0,1,0 },  // 88
    new double[] {1,0.64,1,0,0,0.7200,1,0,0 },  // 89
    new double[] {0,0.47,1,0,0,0.5870,0,0,1 },  // 90
    new double[] {1,0.45,0,1,0,0.5280,0,1,0 },  // 91
    new double[] {0,0.25,0,0,1,0.4090,1,0,0 },  // 92
    new double[] {1,0.38,1,0,0,0.4840,1,0,0 },  // 93
    new double[] {1,0.55,0,0,1,0.6000,0,1,0 },  // 94
    new double[] {0,0.44,1,0,0,0.6060,0,1,0 },  // 95
    new double[] {1,0.33,1,0,0,0.4100,0,1,0 },  // 96
    new double[] {1,0.34,0,0,1,0.3900,0,1,0 },  // 97
    new double[] {1,0.27,0,1,0,0.3370,0,0,1 },  // 98
    new double[] {1,0.32,0,1,0,0.4070,0,1,0 },  // 99
    new double[] {1,0.42,0,0,1,0.4700,0,1,0 },  // 100
    new double[] {0,0.24,0,0,1,0.4030,1,0,0 },  // 101
    new double[] {1,0.42,0,1,0,0.5030,0,1,0 },  // 102
    new double[] {1,0.25,0,0,1,0.2800,0,0,1 },  // 103
    new double[] {1,0.51,0,1,0,0.5800,0,1,0 },  // 104
    new double[] {0,0.55,0,1,0,0.6350,0,0,1 },  // 105
    new double[] {1,0.44,1,0,0,0.4780,0,0,1 },  // 106
    new double[] {0,0.18,1,0,0,0.3980,1,0,0 },  // 107
    new double[] {0,0.67,0,1,0,0.7160,0,0,1 },  // 108
    new double[] {1,0.45,0,0,1,0.5000,0,1,0 },  // 109
    new double[] {1,0.48,1,0,0,0.5580,0,1,0 },  // 110
    new double[] {0,0.25,0,1,0,0.3900,0,1,0 },  // 111
    new double[] {0,0.67,1,0,0,0.7830,0,1,0 },  // 112
    new double[] {1,0.37,0,0,1,0.4200,0,1,0 },  // 113
    new double[] {0,0.32,1,0,0,0.4270,0,1,0 },  // 114
    new double[] {1,0.48,1,0,0,0.5700,0,1,0 },  // 115
    new double[] {0,0.66,0,0,1,0.7500,0,0,1 },  // 116
    new double[] {1,0.61,1,0,0,0.7000,1,0,0 },  // 117
    new double[] {0,0.58,0,0,1,0.6890,0,1,0 },  // 118
    new double[] {1,0.19,1,0,0,0.2400,0,0,1 },  // 119
    new double[] {1,0.38,0,0,1,0.4300,0,1,0 },  // 120
    new double[] {0,0.27,1,0,0,0.3640,0,1,0 },  // 121
    new double[] {1,0.42,1,0,0,0.4800,0,1,0 },  // 122
    new double[] {1,0.60,1,0,0,0.7130,1,0,0 },  // 123
    new double[] {0,0.27,0,0,1,0.3480,1,0,0 },  // 124
    new double[] {1,0.29,0,1,0,0.3710,1,0,0 },  // 125
    new double[] {0,0.43,1,0,0,0.5670,0,1,0 },  // 126
    new double[] {1,0.48,1,0,0,0.5670,0,1,0 },  // 127
    new double[] {1,0.27,0,0,1,0.2940,0,0,1 },  // 128
    new double[] {0,0.44,1,0,0,0.5520,1,0,0 },  // 129
    new double[] {1,0.23,0,1,0,0.2630,0,0,1 },  // 130
    new double[] {0,0.36,0,1,0,0.5300,0,0,1 },  // 131
    new double[] {1,0.64,0,0,1,0.7250,1,0,0 },  // 132
    new double[] {1,0.29,0,0,1,0.3000,0,0,1 },  // 133
    new double[] {0,0.33,1,0,0,0.4930,0,1,0 },  // 134
    new double[] {0,0.66,0,1,0,0.7500,0,0,1 },  // 135
    new double[] {0,0.21,0,0,1,0.3430,1,0,0 },  // 136
    new double[] {1,0.27,1,0,0,0.3270,0,0,1 },  // 137
    new double[] {1,0.29,1,0,0,0.3180,0,0,1 },  // 138
    new double[] {0,0.31,1,0,0,0.4860,0,1,0 },  // 139
    new double[] {1,0.36,0,0,1,0.4100,0,1,0 },  // 140
    new double[] {1,0.49,0,1,0,0.5570,0,1,0 },  // 141
    new double[] {0,0.28,1,0,0,0.3840,1,0,0 },  // 142
    new double[] {0,0.43,0,0,1,0.5660,0,1,0 },  // 143
    new double[] {0,0.46,0,1,0,0.5880,0,1,0 },  // 144
    new double[] {1,0.57,1,0,0,0.6980,1,0,0 },  // 145
    new double[] {0,0.52,0,0,1,0.5940,0,1,0 },  // 146
    new double[] {0,0.31,0,0,1,0.4350,0,1,0 },  // 147
    new double[] {0,0.55,1,0,0,0.6200,0,0,1 },  // 148
    new double[] {1,0.50,1,0,0,0.5640,0,1,0 },  // 149
    new double[] {1,0.48,0,1,0,0.5590,0,1,0 },  // 150
    new double[] {0,0.22,0,0,1,0.3450,1,0,0 },  // 151
    new double[] {1,0.59,0,0,1,0.6670,1,0,0 },  // 152
    new double[] {1,0.34,1,0,0,0.4280,0,0,1 },  // 153
    new double[] {0,0.64,1,0,0,0.7720,0,0,1 },  // 154
    new double[] {1,0.29,0,0,1,0.3350,0,0,1 },  // 155
    new double[] {0,0.34,0,1,0,0.4320,0,1,0 },  // 156
    new double[] {0,0.61,1,0,0,0.7500,0,0,1 },  // 157
    new double[] {1,0.64,0,0,1,0.7110,1,0,0 },  // 158
    new double[] {0,0.29,1,0,0,0.4130,1,0,0 },  // 159
    new double[] {1,0.63,0,1,0,0.7060,1,0,0 },  // 160
    new double[] {0,0.29,0,1,0,0.4000,1,0,0 },  // 161
    new double[] {0,0.51,1,0,0,0.6270,0,1,0 },  // 162
    new double[] {0,0.24,0,0,1,0.3770,1,0,0 },  // 163
    new double[] {1,0.48,0,1,0,0.5750,0,1,0 },  // 164
    new double[] {1,0.18,1,0,0,0.2740,1,0,0 },  // 165
    new double[] {1,0.18,1,0,0,0.2030,0,0,1 },  // 166
    new double[] {1,0.33,0,1,0,0.3820,0,0,1 },  // 167
    new double[] {0,0.20,0,0,1,0.3480,1,0,0 },  // 168
    new double[] {1,0.29,0,0,1,0.3300,0,0,1 },  // 169
    new double[] {0,0.44,0,0,1,0.6300,1,0,0 },  // 170
    new double[] {0,0.65,0,0,1,0.8180,1,0,0 },  // 171
    new double[] {0,0.56,1,0,0,0.6370,0,0,1 },  // 172
    new double[] {0,0.52,0,0,1,0.5840,0,1,0 },  // 173
    new double[] {0,0.29,0,1,0,0.4860,1,0,0 },  // 174
    new double[] {0,0.47,0,1,0,0.5890,0,1,0 },  // 175
    new double[] {1,0.68,1,0,0,0.7260,0,0,1 },  // 176
    new double[] {1,0.31,0,0,1,0.3600,0,1,0 },  // 177
    new double[] {1,0.61,0,1,0,0.6250,0,0,1 },  // 178
    new double[] {1,0.19,0,1,0,0.2150,0,0,1 },  // 179
    new double[] {1,0.38,0,0,1,0.4300,0,1,0 },  // 180
    new double[] {0,0.26,1,0,0,0.4230,1,0,0 },  // 181
    new double[] {1,0.61,0,1,0,0.6740,1,0,0 },  // 182
    new double[] {1,0.40,1,0,0,0.4650,0,1,0 },  // 183
    new double[] {0,0.49,1,0,0,0.6520,0,1,0 },  // 184
    new double[] {1,0.56,1,0,0,0.6750,1,0,0 },  // 185
    new double[] {0,0.48,0,1,0,0.6600,0,1,0 },  // 186
    new double[] {1,0.52,1,0,0,0.5630,0,0,1 },  // 187
    new double[] {0,0.18,1,0,0,0.2980,1,0,0 },  // 188
    new double[] {0,0.56,0,0,1,0.5930,0,0,1 },  // 189
    new double[] {0,0.52,0,1,0,0.6440,0,1,0 },  // 190
    new double[] {0,0.18,0,1,0,0.2860,0,1,0 },  // 191
    new double[] {0,0.58,1,0,0,0.6620,0,0,1 },  // 192
    new double[] {0,0.39,0,1,0,0.5510,0,1,0 },  // 193
    new double[] {0,0.46,1,0,0,0.6290,0,1,0 },  // 194
    new double[] {0,0.40,0,1,0,0.4620,0,1,0 },  // 195
    new double[] {0,0.60,1,0,0,0.7270,0,0,1 },  // 196
    new double[] {1,0.36,0,1,0,0.4070,0,0,1 },  // 197
    new double[] {1,0.44,1,0,0,0.5230,0,1,0 },  // 198
    new double[] {1,0.28,1,0,0,0.3130,0,0,1 },  // 199
    new double[] {1,0.54,0,0,1,0.6260,1,0,0 }   // 200
    };
    return trainData;
}
double[][] FeedTestData()
{
    double[][] testData = new double[][]
    {
       new double[] { 0,0.51,1,0,0,0.6120,0,1,0 },  // 0
       new double[] { 0,0.32,0,1,0,0.4610,0,1,0 },  // 1
       new double[] { 1,0.55,1,0,0,0.6270,1,0,0 },  // 2
       new double[] { 1,0.25,0,0,1,0.2620,0,0,1 },  // 3
       new double[] { 1,0.33,0,0,1,0.3730,0,0,1 },  // 4
       new double[] { 0,0.29,0,1,0,0.4620,1,0,0 },  // 5
       new double[] { 1,0.65,1,0,0,0.7270,1,0,0 },  // 6
       new double[] { 0,0.43,0,1,0,0.5140,0,1,0 },  // 7
       new double[] { 0,0.54,0,1,0,0.6480,0,0,1 },  // 8
       new double[] { 1,0.61,0,1,0,0.7270,1,0,0 },  // 9
       new double[] { 1,0.52,0,1,0,0.6360,1,0,0 }, // 10
       new double[] { 1,0.30,0,1,0,0.3350,0,0,1 }, // 11
       new double[] { 1,0.29,1,0,0,0.3140,0,0,1 }, // 12
       new double[] { 0,0.47,0,0,1,0.5940,0,1,0 }, // 13
       new double[] { 1,0.39,0,1,0,0.4780,0,1,0 }, // 14
       new double[] { 1,0.47,0,0,1,0.5200,0,1,0 }, // 15
       new double[] { 0,0.49,1,0,0,0.5860,0,1,0 }, // 16
       new double[] { 0,0.63,0,0,1,0.6740,0,0,1 }, // 17
       new double[] { 0,0.30,1,0,0,0.3920,1,0,0 }, // 18
       new double[] { 0,0.61,0,0,1,0.6960,0,0,1 }, // 19
       new double[] { 0,0.47,0,0,1,0.5870,0,1,0 }, // 20
       new double[] { 1,0.30,0,0,1,0.3450,0,0,1 }, // 21
       new double[] { 0,0.51,0,0,1,0.5800,0,1,0 }, // 22
       new double[] { 0,0.24,1,0,0,0.3880,0,1,0 }, // 23
       new double[] { 0,0.49,1,0,0,0.6450,0,1,0 }, // 24
       new double[] { 1,0.66,0,0,1,0.7450,1,0,0 }, // 25
       new double[] { 0,0.65,1,0,0,0.7690,1,0,0 }, // 26
       new double[] { 0,0.46,0,1,0,0.5800,1,0,0 }, // 27
       new double[] { 0,0.45,0,0,1,0.5180,0,1,0 }, // 28
       new double[] { 0,0.47,1,0,0,0.6360,1,0,0 }, // 29
       new double[] { 0,0.29,1,0,0,0.4480,1,0,0 }, // 30
       new double[] { 0,0.57,0,0,1,0.6930,0,0,1 }, // 31
       new double[] { 0,0.20,1,0,0,0.2870,0,0,1 }, // 32
       new double[] { 0,0.35,1,0,0,0.4340,0,1,0 }, // 33
       new double[] { 0,0.61,0,0,1,0.6700,0,0,1 }, // 34
       new double[] { 0,0.31,0,0,1,0.3730,0,1,0 }, // 35
       new double[] { 1,0.18,1,0,0,0.2080,0,0,1 }, // 36
       new double[] { 1,0.26,0,0,1,0.2920,0,0,1 }, // 37
       new double[] { 0,0.28,1,0,0,0.3640,0,0,1 }, // 38
       new double[] { 0,0.59,0,0,1,0.6940,0,0,1 }  // 39
    };
    return testData;
}