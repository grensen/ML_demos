// https://github.com/grensen/easy_regression
using System.Linq;
using System;

// Initialize hyperparameters
int seed = 1337;
int epochs = 5;
float learningRate = 2.0f;

Console.WriteLine("\nBegin Easy Regression on Iris Dataset\n");
Console.WriteLine($"Seed     = {seed}");
Console.WriteLine($"Epochs   = {epochs}");
Console.WriteLine($"Learning = {learningRate:F3}");
// Load Iris training and test datasets
float[][] trainingData = LoadIrisTrainingDataFloat();
float[][] testData = LoadIrisTestDataFloat();
Console.WriteLine("\nOriginal Training Data:");
PrintIrisData(trainingData, showTitle: true);

// Step 1: Training without normalization
Console.WriteLine("Training without Normalization:");
float[] trainedWeights0 = RunTraining(trainingData, testData, learningRate, epochs, seed);

// Normalize the training and test data using Divide-By-Constant normalization
float[] constants = { 9.0f, 5.0f, 8.0f, 3.0f };
float[][] normDivideByConstantTrainingData = DivideByConstantNormalization(trainingData, constants);
float[][] normDivideByConstantTestData = DivideByConstantNormalization(testData, constants);

Console.Write($"\nUsing Divide-By-Constant Normalization, constants: ({string.Join(", ", constants.Select(c => c.ToString("F1")))}):\n");
PrintIrisData(normDivideByConstantTrainingData);
Console.WriteLine("Training with Divide-By-Constant Normalization:");
float[] trainedWeights3 = RunTraining(normDivideByConstantTrainingData, normDivideByConstantTestData, learningRate, epochs, seed);

// Normalize the training and test data using z-score normalization
Console.WriteLine("\nUsing Z-Score Normalization:");
float[][] normZScoreTrainingData = ZScoreNormalization(trainingData);
float[][] normZScoreTestData = ZScoreNormalization(testData);
PrintIrisData(normZScoreTrainingData);
Console.WriteLine("Training with Z-Score Normalization:");
float[] trainedWeights1 = RunTraining(normZScoreTrainingData, normZScoreTestData, learningRate, epochs, seed);

// Normalize the training and test data using Min-Max normalization [-1, 1]
Console.WriteLine($"\nUsing Min-Max Normalization (Scaling to [{-1}, {1}]):");
float[][] normMinusOnePlusOneTrainingData = MinMaxNormalization(trainingData, -1, 1);
float[][] normMinusOnePlusOneTestData = MinMaxNormalization(testData, -1, 1);
PrintIrisData(normMinusOnePlusOneTrainingData);
Console.WriteLine($"Training with Min-Max Normalization (Scaling to [{-1}, {1}]):");
float[] trainedWeights4 = RunTraining(normMinusOnePlusOneTrainingData, normMinusOnePlusOneTestData, learningRate, epochs, seed);

// Normalize the training and test data using Min-Max normalization [0, 1]
Console.WriteLine($"\nUsing Min-Max Normalization (Scaling to [{0}, {1}]):");
float[][] normMinMaxTrainingData = MinMaxNormalization(trainingData, 0, 1);
float[][] normMinMaxTestData = MinMaxNormalization(testData, 0, 1);
PrintIrisData(normMinMaxTrainingData);
Console.WriteLine($"Training with Min-Max Normalization (Scaling to [{0}, {1}]):");
float[] trainedWeights2 = RunTraining(normMinMaxTrainingData, normMinMaxTestData, learningRate, epochs, seed);

Console.WriteLine("\nEnd demo");

static float[] RunTraining(float[][] trainingData, float[][] testData, float lr, int epochs, int seed)
{
    // Initialize the random number generator with the specified seed
    Random rng = new Random(seed);

    int inputSize = 4; // Number of input features (sepal length, sepal width, petal length, petal width)
    int outputSize = 3; // Number of output classes (Iris setosa, Iris versicolor, Iris virginica)

    // init zero weights
    float[] weights = new float[inputSize * outputSize];

    for (int ep = 0; ep < epochs; ep++)
    {
        // Shuffle the indices array using Fisher-Yates (Knuth) shuffle
        int[] indices = Shuffle(trainingData.Length, rng.Next());

        // more efficient learning rate - reduced weights impact each epoch
        for (int i = 0; i < weights.Length; i++) weights[i] += weights[i] * lr;

        // get training accuracy
        int correctTraining = 0;
        for (int x = 0; x < trainingData.Length; x++)
            correctTraining += Test(trainingData[indices[x]], weights, true);

        // info each epoch
        if (ep < 1 || ep > epochs - 2 || ep == epochs / 2)
            Console.WriteLine($"Epoch = {(ep + 1),2} | Accuracy = {correctTraining * 100.0 / trainingData.Length:F2}%");
        //  else if (ep == 2)  Console.WriteLine("...");
    }

    // test trained weights on test data
    int correctTest = 0;
    for (int x = 0; x < testData.Length; x++)
        correctTest += Test(testData[x], weights);

    Console.WriteLine("Test accuracy = " + (correctTest * 100.0 / testData.Length).ToString("F2") + "%");

    return weights;
}
static int Test(float[] data, float[] weights, bool training = false)
{
    int inputSize = 4; // Number of input features (sepal length, sepal width, petal length, petal width)
    int outputSize = 3; // Number of output classes (Iris setosa, Iris versicolor, Iris virginica)
    int correct = 0;

    // feed forward
    float[] outputs = new float[outputSize];
    for (int i = 0; i < inputSize; i++)
        for (int j = 0; j < outputSize; j++)
            outputs[j] += data[i] * weights[i * outputSize + j];

    int prediction = ArgMax(outputs);
    int targetClass = (int)data[inputSize]; // Assuming the class label is the last element in the data array

    if (training) // (backprop) plus update
        if (prediction != targetClass)
            for (int i = 0; i < inputSize; i++)
            {
                weights[i * outputSize + targetClass] += data[i];
                weights[i * outputSize + prediction] -= data[i];
            }

    correct += prediction == targetClass ? 1 : 0;

    return correct;
}
static int ArgMax(float[] arr)
{
    int prediction = 0;
    float max = arr[0];
    for (int i = 1; i < arr.Length; i++)
        if (arr[i] > max)
        {
            max = arr[i];
            prediction = i;
        }
    return prediction;
}
static int[] Shuffle(int length, int seed, bool doShuffle = true)
{
    Random random = new Random(seed);
    int[] indices = new int[length];
    for (int i = 0; i < length; i++) indices[i] = i;
    if (!doShuffle) return indices;
    for (int i = length - 1; i > 0; i--)
    {
        int j = random.Next(i + 1);
        (indices[i], indices[j]) = (indices[j], indices[i]);
    }
    return indices;
}
static void PrintIrisData(float[][] data, bool showTitle = false)
{
    // Add "Index" to the feature names
    string[] names = { "Index", "Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Label" };

    // Calculate the distance needed to align the values properly
    int[] distance = new int[names.Length];
    for (int i = 0; i < names.Length; i++)
    {
        int maxLength = names[i].Length;
        foreach (var dataPoint in data)
        {
            int length = i == 0 ? dataPoint[i].ToString("F1").Length : dataPoint[i - 1].ToString("F1").Length;
            maxLength = Math.Max(maxLength, length);
        }
        distance[i] = maxLength;
    }

    // Print feature names as titles
    string titleStr = "";
    for (int i = 0; i < names.Length; i++)
    {
        titleStr += names[i].PadRight(distance[i] + 1); // Add 1 to create a space after each column
        if (i < names.Length - 1)
            titleStr += "| ";
    }
    if (showTitle) Console.WriteLine(titleStr);

    // Print specific data points
    PrintDataPoint(data, 0, distance);
    PrintDataPoint(data, data.Length / 2, distance);
    PrintDataPoint(data, data.Length - 1, distance);
}
static void PrintDataPoint(float[][] data, int id, int[] distance)
{
    float[] dataPoint = data[id];
    string valueToPrint = id.ToString().PadRight(distance[0] + 1) + "  ";

    for (int i = 1; i < dataPoint.Length + 1; i++)
    {
        valueToPrint += dataPoint[i - 1].ToString("F1").PadRight(distance[i] + 1);

        if (i < dataPoint.Length)
            valueToPrint += "  ";
    }
    Console.WriteLine(valueToPrint);
}
static float[][] DivideByConstantNormalization(float[][] data, float[] constants)
{
    if (data[0].Length - 1 != constants.Length)
    {
        Console.WriteLine("Number of constants should be equal to the number of features.");
        return null;
    }
    float[][] normalizedData = data.Select(row => (float[])row.Clone()).ToArray();
    for (int i = 0; i < normalizedData[0].Length - 1; i++) // Exclude the label column
    {
        float constant = constants[i];
        for (int j = 0; j < normalizedData.Length; j++)
            normalizedData[j][i] /= constant;
    }
    return normalizedData;
}
static float[][] ZScoreNormalization(float[][] data)
{
    int numFeatures = data[0].Length - 1; // Exclude the label column
    float[][] norms = new float[numFeatures][];
    for (int i = 0; i < numFeatures; i++)
    {
        float mean = data.Select(row => row[i]).Average();
        float stdDev = (float)Math.Sqrt(data.Select(row => Math.Pow(row[i] - mean, 2)).Average());

        norms[i] = new float[] { mean, stdDev };
    }
    float[][] normalizedData = new float[data.Length][];
    for (int j = 0; j < data.Length; j++)
    {
        normalizedData[j] = new float[data[j].Length];
        Array.Copy(data[j], normalizedData[j], data[j].Length); // Copy original data to normalizedData

        for (int i = 0; i < numFeatures; i++)
            normalizedData[j][i] = (data[j][i] - norms[i][0]) / norms[i][1];
    }
    return normalizedData;
}
static float[][] MinMaxNormalization(float[][] data, float minRange, float maxRange)
{
    int numFeatures = data[0].Length - 1; // Exclude the label column
    float[][] normalizedData = new float[data.Length][];
    for (int i = 0; i < data.Length; i++)
        normalizedData[i] = new float[numFeatures + 1]; // Include space for the label column
    for (int i = 0; i < numFeatures; i++)
    {
        float min = data.Select(row => row[i]).Min();
        float max = data.Select(row => row[i]).Max();

        for (int j = 0; j < data.Length; j++)
            normalizedData[j][i] = ((data[j][i] - min) / (max - min)) * (maxRange - minRange) + minRange;
    }

    // Copy the label column from the original data to the normalized data
    for (int i = 0; i < data.Length; i++)
        normalizedData[i][numFeatures] = data[i][numFeatures];
    return normalizedData;
}
static float[][] LoadIrisTrainingDataFloat()
{
    float[][] trainData = new float[][]
    {
        new float[] { 5.1f, 3.5f, 1.4f, 0.2f, 0 },  // 1
        new float[] { 4.9f, 3.0f, 1.4f, 0.2f, 0 },  // 2
        new float[] { 4.7f, 3.2f, 1.3f, 0.2f, 0 },  // 3
        new float[] { 4.6f, 3.1f, 1.5f, 0.2f, 0 },  // 4
        new float[] { 5.0f, 3.6f, 1.4f, 0.2f, 0 },  // 5
        new float[] { 5.4f, 3.9f, 1.7f, 0.4f, 0 },  // 6
        new float[] { 4.6f, 3.4f, 1.4f, 0.3f, 0 },  // 7
        new float[] { 5.0f, 3.4f, 1.5f, 0.2f, 0 },  // 8
        new float[] { 4.4f, 2.9f, 1.4f, 0.2f, 0 },  // 9
        new float[] { 4.9f, 3.1f, 1.5f, 0.1f, 0 },  // 10
        new float[] { 5.4f, 3.7f, 1.5f, 0.2f, 0 },  // 11
        new float[] { 4.8f, 3.4f, 1.6f, 0.2f, 0 },  // 12
        new float[] { 4.8f, 3.0f, 1.4f, 0.1f, 0 },  // 13
        new float[] { 4.3f, 3.0f, 1.1f, 0.1f, 0 },  // 14
        new float[] { 5.8f, 4.0f, 1.2f, 0.2f, 0 },  // 15
        new float[] { 5.7f, 4.4f, 1.5f, 0.4f, 0 },  // 16
        new float[] { 5.4f, 3.9f, 1.3f, 0.4f, 0 },  // 17
        new float[] { 5.1f, 3.5f, 1.4f, 0.3f, 0 },  // 18
        new float[] { 5.7f, 3.8f, 1.7f, 0.3f, 0 },  // 19
        new float[] { 5.1f, 3.8f, 1.5f, 0.3f, 0 },  // 20
        new float[] { 5.4f, 3.4f, 1.7f, 0.2f, 0 },  // 21
        new float[] { 5.1f, 3.7f, 1.5f, 0.4f, 0 },  // 22
        new float[] { 4.6f, 3.6f, 1.0f, 0.2f, 0 },  // 23
        new float[] { 5.1f, 3.3f, 1.7f, 0.5f, 0 },  // 24
        new float[] { 4.8f, 3.4f, 1.9f, 0.2f, 0 },  // 25
        new float[] { 5.0f, 3.0f, 1.6f, 0.2f, 0 },  // 26
        new float[] { 5.0f, 3.4f, 1.6f, 0.4f, 0 },  // 27
        new float[] { 5.2f, 3.5f, 1.5f, 0.2f, 0 },  // 28
        new float[] { 5.2f, 3.4f, 1.4f, 0.2f, 0 },  // 29
        new float[] { 4.7f, 3.2f, 1.6f, 0.2f, 0 },  // 30
        new float[] { 4.8f, 3.1f, 1.6f, 0.2f, 0 },  // 31
        new float[] { 5.4f, 3.4f, 1.5f, 0.4f, 0 },  // 32
        new float[] { 5.2f, 4.1f, 1.5f, 0.1f, 0 },  // 33
        new float[] { 5.5f, 4.2f, 1.4f, 0.2f, 0 },  // 34
        new float[] { 4.9f, 3.1f, 1.5f, 0.1f, 0 },  // 35
        new float[] { 5.0f, 3.2f, 1.2f, 0.2f, 0 },  // 36
        new float[] { 5.5f, 3.5f, 1.3f, 0.2f, 0 },  // 37
        new float[] { 4.9f, 3.1f, 1.5f, 0.1f, 0 },  // 38
        new float[] { 4.4f, 3.0f, 1.3f, 0.2f, 0 },  // 39
        new float[] { 5.1f, 3.4f, 1.5f, 0.2f, 0 },  // 40
        new float[] { 7.0f, 3.2f, 4.7f, 1.4f, 1 },  // 41
        new float[] { 6.4f, 3.2f, 4.5f, 1.5f, 1 },  // 42
        new float[] { 6.9f, 3.1f, 4.9f, 1.5f, 1 },  // 43
        new float[] { 5.5f, 2.3f, 4.0f, 1.3f, 1 },  // 44
        new float[] { 6.5f, 2.8f, 4.6f, 1.5f, 1 },  // 45
        new float[] { 5.7f, 2.8f, 4.5f, 1.3f, 1 },  // 46
        new float[] { 6.3f, 3.3f, 4.7f, 1.6f, 1 },  // 47
        new float[] { 4.9f, 2.4f, 3.3f, 1.0f, 1 },  // 48
        new float[] { 6.6f, 2.9f, 4.6f, 1.3f, 1 },  // 49
        new float[] { 5.2f, 2.7f, 3.9f, 1.4f, 1 },  // 50
        new float[] { 5.0f, 2.0f, 3.5f, 1.0f, 1 },  // 51
        new float[] { 5.9f, 3.0f, 4.2f, 1.5f, 1 },  // 52
        new float[] { 6.0f, 2.2f, 4.0f, 1.0f, 1 },  // 53
        new float[] { 6.1f, 2.9f, 4.7f, 1.4f, 1 },  // 54
        new float[] { 5.6f, 2.9f, 3.6f, 1.3f, 1 },  // 55
        new float[] { 6.7f, 3.1f, 4.4f, 1.4f, 1 },  // 56
        new float[] { 5.6f, 3.0f, 4.5f, 1.5f, 1 },  // 57
        new float[] { 5.8f, 2.7f, 4.1f, 1.0f, 1 },  // 58
        new float[] { 6.2f, 2.2f, 4.5f, 1.5f, 1 },  // 59
        new float[] { 5.6f, 2.5f, 3.9f, 1.1f, 1 },  // 60
        new float[] { 5.9f, 3.2f, 4.8f, 1.8f, 1 },  // 61
        new float[] { 6.1f, 2.8f, 4.0f, 1.3f, 1 },  // 62
        new float[] { 6.3f, 2.5f, 4.9f, 1.5f, 1 },  // 63
        new float[] { 6.1f, 2.8f, 4.7f, 1.2f, 1 },  // 64
        new float[] { 6.4f, 2.9f, 4.3f, 1.3f, 1 },  // 65
        new float[] { 6.6f, 3.0f, 4.4f, 1.4f, 1 },  // 66
        new float[] { 6.8f, 2.8f, 4.8f, 1.4f, 1 },  // 67
        new float[] { 6.7f, 3.0f, 5.0f, 1.7f, 1 },  // 68
        new float[] { 6.0f, 2.9f, 4.5f, 1.5f, 1 },  // 69
        new float[] { 5.7f, 2.6f, 3.5f, 1.0f, 1 },  // 70
        new float[] { 5.5f, 2.4f, 3.8f, 1.1f, 1 },  // 71
        new float[] { 5.5f, 2.4f, 3.7f, 1.0f, 1 },  // 72
        new float[] { 5.8f, 2.7f, 3.9f, 1.2f, 1 },  // 73
        new float[] { 6.0f, 2.7f, 5.1f, 1.6f, 1 },  // 74
        new float[] { 5.4f, 3.0f, 4.5f, 1.5f, 1 },  // 75
        new float[] { 6.0f, 3.4f, 4.5f, 1.6f, 1 },  // 76
        new float[] { 6.7f, 3.1f, 4.7f, 1.5f, 1 },  // 77
        new float[] { 6.3f, 2.3f, 4.4f, 1.3f, 1 },  // 78
        new float[] { 5.6f, 3.0f, 4.1f, 1.3f, 1 },  // 79
        new float[] { 5.5f, 2.5f, 4.0f, 1.3f, 1 },  // 80
        new float[] { 6.3f, 3.3f, 6.0f, 2.5f, 2 },  // 81
        new float[] { 5.8f, 2.7f, 5.1f, 1.9f, 2 },  // 82
        new float[] { 7.1f, 3.0f, 5.9f, 2.1f, 2 },  // 83
        new float[] { 6.3f, 2.9f, 5.6f, 1.8f, 2 },  // 84
        new float[] { 6.5f, 3.0f, 5.8f, 2.2f, 2 },  // 85
        new float[] { 7.6f, 3.0f, 6.6f, 2.1f, 2 },  // 86
        new float[] { 4.9f, 2.5f, 4.5f, 1.7f, 2 },  // 87
        new float[] { 7.3f, 2.9f, 6.3f, 1.8f, 2 },  // 88
        new float[] { 6.7f, 2.5f, 5.8f, 1.8f, 2 },  // 89
        new float[] { 7.2f, 3.6f, 6.1f, 2.5f, 2 },  // 90
        new float[] { 6.5f, 3.2f, 5.1f, 2.0f, 2 },  // 91
        new float[] { 6.4f, 2.7f, 5.3f, 1.9f, 2 },  // 92
        new float[] { 6.8f, 3.0f, 5.5f, 2.1f, 2 },  // 93
        new float[] { 5.7f, 2.5f, 5.0f, 2.0f, 2 },  // 94
        new float[] { 5.8f, 2.8f, 5.1f, 2.4f, 2 },  // 95
        new float[] { 6.4f, 3.2f, 5.3f, 2.3f, 2 },  // 96
        new float[] { 6.5f, 3.0f, 5.5f, 1.8f, 2 },  // 97
        new float[] { 7.7f, 3.8f, 6.7f, 2.2f, 2 },  // 98
        new float[] { 7.7f, 2.6f, 6.9f, 2.3f, 2 },  // 99
        new float[] { 6.0f, 2.2f, 5.0f, 1.5f, 2 },  // 100
        new float[] { 6.9f, 3.2f, 5.7f, 2.3f, 2 },  // 101
        new float[] { 5.6f, 2.8f, 4.9f, 2.0f, 2 },  // 102
        new float[] { 7.7f, 2.8f, 6.7f, 2.0f, 2 },  // 103
        new float[] { 6.3f, 2.7f, 4.9f, 1.8f, 2 },  // 104
        new float[] { 6.7f, 3.3f, 5.7f, 2.1f, 2 },  // 105
        new float[] { 7.2f, 3.2f, 6.0f, 1.8f, 2 },  // 106
        new float[] { 6.2f, 2.8f, 4.8f, 1.8f, 2 },  // 107
        new float[] { 6.1f, 3.0f, 4.9f, 1.8f, 2 },  // 108
        new float[] { 6.4f, 2.8f, 5.6f, 2.1f, 2 },  // 109
        new float[] { 7.2f, 3.0f, 5.8f, 1.6f, 2 },  // 110
        new float[] { 7.4f, 2.8f, 6.1f, 1.9f, 2 },  // 111
        new float[] { 7.9f, 3.8f, 6.4f, 2.0f, 2 },  // 112
        new float[] { 6.4f, 2.8f, 5.6f, 2.2f, 2 },  // 113
        new float[] { 6.3f, 2.8f, 5.1f, 1.5f, 2 },  // 114
        new float[] { 6.1f, 2.6f, 5.6f, 1.4f, 2 },  // 115
        new float[] { 7.7f, 3.0f, 6.1f, 2.3f, 2 },  // 116
        new float[] { 6.3f, 3.4f, 5.6f, 2.4f, 2 },  // 117
        new float[] { 6.4f, 3.1f, 5.5f, 1.8f, 2 },  // 118
        new float[] { 6.0f, 3.0f, 4.8f, 1.8f, 2 },  // 119
        new float[] { 6.9f, 3.1f, 5.4f, 2.1f, 2 }   // 120
    };

    return trainData;
}
static float[][] LoadIrisTestDataFloat()
{
    float[][] testData = new float[][]
    {
        new float[] { 5.0f, 3.5f, 1.3f, 0.3f, 0 },  // 1
        new float[] { 4.5f, 2.3f, 1.3f, 0.3f, 0 },  // 2
        new float[] { 4.4f, 3.2f, 1.3f, 0.2f, 0 },  // 3
        new float[] { 5.0f, 3.5f, 1.6f, 0.6f, 0 },  // 4
        new float[] { 5.1f, 3.8f, 1.9f, 0.4f, 0 },  // 5
        new float[] { 4.8f, 3.0f, 1.4f, 0.3f, 0 },  // 6
        new float[] { 5.1f, 3.8f, 1.6f, 0.2f, 0 },  // 7
        new float[] { 4.6f, 3.2f, 1.4f, 0.2f, 0 },  // 8
        new float[] { 5.3f, 3.7f, 1.5f, 0.2f, 0 },  // 9
        new float[] { 5.0f, 3.3f, 1.4f, 0.2f, 0 },  // 10
        new float[] { 5.5f, 2.6f, 4.4f, 1.2f, 1 },  // 11
        new float[] { 6.1f, 3.0f, 4.6f, 1.4f, 1 },  // 12
        new float[] { 5.8f, 2.6f, 4.0f, 1.2f, 1 },  // 13
        new float[] { 5.0f, 2.3f, 3.3f, 1.0f, 1 },  // 14
        new float[] { 5.6f, 2.7f, 4.2f, 1.3f, 1 },  // 15
        new float[] { 5.7f, 3.0f, 4.2f, 1.2f, 1 },  // 16
        new float[] { 5.7f, 2.9f, 4.2f, 1.3f, 1 },  // 17
        new float[] { 6.2f, 2.9f, 4.3f, 1.3f, 1 },  // 18
        new float[] { 5.1f, 2.5f, 3.0f, 1.1f, 1 },  // 19
        new float[] { 5.7f, 2.8f, 4.1f, 1.3f, 1 },  // 20
        new float[] { 6.7f, 3.1f, 5.6f, 2.4f, 2 },  // 21
        new float[] { 6.9f, 3.1f, 5.1f, 2.3f, 2 },  // 22
        new float[] { 5.8f, 2.7f, 5.1f, 1.9f, 2 },  // 23
        new float[] { 6.8f, 3.2f, 5.9f, 2.3f, 2 },  // 24
        new float[] { 6.7f, 3.3f, 5.7f, 2.5f, 2 },  // 25
        new float[] { 6.7f, 3.0f, 5.2f, 2.3f, 2 },  // 26
        new float[] { 6.3f, 2.5f, 5.0f, 1.9f, 2 },  // 27
        new float[] { 6.5f, 3.0f, 5.2f, 2.0f, 2 },  // 28
        new float[] { 6.2f, 3.4f, 5.4f, 2.3f, 2 },  // 29
        new float[] { 5.9f, 3.0f, 5.1f, 1.8f, 2 },  // 30
    };
    return testData;
}
