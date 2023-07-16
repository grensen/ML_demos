// the demo based on the work of Dr. James D. McCaffrey
// "How to Do Naive Bayes with Numeric Data Using C#"
// https://visualstudiomagazine.com/articles/2019/11/12/naive-bayes-csharp.aspx           

// modification for continuous data
// the code for running mean and variance from John D. Cook
// https://www.johndcook.com/blog/standard_deviation/

Console.WriteLine("\nBegin running numeric naive Bayes demo");
Console.WriteLine("\nData looks like: ");
Console.WriteLine("Height  Weight  Foot  Sex");
Console.WriteLine("=========================");
Console.WriteLine("6.00,  180,  11,  0");
Console.WriteLine("5.30,  120,   7,  1");
Console.WriteLine(" . . .");

double[][] data = new double[9][];  // modified, better example
data[0] = new double[] { 6.00, 180, 11, 0 };  // 0 = male
data[1] = new double[] { 5.90, 190, 9, 0 };  // 0 = male
data[2] = new double[] { 5.70, 170, 8, 0 };  // 0 = male
data[3] = new double[] { 5.60, 140, 10, 0 };  // 0 = male

data[4] = new double[] { 5.80, 120, 9, 1 };  // 1 = female
data[5] = new double[] { 5.50, 150, 6, 1 };  // 1 = female
data[6] = new double[] { 5.30, 120, 7, 1 };  // 1 = female
data[7] = new double[] { 5.00, 100, 5, 1 };  // 1 = female

data[8] = new double[] { 5.60, 150, 8, -1 };  // -1 = unknown

Console.WriteLine("\nItem to predict:");
Console.WriteLine("5.60   150   8");

int N = data.Length;  // 8 items + 1 unknown
int nx = 3;  // Number predictor variables
int nc = 2;  // Number classes

var classCts = new int[nc];  // male, female 
var evidenceTerms = new double[nx];
var predictProbs = new double[nx];

RunningStat[] rs = new RunningStat[nx * nc];

ComputePredictedProbabilities();

// display prediction probabilities for last item 
Console.WriteLine("\nPrediction probabilities (male, female):");
for (int c = 0; c < 2; ++c)
    Console.WriteLine("class: " + c + "   " + predictProbs[c].ToString("F6"));
Console.WriteLine("\nPrediction class: " + ArgMax(predictProbs));
Console.WriteLine("\nEnd demo");
Console.ReadLine();

void ComputePredictedProbabilities()
{
    for (int p = 0; p < N - 1; p++)
    {
        // 1. Compute class counts and add values
        ++classCts[(int)data[p][nx]];
        for (int j = 0, c = (int)data[p][nx] * nx; j < nx; ++j, c++) // ht, wt, foot
            rs[c].Push(data[p][j]);

        // 2. Compute evidence terms
        double sumEvidence = 0.0;
        for (int c = 0, k = 0; c < nc; ++c)
        {
            double evi = Math.Log(classCts[c]); // evi = (classCts[c] * 1.0) / N;
            for (int j = 0; j < nx; ++j, k++) // evi *= ProbDensFunc(rs[k].Mean, rs[k].Variance, unk[j]);
                evi += Math.Log(ProbDensFunc(rs[k].Mean, rs[k].Variance, data[p + 1][j]));

            sumEvidence += evidenceTerms[c] = Math.Exp(evi); //sumEvidence += evidenceTerms[c] = evi;
        }

        // 3. Compute predicted probabilities
        for (int c = 0; c < nc; ++c)
            predictProbs[c] = evidenceTerms[c] / (sumEvidence + 1e-8);
    }
}
static int ArgMax(double[] vector)
{
    int result = 0;
    double maxV = vector[0];
    for (int i = 0, ie = vector.Length; i < ie; ++i)
    {
        if (vector[i] > maxV)
        {
            maxV = vector[i];
            result = i;
        }
    }
    return result;
}
static double ProbDensFunc(double u, double v, double x)
{
    double left = 1.0 / ((Math.Sqrt(2 * Math.PI * v) + 1e-8f));
    double right = Math.Exp(-(x - u) * (x - u) / (2 * v + 1e-8f));
    return left * right;
}
struct RunningStat
{
    private int count;
    private double oldMean, newMean, oldVariance, newVariance;

    public RunningStat()
    {
        count = 0;
        oldMean = newMean = oldVariance = newVariance = 0.0;
    }

    public void Push(double x)
    {
        count++;
        if (count == 1)
        {
            oldMean = newMean = x;
            oldVariance = 0.0;
        }
        else
        {
            newMean = oldMean + (x - oldMean) / count;
            newVariance = oldVariance + (x - oldMean) * (x - newMean);

            oldMean = newMean;
            oldVariance = newVariance;
        }
    }

    public double Mean => count > 0 ? newMean : 0.0;
    public double Variance => count > 1 ? newVariance / (count - 1) : 0.0;
    public double StandardDeviation => Math.Sqrt(Variance);
}