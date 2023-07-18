// https://visualstudiomagazine.com/articles/2022/05/02/naive-bayes-classification-csharp.aspx
// https://jamesmccaffrey.wordpress.com/2022/05/16/naive-bayes-classification-using-csharp-in-visual-studio-magazine/

Console.WriteLine("\nBegin naive Bayes classification");
Console.WriteLine("\nData looks like: ");
Console.WriteLine("actor  green  korea  1");
Console.WriteLine("baker  green  italy  0");
Console.WriteLine("diver  hazel  japan  0");
Console.WriteLine("diver  green  japan  1");
Console.WriteLine("clerk  hazel  japan  2");
Console.WriteLine(" . . . ");

string[][] data = GetData();
int nx = 3;  // number predictors (job, eye, country)
int nc = 3;  // number classes (0, 1, 2)
int N = 40;  // number data items

string[] X = new string[] { "baker", "hazel", "italy" };
Console.WriteLine("\nItem to classify: ");
Console.WriteLine("baker  hazel  italy");

int[][] jointCounts = ComputeJointCounts(data, X, nx, nc, N);
LaplacianSmoothing(jointCounts, nx, nc);
int[] yCounts = ComputeClassCounts(data, nx, nc, N);
double[] eTerms = ComputeEvidenceTerms(jointCounts, yCounts, nx, nc, N);
double evidence = CalculateEvidence(eTerms);
double[] probs = CalculateClassProbabilities(eTerms, evidence);

Console.WriteLine("\nClass counts (raw):");
ShowVector(yCounts);

Console.WriteLine("\nPseudo-probabilities each class:");
ShowVector(probs, "F4");

Console.WriteLine("\nEnd naive Bayes demo");
Console.ReadLine();

static int[][] ComputeJointCounts(string[][] data, string[] X, int nx, int nc, int N)
{
    int[][] jointCounts = new int[nx][];
    for (int i = 0; i < nx; ++i)
        jointCounts[i] = new int[nc];
    for (int i = 0; i < N; ++i)  // compute joint counts
    {
        int y = int.Parse(data[i][nx]);  // get the class as int
        for (int j = 0; j < nx; ++j)
            if (data[i][j] == X[j])
                ++jointCounts[j][y];
    }

    return jointCounts;
}
static int[] ComputeClassCounts(string[][] data, int nx, int nc, int N)
{
    int[] yCounts = new int[nc];
    for (int i = 0; i < N; ++i)  // compute class counts
    {
        int y = int.Parse(data[i][nx]);  // get the class as int
        ++yCounts[y];
    }
    return yCounts;
}
static void LaplacianSmoothing(int[][] jointCounts, int nx, int nc)
{
    for (int i = 0; i < nx; ++i)  // Laplacian smoothing
        for (int j = 0; j < nc; ++j)
            ++jointCounts[i][j];
}
static double[] ComputeEvidenceTerms(int[][] jointCounts, int[] yCounts, int nx, int nc, int N)
{
    double[] eTerms = new double[nc];
    for (int k = 0; k < nc; ++k)
    {
        if (true)
        {
            double v = 1.0;  // direct approach
            for (int j = 0; j < nx; ++j)
                v *= (jointCounts[j][k] * 1.0) / (yCounts[k] + nx);
            v *= (yCounts[k] * 1.0) / N;
            eTerms[k] = v;
        }
        else
        {
            double v = 0.0;  // use logs to avoid underflow
            for (int j = 0; j < nx; ++j)
                v += Math.Log(jointCounts[j][k]) - Math.Log(yCounts[k] + nx);
            v += Math.Log(yCounts[k]) - Math.Log(N);
            eTerms[k] = Math.Exp(v);
        }
    }
    return eTerms;
}
static double CalculateEvidence(double[] eTerms)
{
    double evidence = 0.0;
    for (int k = 0; k < eTerms.Length; ++k)
        evidence += eTerms[k];

    return evidence;
}
static double[] CalculateClassProbabilities(double[] eTerms, double evidence)
{
    double[] probs = new double[eTerms.Length];
    for (int k = 0; k < eTerms.Length; ++k)
        probs[k] = eTerms[k] / evidence;

    return probs;
}
static void ShowVector<T>(T[] vector, string format = "")
{
    for (int i = 0; i < vector.Length; ++i)
        Console.Write(string.Format("{0:" + format + "}  ", vector[i]));
    Console.WriteLine();
}
static string[][] GetData()
{
    string[][] m = new string[40][];  // 40 rows
    m[0] = new string[] { "actor", "green", "korea", "1" };
    m[1] = new string[] { "baker", "green", "italy", "0" };
    m[2] = new string[] { "diver", "hazel", "japan", "0" };
    m[3] = new string[] { "diver", "green", "japan", "1" };
    m[4] = new string[] { "clerk", "hazel", "japan", "2" };
    m[5] = new string[] { "actor", "green", "japan", "1" };
    m[6] = new string[] { "actor", "green", "japan", "0" };
    m[7] = new string[] { "clerk", "green", "italy", "1" };
    m[8] = new string[] { "clerk", "green", "italy", "2" };
    m[9] = new string[] { "diver", "green", "japan", "1" };
    m[10] = new string[] { "diver", "green", "japan", "0" };
    m[11] = new string[] { "diver", "green", "japan", "1" };
    m[12] = new string[] { "diver", "green", "japan", "2" };
    m[13] = new string[] { "clerk", "green", "italy", "1" };
    m[14] = new string[] { "diver", "green", "japan", "1" };
    m[15] = new string[] { "diver", "hazel", "japan", "0" };
    m[16] = new string[] { "clerk", "green", "korea", "1" };
    m[17] = new string[] { "baker", "green", "japan", "0" };
    m[18] = new string[] { "actor", "green", "italy", "1" };
    m[19] = new string[] { "actor", "green", "italy", "1" };
    m[20] = new string[] { "diver", "green", "korea", "0" };
    m[21] = new string[] { "baker", "green", "japan", "2" };
    m[22] = new string[] { "diver", "green", "japan", "0" };
    m[23] = new string[] { "baker", "green", "korea", "0" };
    m[24] = new string[] { "diver", "green", "japan", "0" };
    m[25] = new string[] { "actor", "hazel", "italy", "1" };
    m[26] = new string[] { "diver", "hazel", "japan", "0" };
    m[27] = new string[] { "diver", "green", "japan", "2" };
    m[28] = new string[] { "diver", "green", "japan", "0" };
    m[29] = new string[] { "clerk", "hazel", "japan", "2" };
    m[30] = new string[] { "diver", "green", "korea", "0" };
    m[31] = new string[] { "diver", "hazel", "korea", "0" };
    m[32] = new string[] { "diver", "green", "japan", "0" };
    m[33] = new string[] { "diver", "green", "japan", "2" };
    m[34] = new string[] { "diver", "hazel", "japan", "0" };
    m[35] = new string[] { "actor", "hazel", "japan", "1" };
    m[36] = new string[] { "actor", "green", "japan", "0" };
    m[37] = new string[] { "actor", "green", "japan", "1" };
    m[38] = new string[] { "diver", "green", "japan", "0" };
    m[39] = new string[] { "baker", "green", "japan", "0" };
    return m;
}
