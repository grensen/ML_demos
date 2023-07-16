// https://learn.microsoft.com/en-us/archive/msdn-magazine/2019/november/test-run-mixture-model-clustering-using-csharp
// https://jamesmccaffrey.wordpress.com/2019/11/03/mixture-model-clustering-using-c/

const int N = 8; const int K = 3; const int d = 2;

Console.WriteLine("Begin mixture model with demo ");
double[][] x = new double[N][];  // 8 x 2 data
x[0] = new double[] { 0.2, 0.7 };
x[1] = new double[] { 0.1, 0.9 };
x[2] = new double[] { 0.2, 0.8 };
x[3] = new double[] { 0.4, 0.5 };
x[4] = new double[] { 0.5, 0.4 };
x[5] = new double[] { 0.9, 0.3 };
x[6] = new double[] { 0.8, 0.2 };
x[7] = new double[] { 0.7, 0.1 };

Console.WriteLine("Data (height, width): ");
Console.Write("[0] "); VectorShow(x[0]);
Console.Write("[1] "); VectorShow(x[1]);
Console.WriteLine(". . . ");
Console.Write("[7] "); VectorShow(x[7]);
Console.WriteLine("K=3, initing w, a, u, S, Nk");

double[][] w = MatrixCreate(N, K);
double[] a = new double[K] { 1.0 / K, 1.0 / K, 1.0 / K };
double[][] u = MatrixCreate(K, d);
double[][] V = MatrixCreate(K, d, 0.01);
double[] Nk = new double[K];
u[0][0] = 0.2; u[0][1] = 0.7;
u[1][0] = 0.5; u[1][1] = 0.5;
u[2][0] = 0.8; u[2][1] = 0.2;

Console.WriteLine("Performing 5 E-M iterations ");
for (int iter = 0; iter < 5; ++iter)
{
    UpdateMembershipWts(w, x, u, V, a);  // E step
    UpdateNk(Nk, w);  // M steps
    UpdateMixtureWts(a, Nk);
    UpdateMeans(u, w, x, Nk);
    UpdateVariances(V, u, w, x, Nk);
}

Console.WriteLine("Clustering done. \n");
ShowDataStructures(w, Nk, a, u, V);
Console.WriteLine("End mixture model demo");
Console.ReadLine();

static void ShowDataStructures(double[][] w, double[] Nk, double[] a, double[][] u, double[][] V)
{
    Console.WriteLine("w:"); MatrixShow(w, true);
    Console.WriteLine("Nk:"); VectorShow(Nk, true);
    Console.WriteLine("a:"); VectorShow(a, true);
    Console.WriteLine("u:"); MatrixShow(u, true);
    Console.WriteLine("V:"); MatrixShow(V, true);
}
static void UpdateMembershipWts(double[][] w, double[][] x, double[][] u, double[][] V, double[] a)
{
    for (int i = 0; i < N; ++i)
    {
        double rowSum = 0.0;
        for (int k = 0; k < K; ++k)
        {
            double pdf = NaiveProb(x[i], u[k], V[k]);
            w[i][k] = a[k] * pdf;
            rowSum += w[i][k];
        }
        for (int k = 0; k < K; ++k)
            w[i][k] = w[i][k] / rowSum;
    }
}
static void UpdateNk(double[] Nk, double[][] w)
{
    for (int k = 0; k < K; ++k)
    {
        double sum = 0.0;
        for (int i = 0; i < N; ++i)
            sum += w[i][k];
        Nk[k] = sum;
    }
}
static void UpdateMixtureWts(double[] a, double[] Nk)
{
    for (int k = 0; k < K; ++k)
        a[k] = Nk[k] / N;
}
static void UpdateMeans(double[][] u, double[][] w, double[][] x, double[] Nk)
{
    double[][] result = MatrixCreate(K, d);
    for (int k = 0; k < K; ++k)
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < d; ++j)
                result[k][j] += w[i][k] * x[i][j];
        for (int j = 0; j < d; ++j)
            result[k][j] = result[k][j] / Nk[k];
    }
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < d; ++j)
            u[k][j] = result[k][j];
}
static void UpdateVariances(double[][] V, double[][] u, double[][] w, double[][] x, double[] Nk)
{
    double[][] result = MatrixCreate(K, d);
    for (int k = 0; k < K; ++k)
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < d; ++j)
                result[k][j] += w[i][k] * (x[i][j] - u[k][j]) *
                (x[i][j] - u[k][j]);
        for (int j = 0; j < d; ++j)
            result[k][j] = result[k][j] / Nk[k];
    }
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < d; ++j)
            V[k][j] = result[k][j];
}
static double ProbDenFunc(double x, double u, double v)
{
    // Univariate Gaussian
    if (v == 0.0) throw new Exception("0 in ProbDenFun");
    double left = 1.0 / Math.Sqrt(2.0 * Math.PI * v);
    double pwr = -1 * ((x - u) * (x - u)) / (2 * v);
    return left * Math.Exp(pwr);
}
static double NaiveProb(double[] x, double[] u, double[] v)
{
    // Poor man's multivariate Gaussian PDF
    double sum = 0.0;
    for (int j = 0; j < d; ++j)
        sum += ProbDenFunc(x[j], u[j], v[j]);
    return sum / d;
}
static double[][] MatrixCreate(int rows, int cols, double v = 0.0)
{
    double[][] result = new double[rows][];
    for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[i][j] = v;
    return result;
}
static void MatrixShow(double[][] m, bool nl = false)
{
    for (int i = 0; i < m.Length; ++i)
    {
        for (int j = 0; j < m[0].Length; ++j)
        {
            Console.Write(m[i][j].ToString("F4") + "  ");
        }
        Console.WriteLine("");
    }
    if (nl == true)
        Console.WriteLine("");
}
static void VectorShow(double[] v, bool nl = false)
{
    for (int i = 0; i < v.Length; ++i)
        Console.Write(v[i].ToString("F4") + "  ");
    Console.WriteLine("");
    if (nl == true)
        Console.WriteLine("");
}
