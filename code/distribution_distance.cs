// https://visualstudiomagazine.com/articles/2021/08/16/wasserstein-distance.aspx
// https://jamesmccaffrey.wordpress.com/2021/08/23/comparing-wasserstein-distance-with-kullback-leibler-distance/ 

Console.WriteLine("\nBegin compare Wasserstein and symmetric Kullback-Leibler\n");

double[] left = { 0.6, 0.1, 0.1, 0.1, 0.1 };
double[] center = { 0.1, 0.1, 0.6, 0.1, 0.1 };
double[] right = { 0.1, 0.1, 0.1, 0.1, 0.6 };

double klLeftCenter = MySymmKullback(left, center);
double klCenterRight = MySymmKullback(center, right);
double klLeftRight = MySymmKullback(left, right);

double wassLeftCenter = MyWasserstein(left, center);
double wassCenterRight = MyWasserstein(center, right);
double wassLeftRight = MyWasserstein(left, right);

Console.WriteLine("Kullback-Leibler distances:");
Console.WriteLine("Left to Center  : {0:F4}", klLeftCenter);
Console.WriteLine("Center to Right : {0:F4}", klCenterRight);
Console.WriteLine("Left to Right   : {0:F4}\n", klLeftRight);

Console.WriteLine("Wasserstein distances:");
Console.WriteLine("Left to Center  : {0:F4}", wassLeftCenter);
Console.WriteLine("Center to Right : {0:F4}", wassCenterRight);
Console.WriteLine("Left to Right   : {0:F4}\n", wassLeftRight);

double jsLeftCenter = JensenShannon(left, center);
double jsCenterRight = JensenShannon(center, right);
double jsLeftRight = JensenShannon(left, right);

double hLeftCenter = Hellinger(left, center);
double hCenterRight = Hellinger(center, right);
double hLeftRight = Hellinger(left, right);

Console.WriteLine("Jensen-Shannon distances:");
Console.WriteLine("Left to Center  : {0:F4}", jsLeftCenter);
Console.WriteLine("Center to Right : {0:F4}", jsCenterRight);
Console.WriteLine("Left to Right   : {0:F4}\n", jsLeftRight);

Console.WriteLine("Hellinger distances:");
Console.WriteLine("Left to Center  : {0:F4}", hLeftCenter);
Console.WriteLine("Center to Right : {0:F4}", hCenterRight);
Console.WriteLine("Left to Right   : {0:F4}\n", hLeftRight);

Console.WriteLine("End demo");

static double MyWasserstein(double[] dirt, double[] holes)
{
    double[] dirtCopy = (double[])dirt.Clone();
    double[] holesCopy = (double[])holes.Clone();
   
    double totWork = 0.0;
    while (true)  // TODO: add sanity counter check
    {
        int fromIdx = WassFirstNonzero(dirtCopy);
        int toIdx = WassFirstNonzero(holesCopy);
        if (fromIdx == -1 || toIdx == -1) break;

        double flow, dist;
        WassMoveDirt(dirtCopy, fromIdx, holesCopy, toIdx, out flow, out dist);
        totWork += flow * dist;
    }
    return totWork;
}
static int WassFirstNonzero(double[] dirtOrHoles)
{
    for (int i = 0; i < dirtOrHoles.Length; i++)
        if (dirtOrHoles[i] > 0.0)
            return i;
    return -1;  // no cells found
}
static void WassMoveDirt(double[] dirt, int fromIdx, double[] holes, int toIdx, out double flow, out double dist)
{
    double dirtValue = dirt[fromIdx];
    double holesValue = holes[toIdx];

    // move as much dirt as possible to holes
    if (dirtValue <= holesValue)  // use all dirt
    {
        flow = dirtValue;
        dirt[fromIdx] = 0.0;  // all dirt got moved
        holes[toIdx] -= flow;  // less to fill now
    }
    else  // use just part of dirt
    {
        flow = holesValue;  // fill remainder of hole
        dirt[fromIdx] -= flow;
        holes[toIdx] = 0.0;  // hole is filled
    }

    dist = Math.Abs(fromIdx - toIdx);
}
static double KL(double[] p, double[] q)
{
    double sum = 0.0;
    for (int i = 0; i < p.Length; i++)
        sum += p[i] * Math.Log(p[i] / q[i]);
    return sum;
}
static double MySymmKullback(double[] p, double[] q)
{
    double a = KL(p, q);
    double b = KL(q, p);
    return a + b;
}
static double JensenShannon(double[] p, double[] q)
{
    double[] m = new double[p.Length];
    for (int i = 0; i < p.Length; i++)
        m[i] = 0.5 * (p[i] + q[i]);  // avg of P and Q
    double a = KL(p, m);
    double b = KL(q, m);
    return Math.Sqrt((a + b) / 2);
}
static double Hellinger(double[] p, double[] q)
{
    double sum = 0.0;
    for (int i = 0; i < p.Length; i++)
        sum += Math.Pow(Math.Sqrt(p[i]) - Math.Sqrt(q[i]), 2);
    double result = (1.0 / Math.Sqrt(2.0)) * Math.Sqrt(sum);
    return result;
}
