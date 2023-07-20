// https://visualstudiomagazine.com/articles/2021/12/01/traveling-salesman.aspx
// Simulated Annealing with Lin-Kernighan Optimization | SA-LK Optimization

Console.WriteLine("\nBegin TSP simulated annealing Lin-Kernighan optimization demo\n");

int cities = 20;
Console.WriteLine($"Setting num_cities = {cities}");
Console.WriteLine("20! = 2,432,902,008,176,640,000 ");
Console.WriteLine($"Optimal solution is 0, 1, 2, ..., {cities - 1}");
Console.WriteLine($"Optimal solution has total distance = {cities - 1:F1}\n");

int seed = 1337;
int maxIter = 2000;
int optimize = 3;
double startTemperature = 10000.0;
double alpha = 0.99;

Console.WriteLine("Hyperparameter Settings:");
Console.WriteLine($"Seed          = {seed}");
Console.WriteLine($"Iterations    = {maxIter}");
Console.WriteLine($"Optimizations = {optimize}");
Console.WriteLine($"Temperature   = {startTemperature:F1}");
Console.WriteLine($"Alpha         = {alpha:F2}\n");

int[] soln = Solve(cities, new Random(seed), maxIter, optimize, startTemperature, alpha);

Console.WriteLine("Best solution found:");
PrintVector(soln);
double dist = TotalDist(soln);
Console.WriteLine($"Total distance = {dist:F1}, error = {Error(soln):F1}\n");

Console.WriteLine("End demo");

static double TotalDist(int[] route)
{
    double d = 0.0;
    int n = route.Length;
    for (int i = 0; i < n - 1; i++)
        d += route[i] < route[i + 1] ?
            (route[i + 1] - route[i]) * 1.0 : (route[i] - route[i + 1]) * 1.5;
    return d;
}
static double Error(int[] route)
{
    int n = route.Length;
    double d = TotalDist(route);
    double minDist = n - 1;
    return d - minDist;
}
static int[] Adjacent(int[] route, Random rnd)
{
    int n = route.Length;
    int[] result = (int[])route.Clone();
    Swap(result, rnd.Next(n), rnd.Next(n));
    return result;
}
static int[] Solve(int numCities, Random rnd, int maxIter, int optimize, double temperature, double alpha)
{
    Console.WriteLine("Initial guess:");
    int[] soln = GenerateRandomSolution(numCities, rnd);
    PrintVector(soln);
    double dist = TotalDist(soln);
    double err = Error(soln);
    Console.WriteLine($"Total distance = {dist:F1}, error = {err:F1}\n");

    Console.WriteLine("Starting Solve()");
    double currTemperature = temperature;
    int iteration = 0;

    while (iteration < maxIter && err > 0.0)
    {
        int[] adjRoute = Adjacent(soln, rnd);
        double adjErr = Error(adjRoute);

        if (adjErr < err)
        {
            soln = adjRoute;
            err = adjErr;
        }
        else if (rnd.NextDouble() < Math.Exp((err - adjErr) / currTemperature))
        {
            soln = adjRoute;
            err = adjErr;
        }

        for (int i = 0; i < optimize; i++)       
            soln = OptimizeSolution(soln);
        dist = TotalDist(soln);

        if ((iteration + 1) % 10 == 0 || iteration == maxIter - 1 || iteration == 0)
            Console.WriteLine($"iter = {iteration + 1,4} | dist = {dist,6:F2} | error = {err,6:F2} | temp = {currTemperature,8:F2}");

        currTemperature *= alpha;
        iteration++;
    }
    Console.WriteLine($"Solved after {iteration} iterations\n");
    return soln;
}
static int[] OptimizeSolution(int[] route)
{
    int n = route.Length;
    for (int i = 0; i < n - 2; i++)
        for (int j = i + 2; j < n; j++)
        {
            int[] newRoute = PerformLinKernighanOptimization(route, i, j);
            double distBefore = TotalDist(route);
            double distAfter = TotalDist(newRoute);
            if (distAfter < distBefore)
                route = newRoute;
        }
    return route;
}
static int[] PerformLinKernighanOptimization(int[] route, int i, int j)
{
    int[] newRoute = (int[])route.Clone();
    // while (i < j) { Swap(newRoute, i, j); i++; j--; }
    Array.Reverse(newRoute, i, j - i + 1); // Reverse the subarray from index i to j
    return newRoute;
}
static int[] GenerateRandomSolution(int numCities, Random rnd)
{
    int[] soln = new int[numCities];
    for (int i = 0; i < numCities; i++)
        soln[i] = i;
    Shuffle(soln, rnd);
    return soln;
}
static void Shuffle(int[] array, Random rnd)
{
    for (int i = array.Length - 1; i > 0; i--)
        Swap(array, i, rnd.Next(i + 1));
}
static void Swap(int[] a, int i, int j) => (a[j], a[i]) = (a[i], a[j]);
static void PrintVector(int[] vector)
{
    Console.Write("[ ");
    for (int i = 0; i < vector.Length; i++)
    {
        Console.Write($"{vector[i],2}");
        if (i < vector.Length - 1)
            Console.Write(" ");
    }
    Console.WriteLine(" ]");
}

