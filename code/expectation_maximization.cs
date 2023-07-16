// https://github.com/grensen/ExpectationMaximization

Console.WriteLine("\nBegin expectation maximization coin flip demo\n");
Console.WriteLine("Given a set of 10 coin tosses from coin A and coin B");

Console.WriteLine("\n1. Maximum likelihood      Coin A    Coin B");
Console.WriteLine("B: H T T T H H T H T H               5H, 5T");
Console.WriteLine("A: H H H H T H H H H H     9H, 1T");
Console.WriteLine("A: H T H H H H H T H H     8H, 2T");
Console.WriteLine("B: H T H T T T H H T T               4H, 6T");
Console.WriteLine("A: T H H H T H H H T H     7H, 3T");
Console.WriteLine("                           ======    ======");
Console.WriteLine("                          24H, 6T    9H,11T");

Console.WriteLine("\nGoal is to find MLE of:");
Console.WriteLine("thetaA = 24 / (24 + 6) = 0.80");
Console.WriteLine("thetaB =  9 / (9 + 11) = 0.45");

Console.WriteLine("\n2. Expectation maximization");
Console.WriteLine("EM starts with initial hyperparameters: ");
Console.WriteLine("thetaA = 0.6 and thetaB = 0.5");
Console.WriteLine("\nE-step  Prop A  Prob B   Coin A         Coin B");

double thetaA = 0.6, thetaB = 0.5; // initial guess
int N = 10; // tosses
int[] heads = { 5, 9, 8, 4, 7 };

ExpectationMaximization(heads, N, ref thetaA, ref thetaB);

Console.WriteLine($"\nExpectation maximization after {N} M-steps");
Console.WriteLine($"Prediction EM: thetaA = {thetaA:F2} and thetaB = {thetaB:F2}");
Console.WriteLine("Target MLE:    thetaA = 0.80 and thetaB = 0.45");
Console.WriteLine("\nEnd expectation maximization demo");

void ExpectationMaximization(int[] heads, int N, ref double thetaA, ref double thetaB)
{
    for (int x = 0; x < N; x++) // loop 10 times till converge
    {
        double headsA = 0, tailsA = 0, headsB = 0, tailsB = 0;
        // 2. E-Step
        for (int i = 0; i < heads.Length; i++)
        {
            int head = heads[i], tail = N - head;
            // compute likelihood using binomial distribution
            double lbA = Math.Pow(thetaA, head) * Math.Pow((1.0 - thetaA), tail);
            double lbB = Math.Pow(thetaB, head) * Math.Pow((1.0 - thetaB), tail);

            // normalize by using A/A+B
            double probA = lbA / (lbA + lbB);
            double probB = lbB / (lbA + lbB);

            // accumulate estimated tosses
            headsA += head * probA;
            headsB += head * probB;
            tailsA += tail * probA;
            tailsB += tail * probB;

            if (x == 0)
                Console.WriteLine($"{i,1}\t{probA:F2}    {probB:F2}     {(head * probA):F2}  {(tail * probA):F2}     {(head * probB):F2}  {(tail * probB):F2}");
        }

        // 3. M-Step, normalize 
        thetaA = headsA / (headsA + tailsA);
        thetaB = headsB / (headsB + tailsB);
        if (x == 0)
        {
            Console.WriteLine("                         ==========     ==========");
            Console.WriteLine($"\t\t\t{headsA.ToString("F2"),4}  {tailsA.ToString("F2"),4}    {headsB.ToString("F2"),4}  {tailsB.ToString("F2"),4}\n");
            Console.WriteLine($"M-step {x}: thetaA = {thetaA:F2} and thetaB = {thetaB:F2}");
        }
    }
}