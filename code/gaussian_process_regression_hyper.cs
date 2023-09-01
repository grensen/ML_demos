// https://jamesmccaffrey.wordpress.com/2023/08/16/yet-another-look-at-gaussian-process-regression-using-csharp/
using System;
using System.IO;

namespace GaussianProcessRegression
{
    class GPRProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nGaussian Process Regression ");
            /*
            // 1. Load data
            Console.WriteLine("\nLoading data from file ");
            string trainFile =
              "..\\..\\..\\Data\\synthetic_train_200.txt";
            double[][] trainX = Utils.MatLoad(trainFile,
              new int[] { 0, 1, 2, 3, 4, 5 }, ',', "#");
            double[] trainY = Utils.MatToVec(Utils.MatLoad(trainFile,
              new int[] { 6 }, ',', "#"));

            string testFile =
              "..\\..\\..\\Data\\synthetic_test_40.txt";
            double[][] testX = Utils.MatLoad(testFile,
              new int[] { 0, 1, 2, 3, 4, 5 }, ',', "#");
            double[] testY = Utils.MatToVec(Utils.MatLoad(testFile,
              new int[] { 6 }, ',', "#"));
            Console.WriteLine("Done ");
            */

            // 1. Load data
            Console.WriteLine("\nLoading Synthetic dataset from array");
            int[] cols = { 0, 1, 2, 3, 4, 5, 6 };
            double[][] trainX = Extract2d(FeedSynthetic200TrainData(), cols[0..^1]);
            double[] trainY = Extract1d(FeedSynthetic200TrainData(), cols[^1..^0]);
            double[][] testX = Extract2d(FeedSynthetic40TestData(), cols[0..^1]);
            double[] testY = Extract1d(FeedSynthetic40TestData(), cols[^1..^0]);

            Console.WriteLine("\nFirst four X predictors: ");
            for (int i = 0; i < 4; ++i)
                Utils.VecShow(trainX[i], 4, 8);
            Console.WriteLine("\nFirst four y targets: ");
            for (int i = 0; i < 4; ++i)
                Console.WriteLine(trainY[i].ToString("F4").PadLeft(8));

            // 2. explore alpha, theta, lenScale hyperparameters
            Console.WriteLine("\nExploring hyperparameters ");
            double[] alphas = new double[] { 0.01, 0.02, 0.03 };
            double[] thetas = new double[] {   6, 8};
            double[] lenScales = new double[] { 1.3, 1.4};

            double trainRMSE = 0.0;
            double testRMSE = 0.0;
            double trainAcc = 0.0;
            double testAcc = 0.0;

            foreach (double t in thetas)
            {
                foreach (double s in lenScales)
                {
                    foreach (double a in alphas)
                    {
                        GPR gprModel = 
                            new GPR(t, s, a, trainX, trainY);
                        trainRMSE = 
                            gprModel.RootMSE(trainX, trainY);
                        testRMSE = 
                            gprModel.RootMSE(testX, testY);

                        trainAcc = gprModel.Accuracy(trainX,
                          trainY, 0.10);
                        testAcc = gprModel.Accuracy(testX,
                          testY, 0.10);

                        Console.Write(" theta = " +
                          t.ToString("F2"));
                        Console.Write("  lenScale = " +
                          s.ToString("F2"));
                        Console.Write("  alpha = " +
                          a.ToString("F4"));
                        Console.Write("  |  train RMSE = " +
                          trainRMSE.ToString("F4"));
                        Console.Write("  test RMSE = " +
                          testRMSE.ToString("F4"));
                        Console.Write("  train acc = " +
                          trainAcc.ToString("F4"));
                        Console.Write("  test acc = " +
                          testAcc.ToString("F4"));
                        Console.WriteLine("");
                    }
                }
            }

            // 3. create and train model
            Console.WriteLine("\nCreating and training GPR model ");
            double theta = 6.0;    // "constant kernel"
            double lenScale = 1.3;  // RBF parameter
            double alpha = 0.01;    // noise
            Console.WriteLine("Setting theta = " +
              theta.ToString("F2") +
              ", lenScale = " +
              lenScale.ToString("F2") +
              ", alpha = " + alpha.ToString("F2"));

            GPR model = new GPR(theta, lenScale, alpha,
              trainX, trainY);  // create and train

            // 4. evaluate model
            Console.WriteLine("\nEvaluate model acc" +
              " (within 0.10 of true)");
            trainAcc = model.Accuracy(trainX, trainY, 0.10);
            testAcc = model.Accuracy(testX, testY, 0.10);
            Console.WriteLine("Train acc = " +
              trainAcc.ToString("F4"));
            Console.WriteLine("Test acc = " +
              testAcc.ToString("F4"));

            // 5. use model for previously unseen data
            Console.WriteLine("\nPredicting for (0.1, 0.2, 0.3," +
              " 0.4, 0.5, 0.6) ");
            double[][] X = Utils.MatCreate(1, 6);
            X[0] = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };

            double[][] results = model.Predict(X);

            Console.WriteLine("Predicted y value: ");
            double[] means = results[0];
            Utils.VecShow(means, 4, 8);
            Console.WriteLine("Predicted std: ");
            double[] stds = results[1];
            Utils.VecShow(stds, 4, 8);

            Console.WriteLine("\nEnd GPR demo ");
            Console.ReadLine();
        } // Main()

        static double[][] Extract2d(double[][] data, int[] columns)
        {
            double[][] result = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                result[i] = new double[columns.Length];
                for (int j = 0; j < columns.Length; j++)
                    result[i][j] = data[i][columns[j]];
            }
            return result;
        }
        static double[] Extract1d(double[][] data, int[] column)
        {
            double[] result = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
                result[i] = data[i][column[0]];
            return result;
        }
        static double[][] FeedSynthetic200TrainData()
        {
            double[][] trainData = new double[][]
            {
                new double[] {-0.1660,0.4406,-0.9998,-0.3953,-0.7065,-0.8153,0.9386 },  // 1
                new double[] {-0.2065,0.0776,-0.1616,0.3704,-0.5911,0.7562,0.4374 },  // 2
                new double[] {-0.9452,0.3409,-0.1654,0.1174,-0.7192,-0.6038,0.8437 },  // 3
                new double[] {0.7528,0.7892,-0.8299,-0.9219,-0.6603,0.7563,0.8765 },  // 4
                new double[] {-0.8033,-0.1578,0.9158,0.0663,0.3838,-0.3690,0.5514 },  // 5
                new double[] {-0.9634,0.5003,0.9777,0.4963,-0.4391,0.5786,0.0463 },  // 6
                new double[] {-0.7935,-0.1042,0.8172,-0.4128,-0.4244,-0.7399,0.6770 },  // 7
                new double[] {-0.0169,-0.8933,0.1482,-0.7065,0.1786,0.3995,0.8482 },  // 8
                new double[] {-0.7953,-0.1719,0.3888,-0.1716,-0.9001,0.0718,0.5987 },  // 9
                new double[] {0.8892,0.1731,0.8068,-0.7251,-0.7214,0.6148,0.5553 },  // 10
                new double[] {-0.2046,-0.6693,0.8550,-0.3045,0.5016,0.4520,0.5287 },  // 11
                new double[] {0.5019,-0.3022,-0.4601,0.7918,-0.1438,0.9297,0.5783 },  // 12
                new double[] {0.3269,0.2434,-0.7705,0.8990,-0.1002,0.1568,0.6709 },  // 13
                new double[] {0.8068,0.1474,-0.9943,0.2343,-0.3467,0.0541,0.8099 },  // 14
                new double[] {0.7719,-0.2855,0.8171,0.2467,-0.9684,0.8589,0.1221 },  // 15
                new double[] {0.8652,0.3936,-0.8680,0.5109,0.5078,0.8460,0.8698 },  // 16
                new double[] {0.4230,-0.7515,-0.9602,-0.9476,-0.9434,-0.5076,0.9699 },  // 17
                new double[] {0.1056,0.6841,-0.7517,-0.4416,0.1715,0.9392,0.9652 },  // 18
                new double[] {0.1221,-0.9627,0.6013,-0.5341,0.6142,-0.2243,0.7474 },  // 19
                new double[] {0.1125,-0.7271,-0.8802,-0.7573,-0.9109,-0.7850,0.9604 },  // 20
                new double[] {-0.5486,0.4260,0.1194,-0.9749,-0.8561,0.9346,0.7454 },  // 21
                new double[] {-0.4953,0.4877,-0.6091,0.1627,0.9400,0.6937,0.9583 },  // 22
                new double[] {-0.5203,-0.0125,0.2399,0.6580,-0.6864,-0.9628,0.5269 },  // 23
                new double[] {0.2127,0.1377,-0.3653,0.9772,0.1595,-0.2397,0.6514 },  // 24
                new double[] {0.1019,0.4907,0.3385,-0.4702,-0.8673,-0.2598,0.6933 },  // 25
                new double[] {0.5055,-0.8669,-0.4794,0.6095,-0.6131,0.2789,0.5812 },  // 26
                new double[] {0.0493,0.8496,-0.4734,-0.8681,0.4701,0.5444,0.9899 },  // 27
                new double[] {0.9004,0.1133,0.8312,0.2831,-0.2200,-0.0280,0.2632 },  // 28
                new double[] {0.2086,0.0991,0.8524,0.8375,-0.2102,0.9265,0.0389 },  // 29
                new double[] {-0.7298,0.0113,-0.9570,0.8959,0.6542,-0.9700,0.9881 },  // 30
                new double[] {-0.6476,-0.3359,-0.7380,0.6190,-0.3105,0.8802,0.7334 },  // 31
                new double[] {0.6895,0.8108,-0.0802,0.0927,0.5972,-0.4286,0.8753 },  // 32
                new double[] {-0.0195,0.1982,-0.9689,0.1870,-0.1326,0.6147,0.9221 },  // 33
                new double[] {0.1557,-0.6320,0.5759,0.2241,-0.8922,-0.1596,0.3223 },  // 34
                new double[] {0.3581,0.8372,-0.9992,0.9535,-0.2468,0.9476,0.6502 },  // 35
                new double[] {0.1494,0.2562,-0.4288,0.1737,0.5000,0.7166,0.9090 },  // 36
                new double[] {0.5102,0.3961,0.7290,-0.3546,0.3416,-0.0983,0.8050 },  // 37
                new double[] {-0.1970,-0.3652,0.2438,-0.1395,0.9476,0.3556,0.8670 },  // 38
                new double[] {-0.6029,-0.1466,-0.3133,0.5953,0.7600,0.8077,0.8145 },  // 39
                new double[] {-0.4953,0.7098,0.0554,0.6043,0.1450,0.4663,0.4205 },  // 40
                new double[] {0.0380,0.5418,0.1377,-0.0686,-0.3146,-0.8636,0.8325 },  // 41
                new double[] {0.9656,-0.6368,0.6237,0.7499,0.3768,0.1390,0.2422 },  // 42
                new double[] {-0.6781,-0.0662,-0.3097,-0.5499,0.1850,-0.3755,0.9766 },  // 43
                new double[] {-0.6141,-0.0008,0.4572,-0.5836,-0.5039,0.7033,0.6395 },  // 44
                new double[] {-0.1683,0.2334,-0.5327,-0.7961,0.0317,-0.0457,0.9878 },  // 45
                new double[] {0.0880,0.3083,-0.7109,0.5031,-0.5559,0.0387,0.6264 },  // 46
                new double[] {0.5706,-0.9553,-0.3513,0.7458,0.6894,0.0769,0.8789 },  // 47
                new double[] {-0.8025,0.3026,0.4070,0.2205,0.5992,-0.9309,0.8814 },  // 48
                new double[] {0.5405,0.4635,-0.4806,-0.4859,0.2646,-0.3094,0.9626 },  // 49
                new double[] {0.5655,0.9809,-0.3995,-0.7140,0.8026,0.0831,0.9948 },  // 50
                new double[] {0.9495,0.2732,0.9878,0.0921,0.0529,-0.7291,0.4988 },  // 51
                new double[] {-0.6792,0.4913,-0.9392,-0.2669,0.7247,0.3854,0.9909 },  // 52
                new double[] {0.3819,-0.6227,-0.1162,0.1632,0.9795,-0.5922,0.9609 },  // 53
                new double[] {0.5003,-0.0860,-0.8861,0.0170,-0.5761,0.5972,0.8313 },  // 54
                new double[] {-0.4053,-0.9448,0.1869,0.6877,-0.2380,0.4997,0.4024 },  // 55
                new double[] {0.9189,0.6079,-0.9354,0.4188,-0.0700,0.8951,0.7782 },  // 56
                new double[] {-0.5571,-0.4659,-0.8371,-0.1428,-0.7820,0.2676,0.8987 },  // 57
                new double[] {0.5324,-0.3151,0.6917,-0.1425,0.6480,0.2530,0.6724 },  // 58
                new double[] {-0.7132,-0.8432,-0.9633,-0.8666,-0.0828,-0.7733,1.0051 },  // 59
                new double[] {-0.0952,-0.0998,-0.0439,-0.0520,0.6063,-0.1952,0.9472 },  // 60
                new double[] {0.8094,-0.9259,0.5477,-0.7487,0.2370,-0.9793,0.8873 },  // 61
                new double[] {0.9024,0.8108,0.5919,0.8305,-0.7089,-0.6845,0.1543 },  // 62
                new double[] {-0.6247,0.2450,0.8116,0.9799,0.4222,0.4636,0.1227 },  // 63
                new double[] {-0.5003,-0.6531,-0.7611,0.6252,-0.7064,-0.4714,0.7072 },  // 64
                new double[] {0.6382,-0.3788,0.9648,-0.4667,0.0673,-0.3711,0.6024 },  // 65
                new double[] {-0.1328,0.0246,0.8778,-0.9381,0.4338,0.7820,0.7793 },  // 66
                new double[] {-0.9454,0.0441,-0.3480,0.7190,0.1170,0.3805,0.7989 },  // 67
                new double[] {-0.4198,-0.9813,0.1535,-0.3771,0.0345,0.8328,0.7482 },  // 68
                new double[] {-0.1471,-0.5052,-0.2574,0.8637,0.8737,0.6887,0.7952 },  // 69
                new double[] {-0.3712,-0.6505,0.2142,-0.1728,0.6327,-0.6297,0.9394 },  // 70
                new double[] {0.4038,-0.5193,0.1484,-0.3020,-0.8861,-0.5424,0.7019 },  // 71
                new double[] {0.0380,-0.6506,0.1414,0.9935,0.6337,0.1887,0.5961 },  // 72
                new double[] {0.9520,0.8031,0.1912,-0.9351,-0.8128,-0.8693,0.6979 },  // 73
                new double[] {0.9507,-0.6640,0.9456,0.5349,0.6485,0.2652,0.2317 },  // 74
                new double[] {0.3375,-0.0462,-0.9737,-0.2940,-0.0159,0.4602,0.9404 },  // 75
                new double[] {-0.7247,-0.9782,0.5166,-0.3601,0.9688,-0.5595,0.8854 },  // 76
                new double[] {-0.3226,0.0478,0.5098,-0.0723,-0.7504,-0.3750,0.6329 },  // 77
                new double[] {0.5403,-0.7393,-0.9542,0.0382,0.6200,-0.9748,0.9951 },  // 78
                new double[] {0.3449,0.3736,-0.1015,0.8296,0.2887,-0.9895,0.7384 },  // 79
                new double[] {0.6608,0.2983,0.3474,0.1570,-0.4518,0.1211,0.4014 },  // 80
                new double[] {0.3435,-0.2951,0.7117,-0.6099,0.4946,-0.4208,0.8328 },  // 81
                new double[] {0.6154,-0.2929,-0.5726,0.5346,-0.3827,0.4665,0.6927 },  // 82
                new double[] {0.4889,-0.5572,-0.5718,-0.6021,-0.7150,-0.2458,0.9428 },  // 83
                new double[] {-0.8389,-0.5366,-0.5847,0.8347,0.4226,0.1078,0.9199 },  // 84
                new double[] {-0.3910,0.6697,-0.1294,0.8469,0.4121,-0.0439,0.6093 },  // 85
                new double[] {-0.1376,-0.1916,-0.7065,0.4586,-0.6225,0.2878,0.6777 },  // 86
                new double[] {0.5086,-0.5785,0.2019,0.4979,0.2764,0.1943,0.5728 },  // 87
                new double[] {0.8906,-0.1489,0.5644,-0.8877,0.6705,-0.6155,0.9314 },  // 88
                new double[] {-0.2098,-0.3998,-0.8398,0.8093,-0.2597,0.0614,0.7874 },  // 89
                new double[] {-0.5871,-0.8476,0.0158,-0.4769,-0.2859,-0.7839,0.9215 },  // 90
                new double[] {0.5751,-0.7868,0.9714,-0.6457,0.1448,-0.9103,0.6820 },  // 91
                new double[] {0.0558,0.4802,-0.7001,0.1022,-0.5668,0.5184,0.7905 },  // 92
                new double[] {0.4458,-0.6469,0.7239,-0.9604,0.7205,0.1178,0.7192 },  // 93
                new double[] {0.4339,0.9747,-0.4438,-0.9924,0.8678,0.7158,0.9754 },  // 94
                new double[] {0.4577,0.0334,0.4139,0.5611,-0.2502,0.5406,0.2290 },  // 95
                new double[] {-0.1963,0.3946,-0.9938,0.5498,0.7928,-0.5214,1.0033 },  // 96
                new double[] {-0.7585,-0.5594,-0.3958,0.7661,0.0863,-0.4266,0.8931 },  // 97
                new double[] {0.2277,-0.3517,-0.0853,-0.1118,0.6563,-0.1473,0.9461 },  // 98
                new double[] {-0.3086,0.3499,-0.5570,-0.0655,-0.3705,0.2537,0.9008 },  // 99
                new double[] {0.5689,-0.0861,0.3125,-0.7363,-0.1340,0.8186,0.8376 },  // 100
                new double[] {0.2110,0.5335,0.0094,-0.0039,0.6858,-0.8644,0.9660 },  // 101
                new double[] {0.0357,-0.6111,0.6959,-0.4967,0.4015,0.0805,0.6900 },  // 102
                new double[] {0.8977,0.2487,0.6760,-0.9841,0.9787,-0.8446,0.9741 },  // 103
                new double[] {-0.9821,0.6455,0.7224,-0.1203,-0.4885,0.6054,0.1633 },  // 104
                new double[] {-0.0443,-0.7313,0.8557,0.7919,-0.0169,0.7134,0.1020 },  // 105
                new double[] {-0.2040,0.0115,-0.6209,0.9300,-0.4116,-0.7931,0.6690 },  // 106
                new double[] {-0.7114,-0.9718,0.4319,0.1290,0.5892,0.0142,0.7851 },  // 107
                new double[] {0.5557,-0.1870,0.2955,-0.6404,-0.3564,-0.6548,0.8685 },  // 108
                new double[] {-0.1827,-0.5172,-0.1862,0.9504,-0.3594,0.9650,0.2623 },  // 109
                new double[] {0.7150,0.2392,-0.4959,0.5857,-0.1341,-0.2850,0.6411 },  // 110
                new double[] {-0.3394,0.3947,-0.4627,0.6166,-0.4094,0.0882,0.6386 },  // 111
                new double[] {0.7768,-0.6312,0.1707,0.7964,-0.1078,0.8437,0.2490 },  // 112
                new double[] {-0.4420,0.2177,0.3649,-0.5436,-0.9725,-0.1666,0.7744 },  // 113
                new double[] {0.5595,-0.6505,-0.3161,-0.7108,0.4335,0.3986,0.9678 },  // 114
                new double[] {0.3770,-0.4932,0.3847,-0.5454,-0.1507,-0.2562,0.8190 },  // 115
                new double[] {0.2633,0.4146,0.2272,0.2966,-0.6601,-0.7011,0.4764 },  // 116
                new double[] {0.0284,0.7507,-0.6321,-0.0743,-0.1421,-0.0054,0.9220 },  // 117
                new double[] {-0.4762,0.6891,0.6007,-0.1467,0.2140,-0.7091,0.8447 },  // 118
                new double[] {0.0192,-0.4061,0.7193,0.3432,0.2669,-0.7505,0.6278 },  // 119
                new double[] {0.8966,0.2902,-0.6966,0.2783,0.1313,-0.0627,0.8403 },  // 120
                new double[] {-0.1439,0.1985,0.6999,0.5022,0.1587,0.8494,0.1350 },  // 121
                new double[] {0.2473,-0.9040,-0.4308,-0.8779,0.4070,0.3369,0.9716 },  // 122
                new double[] {-0.2428,-0.6236,0.4940,-0.3192,0.5906,-0.0242,0.8032 },  // 123
                new double[] {0.2885,-0.2987,-0.5416,-0.1322,-0.2351,-0.0604,0.9201 },  // 124
                new double[] {0.9590,-0.2712,0.5488,0.1055,0.7783,-0.2901,0.7501 },  // 125
                new double[] {-0.9129,0.9015,0.1128,-0.2473,0.9901,-0.8833,0.9796 },  // 126
                new double[] {0.0334,-0.9378,0.1424,-0.6391,0.2619,0.9618,0.7955 },  // 127
                new double[] {0.4169,0.5549,-0.0103,0.0571,-0.6984,-0.2612,0.5340 },  // 128
                new double[] {-0.7156,0.4538,-0.0460,-0.1022,0.7720,0.0552,0.9432 },  // 129
                new double[] {-0.8560,-0.1637,-0.9485,-0.4177,0.0070,0.9319,0.9602 },  // 130
                new double[] {-0.7812,0.3461,-0.0001,0.5542,-0.7128,-0.8336,0.6613 },  // 131
                new double[] {-0.6166,0.5356,-0.4194,-0.5662,-0.9666,-0.2027,0.9244 },  // 132
                new double[] {-0.2378,0.3187,-0.8582,-0.6948,-0.9668,-0.7724,0.9198 },  // 133
                new double[] {-0.3579,0.1158,0.9869,0.6690,0.3992,0.8365,0.1103 },  // 134
                new double[] {-0.9205,-0.8593,-0.0520,-0.3017,0.8745,-0.0209,0.9639 },  // 135
                new double[] {-0.1067,0.7541,-0.4928,-0.4524,-0.3433,0.0951,0.9502 },  // 136
                new double[] {-0.5597,0.3429,-0.7144,-0.8118,0.7404,-0.5263,1.0098 },  // 137
                new double[] {0.0516,-0.8480,0.7483,0.9023,0.6250,-0.4324,0.4214 },  // 138
                new double[] {0.0557,-0.3212,0.1093,0.9488,-0.3766,0.3376,0.2155 },  // 139
                new double[] {-0.3484,0.7797,0.5034,0.5253,-0.0610,-0.5785,0.4579 },  // 140
                new double[] {-0.9170,-0.3563,-0.9258,0.3877,0.3407,-0.1391,0.9870 },  // 141
                new double[] {-0.9203,-0.7304,-0.6132,-0.3287,-0.8954,0.2102,0.9190 },  // 142
                new double[] {0.0241,0.2349,-0.1353,0.6954,-0.0919,-0.9692,0.7485 },  // 143
                new double[] {0.6460,0.9036,-0.8982,-0.5299,-0.8733,-0.1567,0.8185 },  // 144
                new double[] {0.7277,-0.8368,-0.0538,-0.7489,0.5458,0.6828,0.8978 },  // 145
                new double[] {-0.5212,0.9049,0.8878,0.2279,0.9470,-0.3103,0.5292 },  // 146
                new double[] {0.7957,-0.1308,-0.5284,0.8817,0.3684,-0.8702,0.8430 },  // 147
                new double[] {0.2099,0.4647,-0.4931,0.2010,0.6292,-0.8918,0.9813 },  // 148
                new double[] {-0.7390,0.6849,0.2367,0.0626,-0.5034,-0.4098,0.7094 },  // 149
                new double[] {-0.8711,0.7940,-0.5932,0.6525,0.7635,-0.0265,0.9492 },  // 150
                new double[] {0.1969,0.0545,0.2496,0.7101,-0.4357,0.7675,0.1514 },  // 151
                new double[] {-0.5460,0.1920,-0.5211,-0.7372,-0.6763,0.6897,0.9429 },  // 152
                new double[] {0.2044,0.9271,-0.3086,0.1913,0.1980,0.2314,0.8078 },  // 153
                new double[] {-0.6149,0.5059,-0.9854,-0.3435,0.8352,0.1767,1.0019 },  // 154
                new double[] {0.7104,0.2093,0.6452,0.7590,-0.3580,-0.7541,0.2429 },  // 155
                new double[] {-0.7465,0.1796,-0.9279,-0.5996,0.5766,-0.9758,1.0155 },  // 156
                new double[] {-0.3933,-0.9572,0.9950,0.1641,-0.4132,0.8579,0.1326 },  // 157
                new double[] {0.1757,-0.4717,-0.3894,-0.2567,-0.5111,0.1691,0.8829 },  // 158
                new double[] {0.3917,-0.8561,0.9422,0.5061,0.6123,0.5033,0.2117 },  // 159
                new double[] {-0.1087,0.3449,-0.1025,0.4086,0.3633,0.3943,0.7479 },  // 160
                new double[] {0.2372,-0.6980,0.5216,0.5621,0.8082,-0.5325,0.6879 },  // 161
                new double[] {-0.3589,0.6310,0.2271,0.5200,-0.1447,-0.8011,0.6644 },  // 162
                new double[] {-0.7699,-0.2532,-0.6123,0.6415,0.1993,0.3777,0.8791 },  // 163
                new double[] {-0.5298,-0.0768,-0.6028,-0.9490,0.4588,0.4498,0.9867 },  // 164
                new double[] {-0.3392,0.6870,-0.1431,0.7294,0.3141,0.1621,0.5972 },  // 165
                new double[] {0.7889,-0.3900,0.7419,0.8175,-0.3403,0.3661,0.0714 },  // 166
                new double[] {0.7984,-0.8486,0.7572,-0.6183,0.6995,0.3342,0.5188 },  // 167
                new double[] {0.2707,0.6956,0.6437,0.2565,0.9126,0.1798,0.6748 },  // 168
                new double[] {-0.6043,-0.1413,-0.3265,0.9839,-0.2395,0.9854,0.3455 },  // 169
                new double[] {-0.8509,-0.2594,-0.7532,0.2690,-0.1722,0.9818,0.8388 },  // 170
                new double[] {0.8599,-0.7015,-0.2102,-0.0768,0.1219,0.5607,0.8489 },  // 171
                new double[] {-0.4760,0.8216,-0.9555,0.6422,-0.6231,0.3715,0.7498 },  // 172
                new double[] {-0.2896,0.9484,-0.7545,-0.6249,0.7789,0.1668,0.9893 },  // 173
                new double[] {-0.5931,0.7926,0.7462,0.4006,-0.0590,0.6543,0.0882 },  // 174
                new double[] {-0.0083,-0.2730,-0.4488,0.8495,-0.2260,-0.0142,0.6236 },  // 175
                new double[] {-0.2335,-0.4049,0.4352,-0.6183,-0.7636,0.6740,0.6520 },  // 176
                new double[] {0.4883,0.1810,-0.5142,0.2465,0.2767,-0.3449,0.9299 },  // 177
                new double[] {-0.4922,0.1828,-0.1424,-0.2358,-0.7466,-0.5115,0.8690 },  // 178
                new double[] {-0.8413,-0.3943,0.4834,0.2300,0.3448,-0.9832,0.8398 },  // 179
                new double[] {-0.5382,-0.6502,-0.6300,0.6885,0.9652,0.8275,0.9112 },  // 180
                new double[] {-0.3053,0.5604,0.0929,0.6329,-0.0325,0.1799,0.3911 },  // 181
                new double[] {0.0740,-0.2680,0.2086,0.9176,-0.2144,-0.2141,0.3200 },  // 182
                new double[] {0.5813,0.2902,-0.2122,0.3779,-0.1920,-0.7278,0.6309 },  // 183
                new double[] {-0.5641,0.8515,0.3793,0.1976,0.4933,0.0839,0.6464 },  // 184
                new double[] {0.4011,0.8611,0.7252,-0.6651,-0.4737,-0.8568,0.7945 },  // 185
                new double[] {-0.5785,0.0056,-0.7901,-0.2223,0.0760,-0.3216,0.9898 },  // 186
                new double[] {0.1118,0.0735,-0.2188,0.3925,0.3570,0.3746,0.8068 },  // 187
                new double[] {0.2262,0.8715,0.1938,0.9592,-0.1180,0.4792,0.1344 },  // 188
                new double[] {-0.9248,0.5295,0.0366,-0.9894,-0.4456,0.0697,0.9624 },  // 189
                new double[] {0.2992,0.8629,-0.8505,-0.4464,0.8385,0.5300,0.9865 },  // 190
                new double[] {0.1995,0.6659,0.7921,0.9454,0.9970,-0.7207,0.3848 },  // 191
                new double[] {-0.3066,-0.2927,-0.4923,0.8220,0.4513,-0.9481,0.9805 },  // 192
                new double[] {-0.0770,-0.4374,-0.9421,0.7694,0.5420,-0.3405,0.9735 },  // 193
                new double[] {-0.3842,0.8562,0.9538,0.0471,0.9039,0.7760,0.4332 },  // 194
                new double[] {0.0361,-0.2545,0.4207,-0.0887,0.2104,0.9808,0.5813 },  // 195
                new double[] {-0.8220,-0.6302,0.0537,-0.1658,0.6013,0.8664,0.8076 },  // 196
                new double[] {-0.6443,0.7201,0.9148,0.9189,-0.9243,-0.8848,0.1623 },  // 197
                new double[] {-0.2880,0.9074,-0.0461,-0.4435,0.0060,0.2867,0.9211 },  // 198
                new double[] {-0.7775,0.5161,0.7039,0.6885,0.7810,-0.2363,0.3703 },  // 199
                new double[] {-0.5484,0.9426,-0.4308,0.8148,0.7811,0.8450,0.6650 }   // 200
            };
            return trainData;
        }
        static double[][] FeedSynthetic40TestData()
        {
            double[][] testData = new double[][]
            {
               new double[] {-0.6877,0.7594,0.2640,-0.5787,-0.3098,-0.6802,0.9262 },  // 0
               new double[] {-0.6694,-0.6056,0.3821,0.1476,0.7466,-0.5107,0.8790 },  // 1
               new double[] {0.2592,-0.9311,0.0324,0.7265,0.9683,-0.9803,0.9250 },  // 2
               new double[] {-0.9049,-0.9797,-0.0196,-0.9090,-0.4433,0.2799,0.8999 },  // 3
               new double[] {-0.4106,-0.4607,0.1811,-0.2389,0.4050,-0.0078,0.8794 },  // 4
               new double[] {-0.4259,-0.7336,0.8742,0.6097,0.8761,-0.6292,0.6253 },  // 5
               new double[] {0.8663,0.8715,-0.4329,-0.4507,0.1029,-0.6294,0.8922 },  // 6
               new double[] {0.8948,-0.0124,0.9278,0.2899,-0.0314,0.9354,0.2092 },  // 7
               new double[] {-0.7136,0.2647,0.3238,-0.1323,-0.8813,-0.0146,0.6075 },  // 8
               new double[] {-0.4867,-0.2171,-0.5197,0.3729,0.9798,-0.6451,0.9907 },  // 9
               new double[] {0.6429,-0.5380,-0.8840,-0.7224,0.8703,0.7771,0.9871 }, // 10
               new double[] {0.6999,-0.1307,-0.0639,0.2597,-0.6839,-0.9704,0.5031 }, // 11
               new double[] {-0.4690,-0.9691,0.3490,0.1029,-0.3567,0.5604,0.5090 }, // 12
               new double[] {-0.4154,-0.6081,-0.8241,0.7400,-0.8236,0.3674,0.5618 }, // 13
               new double[] {-0.7592,-0.9786,0.1145,0.8142,0.7209,-0.3231,0.8586 }, // 14
               new double[] {0.3393,0.6156,0.7950,-0.0923,0.1157,0.0123,0.6244 }, // 15
               new double[] {0.3840,0.3658,0.0406,0.6569,0.0116,0.6497,0.3585 }, // 16
               new double[] {0.9397,0.4839,-0.4804,0.1625,0.9105,-0.8385,0.9460 }, // 17
               new double[] {-0.8329,0.2383,-0.5510,0.5304,0.1363,0.3324,0.8796 }, // 18
               new double[] {-0.8255,-0.2579,0.3443,-0.6208,0.7915,0.8997,0.8454 }, // 19
               new double[] {0.9231,0.4602,-0.1874,0.4875,-0.4240,-0.3712,0.3832 }, // 20
               new double[] {0.7573,-0.4908,0.5324,0.8820,-0.9979,-0.0478,0.1087 }, // 21
               new double[] {0.3141,0.6866,-0.6325,0.7123,-0.2713,0.7845,0.5467 }, // 22
               new double[] {-0.1647,-0.6616,0.2998,-0.9260,-0.3768,-0.3530,0.8954 }, // 23
               new double[] {0.2149,0.3017,0.6921,0.8552,0.3209,0.1563,0.1767 }, // 24
               new double[] {-0.6918,0.7902,-0.3780,0.0970,0.3641,-0.5271,0.9735 }, // 25
               new double[] {-0.6645,0.0170,0.5837,0.3848,-0.7621,0.8015,0.1184 }, // 26
               new double[] {0.1069,-0.8304,-0.5951,0.7085,0.4119,0.7899,0.8076 }, // 27
               new double[] {-0.3417,0.0560,0.3008,0.1886,-0.5371,-0.1464,0.5610 }, // 28
               new double[] {0.9734,-0.8669,0.4279,-0.3398,0.2509,-0.4837,0.7845 }, // 29
               new double[] {0.3020,-0.2577,-0.4104,0.8235,0.8850,0.2271,0.9088 }, // 30
               new double[] {-0.5766,0.6603,-0.5198,0.2632,0.4215,0.4848,0.9255 }, // 31
               new double[] {-0.2195,0.5197,0.8059,0.1748,-0.8192,-0.7420,0.4317 }, // 32
               new double[] {-0.9212,-0.5169,0.7581,0.9470,0.2108,0.9525,0.1578 }, // 33
               new double[] {-0.9131,0.8971,-0.3774,0.5979,0.6213,0.7200,0.7684 }, // 34
               new double[] {-0.4842,0.8689,0.2382,0.9709,-0.9347,0.4503,0.0743 }, // 35
               new double[] {0.1311,-0.0152,-0.4816,-0.3463,-0.5011,-0.5615,0.9131 }, // 36
               new double[] {-0.8336,0.5540,0.0673,0.4788,0.0308,-0.2001,0.7141 }, // 37
               new double[] {0.9725,-0.9435,0.8655,0.8617,-0.2182,-0.5711,0.1298 }, // 38
               new double[] {0.6064,-0.4921,-0.4184,0.8318,0.8058,0.0708,0.8992 }  // 39
            };
            return testData;
        }

    } // Program

    public class GPR
    {
        public double theta = 0.0;
        public double lenScale = 0.0;
        public double noise = 0.0;
        public double[][] trainX;
        public double[] trainY;
        public double[][] invCovarMat;

        public GPR(double theta, double lenScale,
          double noise, double[][] trainX, double[] trainY)
        {
            this.theta = theta;
            this.lenScale = lenScale;
            this.noise = noise;
            this.trainX = trainX;
            this.trainY = trainY;

            double[][] covarMat =
              this.ComputeCovarMat(this.trainX, this.trainX);
            int n = covarMat.Length;
            for (int i = 0; i < n; ++i)
        covarMat[i][i] += this.noise;  // aka alpha, lambda
            this.invCovarMat = Utils.MatInverse(covarMat);
        } // ctor()

        public double[][] ComputeCovarMat(double[][] X1,
          double[][] X2)
        {
            int n1 = X1.Length; int n2 = X2.Length;
            double[][] result = Utils.MatCreate(n1, n2);
            for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
          result[i][j] = this.KernelFunc(X1[i], X2[j]);
            return result;
        }

        public double KernelFunc(double[] x1, double[] x2)
        {
            // constant * RBF
            int dim = x1.Length;
            double sum = 0.0;  // Euclidean distance squared
            for (int i = 0; i < dim; ++i)
        sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
            double term =
              -1 / (2 * (this.lenScale * this.lenScale));
            return this.theta * Math.Exp(sum * term);
        }

        public double[][] Predict(double[][] X)
        {
            // trainx = (200,6)
            // trainY = (200)
            // this.invCovarMat = (200,200)
            // X = (n,6)

            double[][] result = new double[2][];  // means, stds

            // X = to predict, X* = train X, Y* = trainY as matrix
            // means = K(X,X*)  *  inv(K(X*,X*))   *  Y*

            double[][] a =
              this.ComputeCovarMat(X, this.trainX);  // (n,200)
            double[][] b =
              Utils.MatProduct(a, this.invCovarMat); // (n,200)
            double[][] c = Utils.VecToMat(this.trainY,
              trainY.Length, 1);  //  (200,1)
            double[][] d = Utils.MatProduct(b, c);  // (n,1)
            double[] means = Utils.MatToVec(d);     // (n)

            // sigmas matrix = K(X,X) - [ a * invCoverMat * (a)T ]
            double[][] e = this.ComputeCovarMat(X, X);
            double[][] f = Utils.MatProduct(a, this.invCovarMat);
            double[][] g =
              Utils.MatProduct(f, Utils.MatTranspose(a));
            double[][] h = Utils.MatDifference(e, g);

            int n = h.Length;
            double[] stds = new double[n];  // sqrt of diag elements
            for (int i = 0; i < n; ++i)
        stds[i] = Math.Sqrt(h[i][i]);

            result[0] = means;
            result[1] = stds;

            return result;
        } // Predict()

        public double Accuracy(double[][] dataX, double[] dataY,
          double pctClose)
        {
            int numCorrect = 0; int numWrong = 0;
            // get all predictions
            double[][] results = this.Predict(dataX);

            double[] y_preds = results[0];
            for (int i = 0; i < y_preds.Length; ++i)
      {
                if (Math.Abs(y_preds[i] - dataY[i])
                  < Math.Abs(pctClose * dataY[i]))
          numCorrect += 1;
        else
                    numWrong += 1;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        public double RootMSE(double[][] dataX, double[] dataY)
        {
            // get all predictions
            double[][] results = this.Predict(dataX);
            double[] y_preds = results[0];  // no need stds

            double sumSquaredErr = 0.0;
            for (int i = 0; i < y_preds.Length; ++i)
        sumSquaredErr += (y_preds[i] - dataY[i]) *
          (y_preds[i] - dataY[i]);

            return Math.Sqrt(sumSquaredErr / dataY.Length);
        }

    } // class GPR

    public class Utils
    {
        public static double[][] VecToMat(double[] vec,
          int nRows, int nCols)
        {
            double[][] result = MatCreate(nRows, nCols);
            int k = 0;
            for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
          result[i][j] = vec[k++];
            return result;
        }

        public static double[][] MatCreate(int rows,
          int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
            return result;
        }

        public static double[] MatToVec(double[][] m)
        {
            int rows = m.Length;
            int cols = m[0].Length;
            double[] result = new double[rows * cols];
            int k = 0;
            for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
        {
                result[k++] = m[i][j];
            }
            return result;
        }

        public static double[][] MatProduct(double[][] matA,
          double[][] matB)
        {
            int aRows = matA.Length;
            int aCols = matA[0].Length;
            int bRows = matB.Length;
            int bCols = matB[0].Length;
            if (aCols != bRows)
                throw new Exception("Non-conformable matrices");

            double[][] result = MatCreate(aRows, bCols);

            for (int i = 0; i < aRows; ++i) // each row of A
        for (int j = 0; j < bCols; ++j) // each col of B
          for (int k = 0; k < aCols; ++k)
            result[i][j] += matA[i][k] * matB[k][j];

            return result;
        }

        public static double[][] MatTranspose(double[][] m)
        {
            int nr = m.Length;
            int nc = m[0].Length;
            double[][] result = MatCreate(nc, nr);  // note
            for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
          result[j][i] = m[i][j];
            return result;
        }

        public static double[][] MatDifference(double[][] matA,
          double[][] matB)
        {
            int nr = matA.Length;
            int nc = matA[0].Length;
            double[][] result = MatCreate(nc, nr);  // note
            for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
          result[j][i] = matA[i][j] - matB[i][j];
            return result;
        }

        // -------

        public static double[][] MatInverse(double[][] m)
        {
            // assumes determinant is not 0
            // that is, the matrix does have an inverse
            int n = m.Length;
            double[][] result = MatCreate(n, n); // make a copy
            for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          result[i][j] = m[i][j];

            double[][] lum; // combined lower & upper
            int[] perm;  // out parameter
            MatDecompose(m, out lum, out perm);  // ignore return

            double[] b = new double[n];
            for (int i = 0; i < n; ++i)
      {
                for (int j = 0; j < n; ++j)
          if (i == perm[j])
                    b[j] = 1.0;
                else
                    b[j] = 0.0;

                double[] x = Reduce(lum, b); // 
                for (int j = 0; j < n; ++j)
          result[j][i] = x[j];
            }
            return result;
        }

        static int MatDecompose(double[][] m,
          out double[][] lum, out int[] perm)
        {
            // Crout's LU decomposition for matrix determinant
            // and inverse.
            // stores combined lower & upper in lum[][]
            // stores row permuations into perm[]
            // returns +1 or -1 according to even or odd number
            // of row permutations.
            // lower gets dummy 1.0s on diagonal (0.0s above)
            // upper gets lum values on diagonal (0.0s below)

            // even (+1) or odd (-1) row permutatuions
            int toggle = +1;
            int n = m.Length;

            // make a copy of m[][] into result lu[][]
            lum = MatCreate(n, n);
            for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          lum[i][j] = m[i][j];

            // make perm[]
            perm = new int[n];
            for (int i = 0; i < n; ++i)
        perm[i] = i;

            for (int j = 0; j < n - 1; ++j) // note n-1 
      {
                double max = Math.Abs(lum[j][j]);
                int piv = j;

                for (int i = j + 1; i < n; ++i) // pivot index
        {
                    double xij = Math.Abs(lum[i][j]);
                    if (xij > max)
          {
                        max = xij;
                        piv = i;
                    }
                } // i

                if (piv != j)
                {
                    double[] tmp = lum[piv]; // swap rows j, piv
                    lum[piv] = lum[j];
                    lum[j] = tmp;

                    int t = perm[piv]; // swap perm elements
                    perm[piv] = perm[j];
                    perm[j] = t;

                    toggle = -toggle;
                }

                double xjj = lum[j][j];
                if (xjj != 0.0)
                {
                    for (int i = j + 1; i < n; ++i)
          {
                        double xij = lum[i][j] / xjj;
                        lum[i][j] = xij;
                        for (int k = j + 1; k < n; ++k)
              lum[i][k] -= xij * lum[j][k];
                    }
                }

            } // j

            return toggle;  // for determinant
        } // MatDecompose

        static double[] Reduce(double[][] luMatrix,
          double[] b) // helper
        {
            int n = luMatrix.Length;
            double[] x = new double[n];
            //b.CopyTo(x, 0);
            for (int i = 0; i < n; ++i)
        x[i] = b[i];

            for (int i = 1; i < n; ++i)
      {
                double sum = x[i];
                for (int j = 0; j < i; ++j)
          sum -= luMatrix[i][j] * x[j];
                x[i] = sum;
            }

            x[n - 1] /= luMatrix[n - 1][n - 1];
            for (int i = n - 2; i >= 0; --i)
      {
                double sum = x[i];
                for (int j = i + 1; j < n; ++j)
          sum -= luMatrix[i][j] * x[j];
                x[i] = sum / luMatrix[i][i];
            }

            return x;
        } // Reduce

        // -------

        static int NumNonCommentLines(string fn,
          string comment)
        {
            int ct = 0;
            string line = "";
            FileStream ifs = new FileStream(fn,
              FileMode.Open);
            StreamReader sr = new StreamReader(ifs);
            while ((line = sr.ReadLine()) != null)
                if (line.StartsWith(comment) == false)
                    ++ct;
            sr.Close(); ifs.Close();
            return ct;
        }

        public static double[][] MatLoad(string fn,
          int[] usecols, char sep, string comment)
        {
            // count number of non-comment lines
            int nRows = NumNonCommentLines(fn, comment);

            int nCols = usecols.Length;
            double[][] result = MatCreate(nRows, nCols);
            string line = "";
            string[] tokens = null;
            FileStream ifs = new FileStream(fn, FileMode.Open);
            StreamReader sr = new StreamReader(ifs);

            int i = 0;
            while ((line = sr.ReadLine()) != null)
            {
                if (line.StartsWith(comment) == true)
                    continue;
                tokens = line.Split(sep);
                for (int j = 0; j < nCols; ++j)
        {
                    int k = usecols[j];  // into tokens
                    result[i][j] = double.Parse(tokens[k]);
                }
                ++i;
            }
            sr.Close(); ifs.Close();
            return result;
        }

        public static void MatShow(double[][] m,
          int dec, int wid)
        {
            for (int i = 0; i < m.Length; ++i)
      {
                for (int j = 0; j < m[0].Length; ++j)
        {
                    double v = m[i][j];
                    if (Math.Abs(v) < 1.0e-8) v = 0.0; // hack
                    Console.Write(v.ToString("F" +
                      dec).PadLeft(wid));
                }
                Console.WriteLine("");
            }
        }

        public static void VecShow(int[] vec, int wid)
        {
            for (int i = 0; i < vec.Length; ++i)
        Console.Write(vec[i].ToString().PadLeft(wid));
            Console.WriteLine("");
        }

        public static void VecShow(double[] vec,
          int dec, int wid)
        {
            for (int i = 0; i < vec.Length; ++i)
      {
                double x = vec[i];
                if (Math.Abs(x) < 1.0e-8) x = 0.0;  // hack
                Console.Write(x.ToString("F" +
                  dec).PadLeft(wid));
            }
            Console.WriteLine("");
        }

    } // class Utils

} // ns

