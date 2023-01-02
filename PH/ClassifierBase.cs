using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Analysis;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace PH
{
    public class ClassifierBase
    {
        MulticlassSupportVectorLearning<Linear> SVM;
        RandomForestLearning RF;
        C45Learning C45;

        public ClassifierBase()
        {
        }

        public MulticlassSupportVectorMachine<Linear> SVMFit(double[][] trainX, int[] trainY)
        {
            SVM = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };
            var svm = SVM.Learn(trainX, trainY);
            return svm;
        }

        public int[] SVMPredict(double[][] testX, MulticlassSupportVectorMachine<Linear> svm)
        {
            int[] predicted = svm.Decide(testX);
            return predicted;
        }

        public RandomForest RFFit(double[][] trainX, int[] trainY)
        {
            RF = new RandomForestLearning()
            {
                NumberOfTrees = 10, // use 10 trees in the forest
            };

            var rf = RF.Learn(trainX, trainY);
            return rf;
        }

        public int[] RFPredict(double[][] testX, RandomForest rf)
        {
            int[] predicted = rf.Decide(testX);
            return predicted;
        }

        public DecisionTree C45Fit(double[][] trainX, int[] trainY)
        {
            C45 = new C45Learning();
            var c45 = C45.Learn(trainX, trainY);
            return c45;
        }

        public int[] C45Predict(double[][] testX, DecisionTree c45)
        {
            int[] predicted = c45.Decide(testX);
            return predicted;
        }

        public static void ComputeMetrics(string modelName, int[] outputs, int[] predicted)
        {
            var confMat = new GeneralConfusionMatrix(classes: 15, expected: outputs, predicted: predicted).Matrix;
            List<double> tpr = new List<double>();
            List<double> fpr = new List<double>();
            List<double> F1s = new List<double>();

            for (int c = 0; c < 15; c++)
            {
                var metricUnits = new Dictionary<string, double>(){
                {"TP", 0}, {"FP", 0}, {"FN", 0}, {"TN", 0}};
                for (int i = 0; i < 15; i++)
                {
                    for (int j = 0; j < 15; j++)
                    {
                        if (i == j && c == i)
                        {
                            metricUnits["TP"] += confMat[i, j];
                        }
                        else if (c == j)
                        {
                            metricUnits["FP"] += confMat[i, j];
                        }
                        else if (c == i)
                        {
                            metricUnits["FN"] += confMat[i, j];
                        }
                        else
                        {
                            metricUnits["TN"] += confMat[i, j];
                        }
                    }
                }
                double TPR = metricUnits["TP"] / (metricUnits["TP"] + metricUnits["FN"]);
                tpr.Add(TPR);

                double FPR = metricUnits["FP"] / (metricUnits["FP"] + metricUnits["TN"]);
                fpr.Add(FPR);

                double F1 = (2 * metricUnits["TP"]) / (2 * metricUnits["TP"] + metricUnits["FP"] + metricUnits["FN"]);
                F1s.Add(F1);
            }

            string averageTPR = (tpr.Count > 0 ? tpr.Average() : 0.0).ToString("F3", new CultureInfo("en-us", false));
            string averageFPR = (fpr.Count > 0 ? fpr.Average() : 0.0).ToString("F3", new CultureInfo("en-us", false));
            string averageF1 = (F1s.Count > 0 ? F1s.Average() : 0.0).ToString("F3", new CultureInfo("en-us", false));
            Console.WriteLine(String.Format("{0,-14}| {1,-5}{2:0.000} | {3,-4}{4:0.000} | {5,-4}{6:0.000}", modelName, "TPR", averageTPR, "FPR", averageFPR, "F1", averageF1));

        }

        public static int[] EncodeLabels(string[] y)
        {

            int[] encodedLabels = new int[y.Length];
            var labelEncodes = new Dictionary<string, int>(){
                {"adobe", 0}, {"alibaba", 1}, {"amazon", 2}, {"apple", 3}, {"boa", 4},
                {"chase", 5}, {"dhl", 6}, {"dropbox", 7}, {"facebook", 8}, {"linkedin", 9},
                {"microsoft", 10}, {"other", 11}, {"paypal", 12}, {"wellsfargo", 13}, {"yahoo", 14}};

            for (int i = 0; i < y.Length; i++)
            {
                encodedLabels[i] = labelEncodes[y[i]];
            }

            return encodedLabels;
        }

        public static List<int> FindColumnIndicesToDrop(double[][] inputArr)
        {
            List<int> indicesToDrop = new List<int>();
            IList<DecisionVariable> list1 = (IList<DecisionVariable>)DecisionVariable.FromData(inputArr);
            for(int i = 0; i < inputArr[0].Length; i++)
            {
                if (list1[i].Range.Length == 0)
                {
                    indicesToDrop.Add(i);
                }
            }
            return indicesToDrop;
        }

        public static double[][] DeleteColumnsWithZeroRange(double[][] inputArr)
        {
            int[] indexes = FindColumnIndicesToDrop(inputArr).ToArray();
            List<double[]> returningArray = new List<double[]>();

            for (int i = 0; i < inputArr.Length; i++)
            {
                List<double> row = new List<double>();
                for (int j = 0; j < inputArr[0].Length; j++)
                {
                    if (Array.Exists(indexes, elem => elem == j))
                    {
                        continue;
                    }
                    else
                    {
                        row.Add(inputArr[i][j]);
                    }
                }
                returningArray.Add(row.ToArray());
            }
            return returningArray.ToArray();
        }

    }
}
