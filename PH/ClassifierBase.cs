using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Filters;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PH
{
    public class ClassifierBase
    {
        MulticlassSupportVectorLearning<Linear> SVM;
        RandomForestLearning RF;
        C45Learning C45;

        public ClassifierBase()
        {
            // SVM 
            SVM = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // RandomForest
            RF = new RandomForestLearning()
            {
                NumberOfTrees = 10, // use 10 trees in the forest
            };

            // C4.5
            C45 = new C45Learning();
        }

        public MulticlassSupportVectorMachine<Linear> SVMFit(double[][] trainX, int[] trainY)
        {
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
            var c45 = C45.Learn(trainX, trainY);
            return c45;
        }

        public int[] C45Predict(double[][] testX, DecisionTree c45)
        {
            int[] predicted = c45.Decide(testX);
            return predicted;
        }

        public double ComputeError(int[] outputs, int[] predicted)
        {
            double error = new ZeroOneLoss(outputs).Loss(predicted);
            return error;
        }

        public static int[] EncodeLabels(List<string> y)
        {
            //DataTable table = new DataTable("Labels");
            //table.Columns.Add("Label", typeof(string));
            //foreach (string item in y)
            //{
            //    table.Rows.Add(item);
            //}
            //var codebook = new Codification("Output", labels);
            //return codebook.Translate("Output", labels);

            int[] encodedLabels = new int[y.Count];
            var labelEncodes = new Dictionary<string, int>(){
                {"adobe", 0}, {"alibaba", 1}, {"amazon", 2}, {"apple", 3}, {"boa", 4},
                {"chase", 5}, {"dhl", 6}, {"dropbox", 7}, {"facebook", 8}, {"linkedin", 9},
                {"microsoft", 10}, {"other", 11}, {"paypal", 12}, {"wellsfargo", 13}, {"yahoo", 14}};

            for (int i = 0; i < y.Count; i++)
            {
                encodedLabels[i] = labelEncodes[y[i]];
            }

            return encodedLabels;
        }
    }
}
