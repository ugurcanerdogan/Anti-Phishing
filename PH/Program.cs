using Accord.MachineLearning;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace PH
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // Checking given dataset directory argument and its subdirectories.
            string mainPath = Helpers.CheckDirectory(args[Array.IndexOf(args, "-dataset") + 1]);
            string trainPath = Helpers.CheckDirectory(mainPath + @"\train\");
            string valPath = Helpers.CheckDirectory(mainPath + @"\val\");

            // Checking given mode argument.
            string mode = Helpers.CheckMode(args[Array.IndexOf(args, "-mode") + 1]);
            DescriptorBase descriptorBaseObj = new DescriptorBase();

            switch (mode)
            {
                case "precompute":
                    // Creating folder for precomputed.csv files.
                    Directory.CreateDirectory("pre-computed");
                    // Saving all train and val image paths.
                    Tuple<List<string>, List<string>> trainImagePathsAndLabels = Helpers.ProcessDirectory(trainPath, "train");
                    Tuple<List<string>, List<string>> valImagePathsAndLabels = Helpers.ProcessDirectory(valPath, "val");
                    // Print main dataset path.
                    Helpers.PrintMainPath(mainPath);
                    // Print number of the classes except 'other' class.
                    Console.WriteLine("{0} classes exist on the train set", trainImagePathsAndLabels.Item2.Distinct().Count() - 1);

                    // Compute FCTH data for train&val and save.
                    descriptorBaseObj.ComputeFCTHandSave(trainImagePathsAndLabels.Item1, "train");
                    descriptorBaseObj.ComputeFCTHandSave(valImagePathsAndLabels.Item1, "val");

                    // Compute CEDD data for train&val and save. 
                    descriptorBaseObj.ComputeCEDDandSave(trainImagePathsAndLabels.Item1, "train");
                    descriptorBaseObj.ComputeCEDDandSave(valImagePathsAndLabels.Item1, "val");

                    // Compute SURF data for train&val and save. 
                    Tuple<double[,], KMeansClusterCollection> BOWVandKmeansForSURF = descriptorBaseObj.ComputeSURFForTrainSet(trainImagePathsAndLabels.Item1, true);
                    descriptorBaseObj.ComputeSURFForTestSetandSave(valImagePathsAndLabels.Item1, BOWVandKmeansForSURF.Item2);
                    break;

                case "trainval":

                    // Checking missing CSV files for descriptors. Recompute the missing files, if there is.
                    descriptorBaseObj.CalculateForMissingDescriptorFiles(Helpers.ProcessDirectory(trainPath, "train", false), Helpers.ProcessDirectory(valPath, "val", false));

                    // Reading FCTH samples and labels for train&val sets.
                    Tuple<double[][], string[]> FCTHTrainData = Helpers.ReadFromCSV(@"pre-computed\precomputed_FCTH_train.csv");
                    Tuple<double[][], string[]> FCTHValData = Helpers.ReadFromCSV(@"pre-computed\precomputed_FCTH_val.csv");

                    // Reading CEDD samples and labels for train&val sets.
                    Tuple<double[][], string[]> CEDDTrainData = Helpers.ReadFromCSV(@"pre-computed\precomputed_CEDD_train.csv");
                    Tuple<double[][], string[]> CEDDValData = Helpers.ReadFromCSV(@"pre-computed\precomputed_CEDD_val.csv");

                    // Reading SURF samples and labels for train&val sets.
                    Tuple<double[][], string[]> SURFTrainData = Helpers.ReadFromCSV(@"pre-computed\precomputed_SURF_train.csv");
                    Tuple<double[][], string[]> SURFValData = Helpers.ReadFromCSV(@"pre-computed\precomputed_SURF_val.csv");

                    ClassifierBase classifierBase = new ClassifierBase();

                    // FCTH PART
                    // Deleting zero ranged columns for Decision Tree-based algorithms to prevent error.
                    double[][] ProcessedXTrain = ClassifierBase.DeleteColumnsWithZeroRange(FCTHTrainData.Item1);
                    double[][] ProcessedXVal = ClassifierBase.DeleteColumnsWithZeroRange(FCTHValData.Item1);
                    // Encoding FCTH labels. string --> int
                    int[] encodedTrainLabels = ClassifierBase.EncodeLabels(FCTHTrainData.Item2);
                    int[] encodedValLabels = ClassifierBase.EncodeLabels(FCTHValData.Item2);
                    Console.WriteLine("Training with precomputed_FCTH_train.csv");
                    Stopwatch stopWatch1 = new Stopwatch(); // StopWatch initalization and start.
                    stopWatch1.Start();
                    // Learning phase for all classifiers.
                    RandomForest rf = ClassifierBase.RFFit(ProcessedXTrain, encodedTrainLabels);
                    MulticlassSupportVectorMachine<Linear> svm = ClassifierBase.SVMFit(FCTHTrainData.Item1, encodedTrainLabels);
                    DecisionTree c45 = ClassifierBase.C45Fit(ProcessedXTrain, encodedTrainLabels);
                    stopWatch1.Stop(); // StopWatch stopped.
                    TimeSpan ts1 = stopWatch1.Elapsed;
                    double elapsedTime1 = Math.Round(ts1.TotalSeconds, 2);
                    Console.WriteLine("Done in {0} seconds", elapsedTime1);
                    Console.WriteLine("Testing with precomputed_FCTH_val.csv {0} samples", FCTHValData.Item2.Length);
                    // Predicting phase for all classifiers.
                    int[] predictedRF = ClassifierBase.RFPredict(ProcessedXVal, rf);
                    int[] predictedSVM = ClassifierBase.SVMPredict(FCTHValData.Item1, svm);
                    int[] predictedC45 = ClassifierBase.C45Predict(ProcessedXVal, c45);
                    // Printing performance metrics.
                    ClassifierBase.ComputeMetrics("Random Forest", predictedRF, encodedValLabels);
                    ClassifierBase.ComputeMetrics("SVM", predictedSVM, encodedValLabels);
                    ClassifierBase.ComputeMetrics("C4.5", predictedC45, encodedValLabels);

                    // CEDD PART
                    // Deleting zero ranged columns for Decision Tree-based algorithms to prevent error.
                    ProcessedXTrain = ClassifierBase.DeleteColumnsWithZeroRange(CEDDTrainData.Item1);
                    ProcessedXVal = ClassifierBase.DeleteColumnsWithZeroRange(CEDDValData.Item1);
                    // Encoding CEDD labels. string --> int
                    encodedTrainLabels = ClassifierBase.EncodeLabels(CEDDTrainData.Item2);
                    encodedValLabels = ClassifierBase.EncodeLabels(CEDDValData.Item2);
                    Console.WriteLine("------------------------------------------------");
                    Console.WriteLine("Training with precomputed_CEDD_train.csv");
                    Stopwatch stopWatch2 = new Stopwatch(); // StopWatch initalization and start.
                    stopWatch2.Start();
                    // Learning phase for all classifiers.
                    rf = ClassifierBase.RFFit(ProcessedXTrain, encodedTrainLabels);
                    svm = ClassifierBase.SVMFit(CEDDTrainData.Item1, encodedTrainLabels);
                    c45 = ClassifierBase.C45Fit(ProcessedXTrain, encodedTrainLabels);
                    stopWatch2.Stop(); // StopWatch stopped.
                    TimeSpan ts2 = stopWatch2.Elapsed;
                    double elapsedTime2 = Math.Round(ts2.TotalSeconds, 2);
                    Console.WriteLine("Done in {0} seconds", elapsedTime2);
                    Console.WriteLine("Testing with precomputed_CEDD_val.csv {0} samples", CEDDValData.Item2.Length);
                    // Predicting phase for all classifiers.
                    predictedRF = ClassifierBase.RFPredict(ProcessedXVal, rf);
                    predictedSVM = ClassifierBase.SVMPredict(CEDDValData.Item1, svm);
                    predictedC45 = ClassifierBase.C45Predict(ProcessedXVal, c45);
                    // Printing performance metrics.
                    ClassifierBase.ComputeMetrics("Random Forest", predictedRF, encodedValLabels);
                    ClassifierBase.ComputeMetrics("SVM", predictedSVM, encodedValLabels);
                    ClassifierBase.ComputeMetrics("C4.5", predictedC45, encodedValLabels);

                    // SURF PART
                    // Deleting zero ranged columns for Decision Tree-based algorithms to prevent error.
                    ProcessedXTrain = ClassifierBase.DeleteColumnsWithZeroRange(SURFTrainData.Item1);
                    ProcessedXVal = ClassifierBase.DeleteColumnsWithZeroRange(SURFValData.Item1);
                    // Encoding SURF labels. string --> int
                    encodedTrainLabels = ClassifierBase.EncodeLabels(SURFTrainData.Item2);
                    encodedValLabels = ClassifierBase.EncodeLabels(SURFValData.Item2);
                    Console.WriteLine("------------------------------------------------");
                    Console.WriteLine("Training with precomputed_SURF_train.csv");
                    Stopwatch stopWatch3 = new Stopwatch(); // StopWatch initalization and start.
                    stopWatch3.Start();
                    // Learning phase for all classifiers.
                    rf = ClassifierBase.RFFit(ProcessedXTrain, encodedTrainLabels);
                    svm = ClassifierBase.SVMFit(SURFTrainData.Item1, encodedTrainLabels);
                    c45 = ClassifierBase.C45Fit(ProcessedXTrain, encodedTrainLabels);
                    stopWatch3.Stop(); // StopWatch stopped.
                    TimeSpan ts3 = stopWatch3.Elapsed;
                    double elapsedTime3 = Math.Round(ts3.TotalSeconds, 2);
                    Console.WriteLine("Done in {0} seconds", elapsedTime3);
                    Console.WriteLine("Testing with precomputed_SURF_val.csv {0} samples", SURFValData.Item2.Length);
                    // Predicting phase for all classifiers.
                    predictedRF = ClassifierBase.RFPredict(ProcessedXVal, rf);
                    predictedSVM = ClassifierBase.SVMPredict(SURFValData.Item1, svm);
                    predictedC45 = ClassifierBase.C45Predict(ProcessedXVal, c45);
                    // Printing performance metrics.
                    ClassifierBase.ComputeMetrics("Random Forest", predictedRF, encodedValLabels);
                    ClassifierBase.ComputeMetrics("SVM", predictedSVM, encodedValLabels);
                    ClassifierBase.ComputeMetrics("C4.5", predictedC45, encodedValLabels);
                    break;
            }
            Console.WriteLine("Program finished. Press 'Enter' to terminate...");
            Console.ReadLine();
        }
    }
}