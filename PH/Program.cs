using System;
using System.Collections.Generic;
using System.IO;
using System.Drawing;
using System.Globalization;
using System.Linq;
using Accord.Statistics.Filters;
using System.Data;
using Accord.MachineLearning;

namespace PH
{
    public class Program
    {
        static void Main(string[] args)
        {
            string mainPath = Helpers.CheckDirectory(args[Array.IndexOf(args, "-dataset") + 1]);
            string trainPath = Helpers.CheckDirectory(mainPath + @"\train\");
            string valPath = Helpers.CheckDirectory(mainPath + @"\val\");

            string mode = Helpers.CheckMode(args[Array.IndexOf(args, "-mode") + 1]);
            switch (mode)
            {
                case "precompute":
                    Helpers.PrintMainPath(mainPath);

                    Tuple<List<string>, List<string>> trainImagePathsAndLabels = Helpers.ProcessDirectory(trainPath, "train");
                    Tuple<List<string>, List<string>> validImagePathsAndLabels = Helpers.ProcessDirectory(valPath, "val");
                    Console.WriteLine("{0} classes exist on the train set", trainImagePathsAndLabels.Item2.Distinct().Count() - 1);

                    DescriptorBase descriptorBaseObj = new DescriptorBase();

                    // FCTH 
                    //descriptorBaseObj.ComputeFCTHandSave(trainImagePathsAndLabels.Item1, "train");
                    //descriptorBaseObj.ComputeFCTHandSave(validImagePathsAndLabels.Item1, "val");

                    // CEDD 
                    //descriptorBaseObj.ComputeCEDDandSave(trainImagePathsAndLabels.Item1, "train");
                    //descriptorBaseObj.ComputeCEDDandSave(validImagePathsAndLabels.Item1, "val");

                    // SURF 
                    Tuple<double[,], KMeansClusterCollection> BOWVandKmeansForSURF = descriptorBaseObj.ComputeSURFForTrainSetandSave(trainImagePathsAndLabels.Item1);
                    descriptorBaseObj.ComputeSURFForTestSetandSave(validImagePathsAndLabels.Item1, BOWVandKmeansForSURF.Item2);
                    break;
                case "trainval":

                    break;
            }

            Console.WriteLine("---");
            Console.ReadLine();
        }
    }
}