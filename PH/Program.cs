using System;
using System.Collections.Generic;
using System.IO;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using System.Globalization;
using System.Linq;
using Accord.Statistics.Filters;
using System.Data;

namespace PH
{
    public class Program
    {
        static void Main(string[] args)
        {
            //string trainPath = @"..\phishIRIS_DL_Dataset\train\";
            //string validPath = @"..\phishIRIS_DL_Dataset\valid\";

            //Tuple<List<string>, List<string>> trainImagePathsAndLabels = Helpers.ProcessDirectory(trainPath);
            //Tuple<List<string>, List<string>> validImagePathsAndLabels = Helpers.ProcessDirectory(valPath);
            //Console.WriteLine(Helpers.ReadFromCSV(@"precomputed_CEDD_val.csv").Item2[999]); 

            ////

            string mainPath = Helpers.CheckDirectory(args[Array.IndexOf(args, "-dataset") + 1]);
            Helpers.PrintMainPath(mainPath);

            string trainPath = Helpers.CheckDirectory(mainPath + @"\train\");
            string valPath = Helpers.CheckDirectory(mainPath + @"\val\");

            string mode = Helpers.CheckMode(args[Array.IndexOf(args, "-mode") + 1]);

            Tuple<List<string>, List<string>> trainImagePathsAndLabels = Helpers.ProcessDirectory(trainPath, "train");
            Tuple<List<string>, List<string>> validImagePathsAndLabels = Helpers.ProcessDirectory(valPath, "val");
            Console.WriteLine("{0} classes exist on the train set", trainImagePathsAndLabels.Item2.Distinct().Count() - 1);

            Descriptors descriptors = new Descriptors();

            // FCTH 
            //List<double[]> fcthTrainValues = descriptors.computeFCTH(trainImagePathsAndLabels.Item1, "train");
            //List<double[]> fcthValValues = descriptors.computeFCTH(validImagePathsAndLabels.Item1, "val");

            // CEDD 
            //List<double[]> ceddTrainValues = descriptors.computeCEDD(trainImagePathsAndLabels.Item1, "train");
            //List<double[]> ceddValValues = descriptors.computeCEDD(validImagePathsAndLabels.Item1, "val");

            // SURF 
            double[,] surfTrainValues = descriptors.ComputeSURFandSave(trainImagePathsAndLabels.Item1, "train");
            //double[,] surfValValues = descriptors.ComputeSURFandSave(validImagePathsAndLabels.Item1, "val");

            Console.ReadLine();
        }
    }
}