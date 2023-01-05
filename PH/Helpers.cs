using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace PH
{
    public class Helpers
    {
        public static void PrintMainPath(string mainPath)
        {
            string[] tokens = mainPath.Split(new[] { @"\" }, StringSplitOptions.None);
            string pathToRead = tokens[tokens.Length - 1];
            Console.WriteLine("Reading " + pathToRead + "...");
        }
        public static string CheckMode(string mode)
        {
            if (!(mode.Equals("precompute") || mode.Equals("trainval")))
            {
                throw new Exception("Please enter a valid program mode (either 'precompute' or 'trainval').");
            }
            else
            {
                return mode;
            }
        }

        public static string CheckDirectory(string mainPath)
        {
            if (Directory.Exists(mainPath))
            {
                return mainPath;
            }
            else
            {
                throw new Exception("Please enter a valid directory/file path.");
            }
        }

        public static Tuple<List<string>, List<string>> ProcessDirectory(string path, string folderType, bool printToConsole = true)
        {
            List<string> imagePaths = new List<string>();
            List<string> imageLabels = new List<string>();

            if (Directory.Exists(path))
            {
                // Recurse into subdirectories of this directory.
                string[] subdirectoryEntries = Directory.GetDirectories(path);
                foreach (string subdirectory in subdirectoryEntries)
                {
                    string[] fileEntries = Directory.GetFiles(subdirectory);
                    foreach (string fileName in fileEntries)
                    {
                        imagePaths.Add(fileName);
                        imageLabels.Add(GetLabel(fileName));
                    }
                }
            }
            else
            {
                Console.WriteLine("{0} is not a valid file or directory.", path);
            }

            if (printToConsole)
                Console.WriteLine("{0} images were found in {1} folder", imagePaths.Count, folderType);

            return Tuple.Create(imagePaths, imageLabels);
        }

        public static string[] findMissingPrecomputedFiles()
        {
            string[] preComputedFiles = {
                @"pre-computed\precomputed_CEDD_train.csv",
                @"pre-computed\precomputed_CEDD_val.csv",
                @"pre-computed\precomputed_FCTH_train.csv",
                @"pre-computed\precomputed_FCTH_val.csv",
                @"pre-computed\precomputed_SURF_train.csv",
                @"pre-computed\precomputed_SURF_val.csv" };

            string[] fileEntries = { };
            string[] missingPrecomputedFiles = { };
            if (Directory.Exists("pre-computed"))
            {
                Console.WriteLine("'Pre-computed' folder found.");
                fileEntries = Directory.GetFiles("pre-computed");
                missingPrecomputedFiles = preComputedFiles.Except(fileEntries).ToArray();
                if (missingPrecomputedFiles.Length != 0)
                {
                    foreach (string missingFile in missingPrecomputedFiles)
                    {
                        Console.WriteLine("File: {0} not found!", missingFile);
                    }
                }
                else
                {
                    Console.WriteLine("No missing CSV files.");
                }
            }
            else
            {
                Console.WriteLine("'Pre-computed' folder not found!");
                Directory.CreateDirectory("pre-computed");
                return preComputedFiles;
            }
            return missingPrecomputedFiles;
        }

        public static void SaveArrayAsCSV(Array imgDescArray, string csvFileName, string label)
        {
            using (StreamWriter file = new StreamWriter(csvFileName, true))
            {
                WriteArrayToCSV(imgDescArray, file, label);
            }
        }

        private static void WriteArrayToCSV(Array arr1, TextWriter file, string label)
        {
            foreach (var item in arr1)
            {
                double doubleItem = (double)item;
                string stringItem = doubleItem.ToString(new CultureInfo("en-us", false));
                file.Write(stringItem + ",");
            }
            file.Write(label);
            file.Write(Environment.NewLine);
        }

        public static void WriteHeaderToCSV(string filePath, int featureNumber)
        {
            string headerText = "f1";
            for (int i = 2; i < featureNumber + 1; i++)
            {
                headerText += ",f" + i;
            }
            headerText += ",label";

            using (StreamWriter file = new StreamWriter(filePath, true))
            {
                file.Write(headerText);
                file.Write(Environment.NewLine);
            }
        }

        public static Tuple<double[][], string[]> ReadFromCSV(string path)
        {
            string[] lines = File.ReadAllLines(path);
            double[][] sampleList = new double[lines.Length - 1][];
            string[] labelList = new string[lines.Length - 1];

            // Start from 1 to skip header.
            for (int i = 1; i < lines.Length; i++)
            {
                string line = lines[i];
                string[] columns = line.Split(',');
                double[] featureList = new double[columns.Length - 1];
                for (int j = 0; j < columns.Length; j++)
                {
                    string column = columns[j];
                    try
                    {
                        featureList[j] = Convert.ToDouble(column);
                    }
                    catch (FormatException fe)
                    {
                        labelList[i - 1] = column;
                    }
                }
                sampleList[i - 1] = featureList;
            }
            return Tuple.Create(sampleList, labelList);
        }

        public static string GetLabel(string fullPath)
        {
            string[] tokens = fullPath.Split(new[] { @"\" }, StringSplitOptions.None);
            return tokens[tokens.Length - 2];
        }
    }
}
