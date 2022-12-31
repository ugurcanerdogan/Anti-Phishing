using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using System.Globalization;

namespace PH
{
    public class Helpers
    {

        public static Tuple<List<string>, List<string>> ProcessDirectory(string path, string folderType)
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
                        //Console.WriteLine("Processed file '{0}'.", fileName);
                        imagePaths.Add(fileName);
                        //Console.WriteLine(GetLabel(fileName));
                        imageLabels.Add(GetLabel(fileName));
                    }
                }
            }
            else
            {
                //Console.WriteLine("{0} is not a valid file or directory.", path);
            }
            Console.WriteLine("{0} images were found in {1} folder", imagePaths.Count, folderType);
            return Tuple.Create(imagePaths, imageLabels);
        }

        public static List<string> GetLabelList(List<string> pathList)
        {
            Console.WriteLine("-----------------------------------");
            List<string> labels = new List<string>();
            foreach (string path in pathList)
            {
                labels.Add(GetLabel(path));

                //Console.WriteLine(GetLabel(path)); 
                //Console.WriteLine("Processed file '{0}'.", path);
            }

            return labels;
        }

        public static string GetLabel(string fullPath)
        {
            string[] tokens = fullPath.Split(new[] { @"\" }, StringSplitOptions.None);
            return tokens[tokens.Length - 2];
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

        public static Tuple<List<List<double>>, List<string>> ReadFromCSV(string path)
        {
            List<List<double>> sampleList = new List<List<double>>();
            List<string> labelList = new List<string>();

            string[] lines = File.ReadAllLines(path);
            foreach (string line in lines)
            {
                List<double> columnList = new List<double>();

                string[] columns = line.Split(',');
                foreach (string column in columns)
                {
                    try {
                        columnList.Add(Convert.ToDouble(column));
                    }
                    catch (FormatException fe)
                    {
                        labelList.Add(column);
                    }
                }
                sampleList.Add(columnList);
            }
            return Tuple.Create(sampleList, labelList);
        }

        public static void SaveSurfArrayAsCSV(Array imgDescArray, string csvFileName)
        {
            using (StreamWriter file = new StreamWriter(csvFileName, true))
            {
                WriteSurfArrayToCSV(imgDescArray, file);
            }
        }

        private static void WriteSurfArrayToCSV(Array arr1, TextWriter file)
        {
            string stringItem = "";
            foreach (var item in arr1)
            {
                //if (item is Array)
                //{
                //    writeArrayToCSV(item as Array, file, label);
                //    file.Write(Environment.NewLine);
                //}
                //else
                //{
                //    double doubleItem = (double)item;
                //    string stringItem = doubleItem.ToString(new CultureInfo("en-us", false));

                //    file.Write(stringItem + ",");
                //}

                double doubleItem = (double)item;
                stringItem += doubleItem.ToString(new CultureInfo("en-us", false)) + ",";


            }
            stringItem = stringItem.Substring(0, stringItem.Length-1);
            file.Write(stringItem);
            file.Write(Environment.NewLine);
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
                //if (item is Array)
                //{
                //    writeArrayToCSV(item as Array, file, label);
                //    file.Write(Environment.NewLine);
                //}
                //else
                //{
                //    double doubleItem = (double)item;
                //    string stringItem = doubleItem.ToString(new CultureInfo("en-us", false));

                //    file.Write(stringItem + ",");
                //}

                double doubleItem = (double)item;
                string stringItem = doubleItem.ToString(new CultureInfo("en-us", false));

                file.Write(stringItem + ",");
            }
            file.Write(label);
            file.Write(Environment.NewLine);
        }

        public static void PrintMainPath(string mainPath)
        {
            string[] tokens = mainPath.Split(new[] { @"\" }, StringSplitOptions.None);
            string pathToRead = tokens[tokens.Length - 1];
            Console.WriteLine("Reading " + pathToRead + "...");
        }
    }
}
