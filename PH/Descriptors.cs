using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using Accord.Imaging;
using Accord.Math;
using Accord.MachineLearning;
using Accord.Statistics.Filters;

namespace PH
{
    public class Descriptors
    {        
        FCTH_Descriptor.FCTH fcth;
        CEDD_Descriptor.CEDD cedd;
        SpeededUpRobustFeaturesDetector surf;
        KMeans kmeans; // faster..

        public Descriptors()
        {
            fcth = new FCTH_Descriptor.FCTH();            
            cedd = new CEDD_Descriptor.CEDD();
            surf = new SpeededUpRobustFeaturesDetector(threshold: 0.0002f, octaves: 5, initial: 2);
            kmeans = new KMeans(k: 400);
        }

        public double[] GetFCTHTable(Bitmap image)
        {
            return fcth.Apply(image,2);
        }

        public double[] GetCEDDTable(Bitmap image)
        {
            return cedd.Apply(image);
        }

        public List<double[]> ComputeFCTHandSave(List<string> imagePaths, string folderType)
        {
            List<double[]> fcthList = new List<double[]>();
            Console.WriteLine("FCTH features are being extracted for {0}...", folderType);
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            foreach (string imgPath in imagePaths)
            {
                Bitmap bitmapImg = new Bitmap(imgPath);
                double[] fcthTable = GetFCTHTable(bitmapImg);
                string imgLabel = Helpers.GetLabel(imgPath);
                string fileName = @"pre-computed\precomputed_FCTH_" + folderType + ".csv";
                Helpers.SaveArrayAsCSV(fcthTable, fileName, imgLabel);
                fcthList.Add(fcthTable);
            }

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            double elapsedTime = Math.Round(ts.TotalSeconds, 2);

            Console.WriteLine("Done. precomputed_FCTH_{0}.csv is regenerated in {1} seconds", folderType, elapsedTime);
            return fcthList;
        }

        public List<double[]> ComputeCEDDandSave(List<string> imagePaths, string folderType)
        {
            List<double[]> ceddList = new List<double[]>();
            Console.WriteLine("CEDD features are being extracted for {0}...", folderType);
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            foreach (string imgPath in imagePaths)
            {
                Bitmap bitmapImg = new Bitmap(imgPath);
                double[] ceddTable = GetCEDDTable(bitmapImg);
                string imgLabel = Helpers.GetLabel(imgPath);
                string fileName = @"pre-computed\precomputed_CEDD_" + folderType + ".csv";
                Helpers.SaveArrayAsCSV(ceddTable, fileName, imgLabel);
                ceddList.Add(ceddTable);
            }

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            double elapsedTime = Math.Round(ts.TotalSeconds, 2);

            Console.WriteLine("Done. precomputed_CEDD_{0}.csv is regenerated in {1} seconds", folderType, elapsedTime);
            return ceddList;
        }

        public double[,] ComputeSURFandSave(List<string> imagePaths, string folderType)
        {
            List<double[]> vStackedDescList = new List<double[]>();
            List<double[][]> descList = new List<double[][]>();
            List<string> labelList = new List<string>();
            int imgCount = 0;

            Console.WriteLine("SURF features are being extracted for {0}...", folderType);
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            foreach (string imgPath in imagePaths)
            {
                imgCount++;
                string imgLabel = Helpers.GetLabel(imgPath);
                labelList.Add(imgLabel);
                Console.WriteLine("1dede1");
                Bitmap bitmapImg = new Bitmap(imgPath);
                List<SpeededUpRobustFeaturePoint> descriptors = surf.ProcessImage(bitmapImg);
                double[][] surfTable = descriptors.Apply(d => d.Descriptor);
                descList.Add(surfTable);

                foreach (double[] item in surfTable) //vstack
                {
                    vStackedDescList.Add(item);
                    Console.WriteLine("2dede2");
                }
            }

            KMeansClusterCollection kmeans = clusterDescriptors(vStackedDescList.ToArray());
            double[,] allFeaturesBoVW = extractFeatures(kmeans, descList, imgCount);

            int labelCtr = 0;
            for (int i = 0; i < allFeaturesBoVW.GetLength(0); i++)
            {
                Console.WriteLine("3dede3");
                string fileName = @"pre-computed\precomputed_SURF_" + folderType + ".csv";
                Helpers.SaveArrayAsCSV(allFeaturesBoVW.GetRow(i), fileName, labelList[labelCtr]);
                labelCtr++;
            }

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            double elapsedTime = Math.Round(ts.TotalSeconds, 2);
            Console.WriteLine("Done. precomputed_SURF_{0}.csv is regenerated in {1} seconds", folderType, elapsedTime);
            return allFeaturesBoVW;
        }


        public KMeansClusterCollection clusterDescriptors(double[][] input)
        {
            // Compute and retrieve the data centroids
            Console.WriteLine("cluster started");
            KMeansClusterCollection clusters = kmeans.Learn(input);
            Console.WriteLine("cluster learned");
            // Use the centroids to parition all the data
            // // // int[] labels = clusters.Decide(input);
            return clusters;
        }


        public double[,] extractFeatures(KMeansClusterCollection kmeans, List<double[][]> descriptors, int imageCount)
        {
            Console.WriteLine("extract started");
            double[,] totalFeatures = new double[imageCount,400];

            for (int i = 0; i < imageCount; i++)
            {
                for (int j = 0; j < descriptors[i].Length; j++)
                {
                    var feature = descriptors[i][j];
                    int idx = kmeans.Decide(feature);
                    totalFeatures[i, idx] += 1;
                    Console.WriteLine("decided");
                }
            }
            return totalFeatures;
        }
    }
}
