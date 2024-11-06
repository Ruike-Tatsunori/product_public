using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;


namespace 試作品ver6
{
    public partial class Form5 : Form
    {
        public string str;
        public int[] dot = new int[3];
        public string tempdata = "24.34";
        public string pressdata;
        public string humdata;

        Scalar Red = new Scalar(0, 0, 255);
        Scalar Red_Yellow = new Scalar(0, 127, 255);
        Scalar Yellow = new Scalar(0, 255, 255);
        Scalar Yellow_Green = new Scalar(0, 255, 127);
        Scalar Green = new Scalar(0, 255, 0);
        Scalar Green_LightBlue = new Scalar(127, 255, 0);
        Scalar LightBlue = new Scalar(255, 255, 0);
        Scalar LightBlue_Blue = new Scalar(255, 127, 0);
        Scalar Blue = new Scalar(255, 0, 0);
        Scalar Blue_Purple = new Scalar(255, 0, 127);
        Scalar Purple = new Scalar(255, 0, 255);
        Scalar Purple_Red = new Scalar(127, 0, 255);
        Scalar color = new Scalar(0, 0, 0);
        Scalar textcolor = new Scalar(0, 0, 0);

        public Form5()
        {
            InitializeComponent();
        }

        private void Form5_Load(object sender, EventArgs e)
        {
            
            pictureBox1.ImageLocation = @"C:\Users\TPC-USER\source\repos\試作品ver6\スライド5.jpg";
            serialPort1.Open(); 
        }

        private void Form5_FormClosed(object sender, FormClosedEventArgs e)
        {
            serialPort1.Close();
        }

        private void Check_input(object sender, EventArgs e)
        {
            tempdata = str.Substring(0, dot[0] + 3);
            humdata = str.Substring(dot[1] + 3, str.Length - (dot[1] + 3));
            pressdata = str.Substring(dot[0] + 3, str.Length - tempdata.Length - humdata.Length);
            humdata = humdata.Substring(0, humdata.Length - 1);
        }

        private void serialPort1_DataReceived(object sender, System.IO.Ports.SerialDataReceivedEventArgs e)
        {
            str = serialPort1.ReadLine();

            string target = ".";

            dot[0] = str.IndexOf(target);

            for (int i = 1; i < 3; i++)
            {
                dot[i] = str.IndexOf(target, dot[i - 1] + 1);
            }


            this.Invoke(new EventHandler(Check_input));
        }

        private void button1_Click(object sender, EventArgs e)
        {
            //カメラキャプチャ開始
            var camera = new OpenCvSharp.VideoCapture(0);

            //カメラのフレームの大きさを設定
            camera.FrameWidth = 1200;
            camera.FrameHeight = 720;

            //カメラ動画を映すwindowを設定
            using (var camerawindow = new Window("camera"))
            {

                //カメラフレームの宣言
                var cameraFrame = new Mat();



                while (true)
                {
                    //カメラのカメラのフレームを取得
                    camera.Read(cameraFrame);

                    //カメラのフレームを取得できない場合、whileを抜ける
                    if (cameraFrame.Empty()) break;

                    //ArUcoのマーカーの読み込み設定
                    var p_dict = OpenCvSharp.Aruco.CvAruco.GetPredefinedDictionary(OpenCvSharp.Aruco.PredefinedDictionaryName.Dict4X4_50);
                    OpenCvSharp.Point2f[][] corners, rejectedImgPoints;

                    //ArUcoのパラメータ設定
                    var detect_param = OpenCvSharp.Aruco.DetectorParameters.Create();

                    int[] ids;

                    // マーカー検出
                    OpenCvSharp.Aruco.CvAruco.DetectMarkers(cameraFrame, p_dict, out corners, out ids, detect_param, out rejectedImgPoints);

                    // 検出されたマーカ情報の描画
                    //OpenCvSharp.Aruco.CvAruco.DrawDetectedMarkers(cameraFrame, corners, ids, new Scalar(0, 255, 0));

                    if (ids.Length != 0)
                    {
                        //温度表示
                        double radX = (corners[0][0].X + corners[0][1].X) / 2;
                        double radY = (corners[0][0].Y + corners[0][1].Y) / 2 - 100;
                        OpenCvSharp.Point center = new OpenCvSharp.Point(radX, radY);
                        TempColor();
                        Cv2.Circle(cameraFrame, center, 100, color, -1);
                        TextColor();
                        Cv2.PutText(cameraFrame, $"temp:{tempdata}[C]", new OpenCvSharp.Point(radX - 90, radY), HersheyFonts.HersheyComplexSmall, 1, textcolor, 1, LineTypes.AntiAlias);
                        //Cv2.PutText(cameraFrame, $"{cameraFrame.Height}", new OpenCvSharp.Point(radX - 90, radY), HersheyFonts.HersheyComplexSmall, 1, textcolor, 1, LineTypes.AntiAlias);

                        //温度計の針
                        Thermometer_Needle(cameraFrame);
                    }
                    //温度計作成
                    Thermometer(cameraFrame);



                    camerawindow.ShowImage(cameraFrame);

                    //カメラwindowを閉じるための設定
                    int key = Cv2.WaitKey(100);
                    if (key == 27)
                    {

                        break;
                    }// ESC キーで閉じる
                }

            }
        }

        private void TempColor()
        {
            double temp = Convert.ToDouble(tempdata);
            if (temp <= -5)
            {
                color = Blue;
            }
            else if (-5 < temp && temp <= 0)
            {
                color = LightBlue_Blue;
            }
            else if (0 < temp && temp <= 5)
            {
                color = LightBlue;
            }
            else if (5 < temp && temp <= 10)
            {
                color = Green_LightBlue;
            }
            else if (10 < temp && temp <= 15)
            {
                color = Green;
            }
            else if (15 < temp && temp <= 20)
            {
                color = Yellow_Green;
            }
            else if (20 < temp && temp <= 25)
            {
                color = Yellow;
            }
            else if (25 < temp && temp <= 30)
            {
                color = Red_Yellow;
            }
            else
            {
                color = Red;
            }
        }

        private void TextColor()
        {
            double temp = Convert.ToDouble(tempdata);
            if (temp <= -5)
            {
                textcolor = Yellow;
            }
            else if (-5 < temp && temp <= 0)
            {
                textcolor = Red_Yellow;
            }
            else if (0 < temp && temp <= 5)
            {
                textcolor = Red;
            }
            else if (5 < temp && temp <= 10)
            {
                textcolor = Purple_Red;
            }
            else if (10 < temp && temp <= 15)
            {
                textcolor = Purple;
            }
            else if (15 < temp && temp <= 20)
            {
                textcolor = Blue_Purple;
            }
            else if (20 < temp && temp <= 25)
            {
                textcolor = Blue;
            }
            else if (25 < temp && temp <= 30)
            {
                textcolor = LightBlue_Blue;
            }
            else
            {
                textcolor = LightBlue;
            }
        }

        private void Thermometer(Mat cameraFrame)
        {
            Cv2.Rectangle(cameraFrame, new OpenCvSharp.Point(1100, 0), new OpenCvSharp.Point(1400, 80), Red, -1);
            Cv2.Rectangle(cameraFrame, new OpenCvSharp.Point(1100, 81), new OpenCvSharp.Point(1400, 160), Red_Yellow, -1);
            Cv2.Rectangle(cameraFrame, new OpenCvSharp.Point(1100, 161), new OpenCvSharp.Point(1400, 240), Yellow, -1);
            Cv2.Rectangle(cameraFrame, new OpenCvSharp.Point(1100, 241), new OpenCvSharp.Point(1400, 320), Yellow_Green, -1);
            Cv2.Rectangle(cameraFrame, new OpenCvSharp.Point(1100, 321), new OpenCvSharp.Point(1400, 400), Green, -1);
            Cv2.Rectangle(cameraFrame, new OpenCvSharp.Point(1100, 401), new OpenCvSharp.Point(1400, 480), Green_LightBlue, -1);
            Cv2.Rectangle(cameraFrame, new OpenCvSharp.Point(1100, 481), new OpenCvSharp.Point(1400, 560), LightBlue, -1);
            Cv2.Rectangle(cameraFrame, new OpenCvSharp.Point(1100, 561), new OpenCvSharp.Point(1400, 640), LightBlue_Blue, -1);
            Cv2.Rectangle(cameraFrame, new OpenCvSharp.Point(1100, 641), new OpenCvSharp.Point(1400, 720), Blue, -1);

            for (int i = 1; i < 45; i++)
            {
                Cv2.Line(cameraFrame, 1100, i * 16, 1110, i * 16, new Scalar(0, 0, 0), 1, LineTypes.Link8);
            }

            for (int j = 1; j < 9; j++)
            {
                Cv2.Line(cameraFrame, 1100, j * 80, 1120, j * 80, new Scalar(0, 0, 0), 1, LineTypes.Link8);
            }

            for (int k = 0; k < 35; k += 5)
            {
                Cv2.PutText(cameraFrame, $"{k}[C]", new OpenCvSharp.Point(1130, 565 - k * 16), HersheyFonts.HersheyComplexSmall, 0.8, new Scalar(0, 0, 0), 1, LineTypes.AntiAlias);
            }

            Cv2.PutText(cameraFrame, $"-5[C]", new OpenCvSharp.Point(1130, 645), HersheyFonts.HersheyComplexSmall, 0.8, new Scalar(0, 0, 0), 1, LineTypes.AntiAlias);




        }
        private void Thermometer_Needle(Mat cameraFrame)
        {
            double needle = 560 - 80 * Convert.ToDouble(tempdata) / 5;
            OpenCvSharp.Point[] point = new OpenCvSharp.Point[] { new OpenCvSharp.Point(1100, needle), new OpenCvSharp.Point(900, needle + 50), new OpenCvSharp.Point(900, needle - 50) };

            Cv2.FillConvexPoly(cameraFrame, point, new Scalar(255, 255, 255), LineTypes.Link8, 0);

            Cv2.Line(cameraFrame, 1100, Convert.ToInt32(needle), 900, Convert.ToInt32(needle + 50), new Scalar(0, 0, 0), 1, LineTypes.Link8);
            Cv2.Line(cameraFrame, 900, Convert.ToInt32(needle + 50), 900, Convert.ToInt32(needle - 50), new Scalar(0, 0, 0), 1, LineTypes.Link8);
            Cv2.Line(cameraFrame, 1100, Convert.ToInt32(needle), 900, Convert.ToInt32(needle - 50), new Scalar(0, 0, 0), 1, LineTypes.Link8);

            Cv2.PutText(cameraFrame, $"temp:{tempdata}[C]", new OpenCvSharp.Point(910, needle + 5), HersheyFonts.HersheyComplexSmall, 0.8, textcolor, 1, LineTypes.AntiAlias);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Form4 f4 = new Form4();
            f4.Visible = true;

            //画面を閉じる
            this.Close();
        }
    }
}
