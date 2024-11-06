using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace 試作品ver6
{
    public partial class Form3 : Form
    {
        public Form3()
        {
            InitializeComponent();
            
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Form4 f4 = new Form4();
            f4.Visible = true;

            //画面を閉じる
            this.Close();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Form2 f2 = new Form2();
            f2.Visible = true;

            //画面を閉じる
            this.Close();
        }

        private void Form3_Load(object sender, EventArgs e)
        {
            pictureBox1.ImageLocation = @"C:\Users\TPC-USER\source\repos\試作品ver6\スライド3.jpg";
        }
    }
}
