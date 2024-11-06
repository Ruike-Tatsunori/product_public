#include <Wire.h>
//wind sensor
#define BME280_ADDRESS 0x76
unsigned long int hum_raw,temp_raw,pres_raw;
signed long int t_fine;

uint16_t dig_T1;
 int16_t dig_T2;
 int16_t dig_T3;
uint16_t dig_P1;
 int16_t dig_P2;
 int16_t dig_P3;
 int16_t dig_P4;
 int16_t dig_P5;
 int16_t dig_P6;
   
 int16_t dig_P7;
 int16_t dig_P8;
 int16_t dig_P9;
 int8_t  dig_H1;
 int16_t dig_H2;
 int8_t  dig_H3;
 int16_t dig_H4;
 int16_t dig_H5;
 int8_t  dig_H6;
 
double temp_act_ave = 0.0, press_act_ave = 0.0,hum_act_ave =0.0;

//choose sound
volatile int dic=1;
volatile int keep=1;

//practice
volatile int pra=0;
volatile int pre=0;

void setup()
{
    //wind sensor
    uint8_t osrs_t = 1;             //Temperature oversampling x 1
    uint8_t osrs_p = 1;             //Pressure oversampling x 1
    uint8_t osrs_h = 1;             //Humidity oversampling x 1
    uint8_t mode = 3;               //Normal mode
    uint8_t t_sb = 6;               //Tstandby 10ms
    uint8_t filter = 0;             //Filter off 
    uint8_t spi3w_en = 0;           //3-wire SPI Disable
    
    uint8_t ctrl_meas_reg = (osrs_t << 5) | (osrs_p << 2) | mode;
    uint8_t config_reg    = (t_sb << 5) | (filter << 2) | spi3w_en;
    uint8_t ctrl_hum_reg  = osrs_h;
    
    byte i;
    double temp_act = 0.0, press_act = 0.0,hum_act=0.0;
    signed long int temp_cal;
    unsigned long int press_cal,hum_cal;
    pinMode(13,OUTPUT); // led check

    Serial.begin(9600);
    Serial.println("MBE280Samplev2 by Yas. Sensor Init value Please Wait 5 SEC..... ");
    Wire.begin();
    
    writeReg(0xF2,ctrl_hum_reg);
    writeReg(0xF4,ctrl_meas_reg);
    writeReg(0xF5,config_reg);
    readTrim();
// Sensor Wait 3
    delay(3000);
// Sensor ave 5 times 
    for( i=0 ; i<5 ; i++){
        readData();
    
        temp_cal = calibration_T(temp_raw);
        press_cal = calibration_P(pres_raw);
        hum_cal = calibration_H(hum_raw);
        temp_act = (double)temp_cal / 100.0;
        press_act = (double)press_cal / 100.0;
        hum_act = (double)hum_cal / 1024.0;  
        temp_act_ave += temp_act, press_act_ave += press_act, hum_act_ave += hum_act;
    delay(200);
    }
    temp_act_ave = temp_act_ave/5 , press_act_ave = press_act_ave/5 , hum_act_ave = hum_act_ave/5;
    Serial.print("TEMP ave : ");
    Serial.print(temp_act_ave);
    Serial.print(" DegC  PRESS ave : ");
    Serial.print(press_act_ave);
    Serial.print(" hPa  HUM ave : ");
    Serial.print(hum_act_ave);
    Serial.println(" %");
    delay(1000);

    //choose sound
    attachInterrupt(0,dics,FALLING);

    //practice
    attachInterrupt(1,practice,FALLING);

    //switch
    pinMode(4,INPUT);
    pinMode(5,INPUT);
    pinMode(6,INPUT);
    pinMode(14,INPUT);
    pinMode(15,INPUT);
    pinMode(16,INPUT);
    pinMode(7,INPUT);
    pinMode(8,INPUT);

    //LED
    pinMode(9,OUTPUT);
    pinMode(10,OUTPUT);
    pinMode(11,OUTPUT);
    pinMode(12,OUTPUT);
    pinMode(13,OUTPUT);
}


void loop()
{
  Serial.print("dic=1");
    while(dic==1)
  {
    digitalWrite(9,HIGH);
    double temp_act = 0.0, press_act = 0.0,hum_act=0.0;
    signed long int temp_cal;
    unsigned long int press_cal,hum_cal;
    
    readData();
    
    temp_cal = calibration_T(temp_raw);
    press_cal = calibration_P(pres_raw);
    hum_cal = calibration_H(hum_raw);
    temp_act = (double)temp_cal / 100.0;
    press_act = (double)press_cal / 100.0;
    hum_act = (double)hum_cal / 1024.0;
// FUITA hantei
    if ( (temp_act > temp_act_ave ) && (press_act  <= (press_act_ave - 0.25)) && (hum_act >= (hum_act_ave)))
    {
      origin();
      digitalWrite(13,HIGH);
    }
    else
    {
      Serial.write('y');
      digitalWrite(13,LOW);
    }
    delay(30);
}
digitalWrite(9,LOW);
Serial.print("dic=2");
while(dic==2)
{
  digitalWrite(10,HIGH);
  double temp_act = 0.0, press_act = 0.0,hum_act=0.0;
    signed long int temp_cal;
    unsigned long int press_cal,hum_cal;
    
    readData();
    
    temp_cal = calibration_T(temp_raw);
    press_cal = calibration_P(pres_raw);
    hum_cal = calibration_H(hum_raw);
    temp_act = (double)temp_cal / 100.0;
    press_act = (double)press_cal / 100.0;
    hum_act = (double)hum_cal / 1024.0;
// FUITA hantei v2
    if ( (temp_act > temp_act_ave ) && (press_act  <= (press_act_ave - 0.25)) && (hum_act >= (hum_act_ave)))
    {
      tranp();
      digitalWrite(13,HIGH);
    }
    else
    {
      Serial.write('y');
      digitalWrite(13,LOW);
    }
    delay(30);
}
digitalWrite(10,LOW);
Serial.print("dic=3");
while(dic==3)
{
  digitalWrite(11,HIGH);
  double temp_act = 0.0, press_act = 0.0,hum_act=0.0;
    signed long int temp_cal;
    unsigned long int press_cal,hum_cal;
    
    readData();
    
    temp_cal = calibration_T(temp_raw);
    press_cal = calibration_P(pres_raw);
    hum_cal = calibration_H(hum_raw);
    temp_act = (double)temp_cal / 100.0;
    press_act = (double)press_cal / 100.0;
    hum_act = (double)hum_cal / 1024.0;
// FUITA hantei v2
    if ( (temp_act > temp_act_ave ) && (press_act  <= (press_act_ave - 0.25)) && (hum_act >= (hum_act_ave)))
    {
      flute();
      digitalWrite(13,HIGH);
    }
    else
    {
      Serial.write('y');
      digitalWrite(13,LOW);
    }
    delay(30);
}
digitalWrite(11,LOW);
while(dic==4)
{
  digitalWrite(12,HIGH);
  double temp_act = 0.0, press_act = 0.0,hum_act=0.0;
    signed long int temp_cal;
    unsigned long int press_cal,hum_cal;
    
    readData();
    
    temp_cal = calibration_T(temp_raw);
    press_cal = calibration_P(pres_raw);
    hum_cal = calibration_H(hum_raw);
    temp_act = (double)temp_cal / 100.0;
    press_act = (double)press_cal / 100.0;
    hum_act = (double)hum_cal / 1024.0;
// FUITA hantei v2
    if ( (temp_act > temp_act_ave ) && (press_act  <= (press_act_ave - 0.25)) && (hum_act >= (hum_act_ave)))
    {
      download();
      digitalWrite(13,HIGH);
    }
    else
    {
      Serial.write('y');
      digitalWrite(13,LOW);
    }
    delay(30);
}
digitalWrite(12,LOW);
while(pra==1)
{ 
  digitalWrite(9,HIGH);
  Serial.print("pra=1");
  bunbunbun();
}
digitalWrite(9,LOW);
while(pra==2)
{
  digitalWrite(10,HIGH);
  Serial.print("pra=2");
  flog_song();
}
digitalWrite(10,LOW);
while(pra==3)
{
  digitalWrite(11,HIGH);
  Serial.print("pra=3");
  shooting_star();
}
digitalWrite(11,LOW);
}
void readTrim()
{
    uint8_t data[32],i=0;                      
    Wire.beginTransmission(BME280_ADDRESS);
    Wire.write(0x88);
    Wire.endTransmission();
    Wire.requestFrom(BME280_ADDRESS,24);      
    while(Wire.available()){
        data[i] = Wire.read();
        i++;
    }
    
    Wire.beginTransmission(BME280_ADDRESS);    // Add 2014/04/06
    Wire.write(0xA1);                          // Add 2014/04/06
    Wire.endTransmission();                    // Add 2014/04/06
    Wire.requestFrom(BME280_ADDRESS,1);        // Add 2014/04/06
    data[i] = Wire.read();                     // Add 2014/04/06
    i++;                                       // Add 2014/04/06
    
    Wire.beginTransmission(BME280_ADDRESS);
    Wire.write(0xE1);
    
    Wire.endTransmission();
    Wire.requestFrom(BME280_ADDRESS,7);        // Fix 2014/04/06
    while(Wire.available()){
        data[i] = Wire.read();
        i++;    
    }
    dig_T1 = (data[1] << 8) | data[0];
    dig_T2 = (data[3] << 8) | data[2];
    dig_T3 = (data[5] << 8) | data[4];
    dig_P1 = (data[7] << 8) | data[6];
    dig_P2 = (data[9] << 8) | data[8];
    dig_P3 = (data[11]<< 8) | data[10];
    dig_P4 = (data[13]<< 8) | data[12];
    dig_P5 = (data[15]<< 8) | data[14];
    dig_P6 = (data[17]<< 8) | data[16];
    dig_P7 = (data[19]<< 8) | data[18];
    dig_P8 = (data[21]<< 8) | data[20];
    dig_P9 = (data[23]<< 8) | data[22];
    dig_H1 = data[24];
    dig_H2 = (data[26]<< 8) | data[25];
    dig_H3 = data[27];
    dig_H4 = (data[28]<< 4) | (0x0F & data[29]);
    dig_H5 = (data[30] << 4) | ((data[29] >> 4) & 0x0F); // Fix 2014/04/06
    dig_H6 = data[31];                                   // Fix 2014/04/06
}
void writeReg(uint8_t reg_address, uint8_t data)
{
    Wire.beginTransmission(BME280_ADDRESS);
    Wire.write(reg_address);
    Wire.write(data);
    Wire.endTransmission();    
}


void readData()
{
    int i = 0;
    uint32_t data[8];
    Wire.beginTransmission(BME280_ADDRESS);
    Wire.write(0xF7);
    Wire.endTransmission();
    Wire.requestFrom(BME280_ADDRESS,8);
    while(Wire.available()){
        data[i] = Wire.read();
        i++;
    }
    pres_raw = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4);
    temp_raw = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4);
    hum_raw  = (data[6] << 8) | data[7];
}


signed long int calibration_T(signed long int adc_T)
{
    
    signed long int var1, var2, T;
    var1 = ((((adc_T >> 3) - ((signed long int)dig_T1<<1))) * ((signed long int)dig_T2)) >> 11;
    var2 = (((((adc_T >> 4) - ((signed long int)dig_T1)) * ((adc_T>>4) - ((signed long int)dig_T1))) >> 12) * ((signed long int)dig_T3)) >> 14;
    
    t_fine = var1 + var2;
    T = (t_fine * 5 + 128) >> 8;
    return T; 
}

unsigned long int calibration_P(signed long int adc_P)
{
    signed long int var1, var2;
    unsigned long int P;
    var1 = (((signed long int)t_fine)>>1) - (signed long int)64000;
    var2 = (((var1>>2) * (var1>>2)) >> 11) * ((signed long int)dig_P6);
    var2 = var2 + ((var1*((signed long int)dig_P5))<<1);
    var2 = (var2>>2)+(((signed long int)dig_P4)<<16);
    var1 = (((dig_P3 * (((var1>>2)*(var1>>2)) >> 13)) >>3) + ((((signed long int)dig_P2) * var1)>>1))>>18;
    var1 = ((((32768+var1))*((signed long int)dig_P1))>>15);
    if (var1 == 0)
    {
        return 0;
    }    
    P = (((unsigned long int)(((signed long int)1048576)-adc_P)-(var2>>12)))*3125;
    if(P<0x80000000)
    {
       P = (P << 1) / ((unsigned long int) var1);   
    }
    else
    {
        P = (P / (unsigned long int)var1) * 2;    
    }
    var1 = (((signed long int)dig_P9) * ((signed long int)(((P>>3) * (P>>3))>>13)))>>12;
    var2 = (((signed long int)(P>>2)) * ((signed long int)dig_P8))>>13;
    P = (unsigned long int)((signed long int)P + ((var1 + var2 + dig_P7) >> 4));
    return P;
}

unsigned long int calibration_H(signed long int adc_H)
{
    signed long int v_x1;
    
    v_x1 = (t_fine - ((signed long int)76800));
    v_x1 = (((((adc_H << 14) -(((signed long int)dig_H4) << 20) - (((signed long int)dig_H5) * v_x1)) + 
              ((signed long int)16384)) >> 15) * (((((((v_x1 * ((signed long int)dig_H6)) >> 10) * 
              (((v_x1 * ((signed long int)dig_H3)) >> 11) + ((signed long int) 32768))) >> 10) + (( signed long int)2097152)) * 
              ((signed long int) dig_H2) + 8192) >> 14));
   v_x1 = (v_x1 - (((((v_x1 >> 15) * (v_x1 >> 15)) >> 7) * ((signed long int)dig_H1)) >> 4));
   v_x1 = (v_x1 < 0 ? 0 : v_x1);
   v_x1 = (v_x1 > 419430400 ? 419430400 : v_x1);
   return (unsigned long int)(v_x1 >> 12);   
}

void dics()
{  
   dic++;
   if(dic>=5)
   {
      dic = 1;
   }
   keep=dic;
   pra=0;
   delay(1000);
}

void practice()
{  
   pra++;
   if(pra>=4)
   {
      pra = 1;
   }
   dic=0;
   delay(10);
}

void origin()
{
   if(digitalRead(4)==LOW)
   {
     Serial.write('a');
   }
   else if(digitalRead(5)==LOW)
   {
     Serial.write('b');
   }
   else if(digitalRead(6)==LOW)
   {
     Serial.write('c');
   }
   else if(digitalRead(14)==LOW)
   {
     Serial.write('d');
   }
   else if(digitalRead(15)==LOW)
   {
     Serial.write('e');
   }
   else if(digitalRead(16)==LOW)
   {
     Serial.write('f');
   }
   else if(digitalRead(7)==LOW)
   {
     Serial.write('g');
   }
   else if(digitalRead(8)==LOW)
   {
     Serial.write('h');
   }
   else
   {
    Serial.write('z');
   }
}

void tranp()
{
   if(digitalRead(4)==LOW&&digitalRead(5)==LOW&&digitalRead(6)==LOW)
   {
     Serial.write('o');
   }
   else if(digitalRead(5)==LOW&&digitalRead(6)==LOW)
   {
     Serial.write('n');
   }
   else if(digitalRead(4)==LOW&&digitalRead(6)==LOW)
   {
     Serial.write('m');
   }
   else if(digitalRead(4)==LOW&&digitalRead(5)==LOW)
   {
     Serial.write('l');
   }
   else if(digitalRead(6)==LOW)
   {
     Serial.write('k');
   }
   else if(digitalRead(5)==LOW)
   {
     Serial.write('j');
   }
   else if(digitalRead(4)==LOW)
   {
     Serial.write('i');
   }
   else
   {
     Serial.write('p');
   }
}

void flute()
{
   if(digitalRead(4)==LOW)
   {
     Serial.write('q');
   }
   else if(digitalRead(5)==LOW)
   {
     Serial.write('r');
   }
   else if(digitalRead(6)==LOW)
   {
     Serial.write('s');
   }
   else if(digitalRead(14)==LOW)
   {
     Serial.write('t');
   }
   else if(digitalRead(15)==LOW)
   {
     Serial.write('u');
   }
   else if(digitalRead(16)==LOW)
   {
     Serial.write('v');
   }
   else if(digitalRead(7)==LOW)
   {
     Serial.write('w');
   }
   else if(digitalRead(8)==LOW)
   {
     Serial.write('x');
   }
   else
   {
    Serial.write('z');
   }
}

void download()
{
  if(digitalRead(4)==LOW)
   {
     Serial.write('A');
   }
   else if(digitalRead(5)==LOW)
   {
     Serial.write('B');
   }
   else if(digitalRead(6)==LOW)
   {
     Serial.write('C');
   }
   else if(digitalRead(14)==LOW)
   {
     Serial.write('D');
   }
   else if(digitalRead(15)==LOW)
   {
     Serial.write('E');
   }
   else if(digitalRead(16)==LOW)
   {
     Serial.write('F');
   }
   else if(digitalRead(7)==LOW)
   {
     Serial.write('G');
   }
   else if(digitalRead(8)==LOW)
   {
     Serial.write('H');
   }
   else
   {
    Serial.write('z');
   }
}

int push()
{
  if(keep == 2)
  {
    if(digitalRead(4)==LOW)
  {
    return 1;
  }
  else if(digitalRead(5)==LOW)
  {
    return 1;
  }
   else if(digitalRead(6)==LOW)
  {
    return 1;
  }
  else
  {
    return 0;
  }
  }
  if(keep == 1||keep == 3)
  {
  if(digitalRead(4)==LOW)
  {
    return 1;
  }
  else if(digitalRead(5)==LOW)
  {
    return 1;
  }
   else if(digitalRead(6)==LOW)
  {
    return 1;
  }
   else if(digitalRead(14)==LOW)
  {
    return 1;
  }
   else if(digitalRead(15)==LOW)
  {
    return 1;
  }
   else if(digitalRead(16)==LOW)
  {
    return 1;
  }
   else if(digitalRead(7)==LOW)
  {
    return 1;
  }
   else if(digitalRead(8)==LOW)
  {
    return 1;
  }
  else
  {
    return 0;
  }
  }
  
}

void praC()
{
  int oshita=0;
  while(1)
  {
    if(pre != pra)
    {
      break;
    }
    oshita = push();
    if(oshita == 1)
    {
      if(keep == 1)
      {
        Serial.write('a');
      }
      else if(keep == 2)
      {
        Serial.write('i');
      }
      else if(keep == 3)
      {
        Serial.write('q');
      }
      else if(keep == 4)
      {
        Serial.write('A');
      }
      delay(10);
      oshita = push();
      
      if(oshita == 0)
      {
        Serial.write('y');
        break;
      }
      
    }
  }
}

void praD()
{
  int oshita=0;
  while(1)
  {
    if(pre != pra)
    {
      break;
    }
    oshita = push();
    if(oshita == 1)
    {
      if(keep == 1)
      {
        Serial.write('b');
      }
      else if(keep == 2)
      {
        Serial.write('j');
      }
      else if(keep == 3)
      {
        Serial.write('r');
      }
      else if(keep == 4)
      {
        Serial.write('B');
      }
      delay(10);
      oshita = push();
      
      if(oshita == 0)
      {
        Serial.write('y');
        break;
      }
      
    }
  }
}

void praE()
{
  int oshita=0;
  while(1)
  {
    if(pre != pra)
    {
      break;
    }
    oshita = push();
    if(oshita == 1)
    {
      if(keep == 1)
      {
        Serial.write('c');
      }
      else if(keep == 2)
      {
        Serial.write('k');
      }
      else if(keep == 3)
      {
        Serial.write('s');
      }
      else if(keep == 4)
      {
        Serial.write('C');
      }
      delay(10);
      oshita = push();
      
      if(oshita == 0)
      {
        Serial.write('y');
        break;
      }
      
    }
  }
}

void praF()
{
  int oshita=0;
  while(1)
  {
    if(pre != pra)
    {
      break;
    }
    oshita = push();
    if(oshita == 1)
    {
      if(keep == 1)
      {
        Serial.write('d');
      }
      else if(keep == 2)
      {
        Serial.write('l');
      }
      else if(keep == 3)
      {
        Serial.write('t');
      }
      else if(keep == 4)
      {
        Serial.write('D');
      }
      delay(10);
      oshita = push();
      
      if(oshita == 0)
      {
        Serial.write('y');
        break;
      }
      
    }
  }
}

void praG()
{
  int oshita=0;
  while(1)
  {
    if(pre != pra)
    {
      break;
    }
    oshita = push();
    if(oshita == 1)
    {
      if(keep == 1)
      {
        Serial.write('e');
      }
      else if(keep == 2)
      {
        Serial.write('m');
      }
      else if(keep == 3)
      {
        Serial.write('u');
      }
      else if(keep == 4)
      {
        Serial.write('E');
      }
      delay(10);
      oshita = push();
      
      if(oshita == 0)
      {
        Serial.write('y');
        break;
      }
      
    }
  }
}

void praA()
{
  int oshita=0;
  while(1)
  {
    if(pre != pra)
    {
      break;
    }
    oshita = push();
    if(oshita == 1)
    {
      if(keep == 1)
      {
        Serial.write('f');
      }
      else if(keep == 2)
      {
        Serial.write('n');
      }
      else if(keep == 3)
      {
        Serial.write('v');
      }
      else if(keep == 4)
      {
        Serial.write('F');
      }
      delay(10);
      oshita = push();
      
      if(oshita == 0)
      {
        Serial.write('y');
        break;
      }
      
    }
  }
}

void praB()
{
  int oshita=0;
  while(1)
  {
    if(pre != pra)
    {
      break;
    }
    oshita = push();
    if(oshita == 1)
    {
      if(keep == 1)
      {
        Serial.write('g');
      }
      else if(keep == 2)
      {
        Serial.write('o');
      }
      else if(keep == 3)
      {
        Serial.write('w');
      }
      else if(keep == 4)
      {
        Serial.write('G');
      }
      delay(10);
      oshita = push();
      
      if(oshita == 0)
      {
        Serial.write('y');
        break;
      }
      
    }
  }
}

void praCC()
{
  int oshita=0;
  while(1)
  {
    if(pre != pra)
    {
      break;
    }
    oshita = push();
    if(oshita == 1)
    {
      if(keep == 1)
      {
        Serial.write('h');
      }
      else if(keep == 2)
      {
        Serial.write('p');
      }
      else if(keep == 3)
      {
        Serial.write('x');
      }
      else if(keep == 4)
      {
        Serial.write('H');
      }
      delay(10);
      oshita = push();
      
      if(oshita == 0)
      {
        Serial.write('y');
        break;
      }
      
    }
  }
}

void bunbunbun()
{
  pre = pra;
  praG();
  praF();
  praE();
  
  praD();
  praE();
  praF();
  praD();
  praC();
  
  praE();
  praF();
  praG();
  praE();
  
  praD();
  praE();
  praF();
  praD();
  
  praE();
  praF();
  praG();
  praE();
  
  praD();
  praE();
  praF();
  praD();

  praG();
  praF();
  praE();

  praD();
  praE();
  praF();
  praD();
  praC();
}

void flog_song()
{
  pre = pra;
  praC();
  praD();
  praE();
  praF();
  praE();
  praD();
  praC();

  praE();
  praF();
  praG();
  praA();
  praG();
  praF();
  praE();

  praC();
  praC();
  praC();
  praC();

  praC();
  praC();
  praD();
  praD();
  praE();
  praE();
  praF();
  praF();
  praE();
  praD();
  praC();
}

void shooting_star()
{
  pre = pra;
  praC();
  praC();
  praG();
  praG();
  praA();
  praA();
  praG();

  praF();
  praF();
  praE();
  praE();
  praD();
  praD();
  praC();

  praG();
  praG();
  praF();
  praF();
  praE();
  praE();
  praD();

  praG();
  praG();
  praF();
  praF();
  praE();
  praE();
  praD();

  praC();
  praC();
  praG();
  praG();
  praA();
  praA();
  praG();

  praF();
  praF();
  praE();
  praE();
  praD();
  praD();
  praC();
}
