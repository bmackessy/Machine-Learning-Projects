// Brian Mackessy
// Dr. Schroeder
// Data Visualization
// HW6: Wind Visualization in the US
// 4/5/18
// 



// uwnd stores the 'u' component of the wind.
// The 'u' component is the east-west component of the wind.
// Positive values indicate eastward wind, and negative
// values indicate westward wind.  This is measured
// in meters per second.
Table uwnd;

// vwnd stores the 'v' component of the wind, which measures the
// north-south component of the wind.  Positive values indicate
// northward wind, and negative values indicate southward wind.
Table vwnd;

// An image to use for the background.  The image I provide is a
// modified version of this wikipedia image:
//https://commons.wikimedia.org/wiki/File:Equirectangular_projection_SW.jpg
// If you want to use your own image, you should take an equirectangular
// map and pick out the subset that corresponds to the range from
// 135W to 65W, and from 55N to 25N
PImage img;

int[][] randies;
int frameCounter;

void setup() {
  // If this doesn't work on your computer, you can remove the 'P3D'
  // parameter.  On many computers, having P3D should make it run faster
  size(700, 400, P3D);
  img = loadImage("background.png");
  uwnd = loadTable("uwnd.csv");
  vwnd = loadTable("vwnd.csv");
  randies = new int[2000][2];

  frameCounter = 0;
  
  for (int i=0; i<2000; i++) {
   randies[i][0] = int(random(700));
   randies[i][1] = int(random(400));
 }
  
  
  
}

void draw() {
  int frames = 20;
  
  for (int i=frameCounter; i<frameCounter+100; i++) {
    randies[frameCounter*99+i][0] = int(random(700));
    randies[frameCounter*99+i][1] = int(random(400));
  }
  
  background(255);
  image(img, 0, 0, width, height);
  drawParticles();
  recalculateParticles();
  frameCounter = (frameCounter + 1) % frames;
  delay(50);
}

void recalculateParticles() {
 for (int i = 0; i<2000; i++) {
   int x = randies[i][0];
   int y = randies[i][1];
   
   // Get the data points
   x = x * uwnd.getColumnCount() / width;
   y = y * uwnd.getRowCount() / height;
   
   float dx = readRaw(uwnd, x, y);
   float dy = -readRaw(vwnd, x, y);
   
   randies[i][0] += dx;
   randies[i][1] += dy;
 }
}


void drawParticles() {
  strokeWeight(2);
  beginShape(POINTS);
  for (int i=0; i<2000; i++) {
    vertex(randies[i][0], randies[i][1]); 
  }
  
  endShape();   
}

void drawMouseLine() {
  // Convert from pixel coordinates into coordinates
  // corresponding to the data.
  float a = mouseX * uwnd.getColumnCount() / (float)width;
  float b = mouseY * uwnd.getRowCount() / (float)height;

  // Since a positive 'v' value indicates north, we need to
  // negate it so that it works in the same coordinates as Processing
  // does.
  float dx = readInterp(uwnd, a, b) * 10;
  float dy = -readInterp(vwnd, a, b) * 10;
  line(mouseX, mouseY, mouseX + dx, mouseY + dy);
}

// Reads a bilinearly-interpolated value at the given a and b
// coordinates.  Both a and b should be in data coordinates.
float readInterp(Table tab, float a, float b) {  
  int x1 = int(a);
  int y1 = int(b);
  int x2 = x1+10;
  int y2 = y1+10;
  
  float val00 = readRaw(tab, x1, y1);
  float val01 = readRaw(tab, x1, y2);
  float val10 = readRaw(tab, x2, y1);
  float val11 = readRaw(tab, x2, y2);
  
  // f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
  // One variable for each term of equation
  float interp1 = val00 * (1-(a-int(a))) * (1-(b-int(b)));
  float interp2 = val10 * (a-int(a)) * (1-(b-int(b)));
  float interp3 = val01 * (1-(a-int(a))) * (b-int(b));
  float interp4 = val11 * (a-int(a)) * (b-int(b));
  // Sum the terms
  float interp = interp1 + interp2 + interp3 + interp4;
  
 
  return readRaw(tab, x1, y1);
}


void randoms(){
 for (int i=0; i<2000; i++) {
   randies[i][0] = int(random(700));
   randies[i][1] = int(random(400));
 }
}



// Reads a raw value 
float readRaw(Table tab, int x, int y) {
  if (x < 0) {
    x = 0;
  }
  if (x >= tab.getColumnCount()) {
    x = tab.getColumnCount() - 1;
  }
  if (y < 0) {
    y = 0;
  }
  if (y >= tab.getRowCount()) {
    y = tab.getRowCount() - 1;
  }
  return tab.getFloat(y,x);
}
