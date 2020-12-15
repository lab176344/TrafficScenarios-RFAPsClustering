
clc;clear all;close all;
load('HighDScenarioClass2.mat')

id0 = yTrain == 0;
id1 = yTrain == 1;
id2 = yTrain == 2;
id3 = yTrain == 3;
id4 = yTrain == 4;
id5 = yTrain == 5;
id6 = yTrain == 6;
id7 = yTrain == 7;
id8 = yTrain == 8;

yTrain(id0) = 7;
yTrain(id1) = 6;
yTrain(id2) = 5;
yTrain(id7) = 0;
yTrain(id6) = 1;
yTrain(id5) = 2;


idT0 = yTest == 0;
idT1 = yTest == 1;
idT2 = yTest == 2;
idT3 = yTest == 3;
idT4 = yTest == 4;
idT5 = yTest == 5;
idT6 = yTest == 6;
idT7 = yTest == 7;
idT8 = yTest == 8;

yTest(idT0) = 7;
yTest(idT1) = 6;
yTest(idT2) = 5;
yTest(idT7) = 0;
yTest(idT6) = 1;
yTest(idT5) = 2;
save('HighDScenarioClass2.mat')



