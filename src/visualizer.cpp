/*
MIT License

Copyright (c) 2017 Chaitanya Sri Krishna Lolla

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

// source: adapted from cityscapes-dataset.org
unsigned char overlayColors[20][3] = {
	{  0,  0,  0},
	{128, 64,128}, // road
	{244, 35,232}, // sidewalk
	{ 250, 150, 70}, // building
	{102,102,156}, // wall
	{190,153,153}, // fence
	{153,153,153}, // pole
	{250,170, 30}, // traffic light
	{220,220,  0}, // traffic sign
	{107,142, 35}, // vegetation
	{152,251,152}, // terrain
	{ 70,130,180}, // sky
	{220, 20, 60}, // person
	{255,  0,  0}, // rider
	{  0,  0,142}, // car
	{  0,  0, 70}, // truck
	{  0, 60,100}, // bus
	{  0, 80,100}, // train
	{  0,  0,230}, // motorcycle
	{119, 11, 32}  // bicycle
};

void getMaxProbabilityAndOutputClass(float * prob, unsigned char * classImg, int input_dims[4], float * output_layer, float threshold )
{

    int numClasses = input_dims[1];
    int height = input_dims[2];
    int width = input_dims[3];

    // Initialize buffers
    memset(prob, 0, width * height * sizeof(float));
    memset(classImg, 0, width * height);

    for(int c = 0; c < numClasses; c++)
    {
        for(int i = 0; i < width * height; i++)
        {
            if((output_layer[i] >= threshold) && (output_layer[i] > prob[i]))
            {
                prob[i] = output_layer[i];
                classImg[i] = c + 1;
            }
        }
        output_layer += (width * height);
    }
}

void overlayInputImage(const Mat& inpImg, unsigned char * classIDBuf, Mat& outImg, cv::Size input_geometry)
{
    if((inpImg.rows != outImg.rows) || (inpImg.cols != outImg.cols))
    {
        printf("The input and overlay images must be of the same dimension\n");
        return;
    }

    Vec3b pix;
    unsigned char classId = 0;
    float alpha = 0.3;
    float oneMinusAlpha = 1.0f - alpha;

    for(int i = 0; i < inpImg.rows; i++)
    {
        for(int j = 0; j < inpImg.cols; j++)
        {
            pix = inpImg.at<Vec3b>(i, j);
            classId = classIDBuf[(i * input_geometry.width) + j];
            pix.val[0] = (overlayColors[classId][2] * alpha) + (pix.val[0] * oneMinusAlpha);
            pix.val[1] = (overlayColors[classId][1] * alpha) + (pix.val[1] * oneMinusAlpha);
            pix.val[2] = (overlayColors[classId][0] * alpha) + (pix.val[2] * oneMinusAlpha);

            pix.val[0] = (pix.val[0] > 255) ? 255 : pix.val[0];
            pix.val[1] = (pix.val[1] > 255) ? 255 : pix.val[1];
            pix.val[2] = (pix.val[2] > 255) ? 255 : pix.val[2];
            outImg.at<Vec3b>(i, j) = pix;
        }
    }

}


int main(int argc, char* argv[])
{

	if(argc < 5) {
		printf("Usage visualizer <dump_softmax bin file> n c h w threshold <input_file_path>\n");
		return -1;
	}

	std::string out_file = argv[1];
	int input_dims[4]={0};
	input_dims[0] = atoi(argv[2]);
	input_dims[1] = atoi(argv[3]);
	input_dims[2] = atoi(argv[4]);
	input_dims[3] = atoi(argv[5]);
	float threshold = atof(argv[6]);

	//Read the input softmax file.
	int total_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
	float * output_layer = new float[total_size];
	FILE * fs;
	fs = fopen(out_file.c_str(), "rb");
	if(!fs){
		std::cout << "Unable to open file :" << out_file << std::endl;
		return -1;
	}
	fread(output_layer,sizeof(float), total_size , fs);
	
	unsigned char * classIDBuf = new unsigned char[input_dims[2] * input_dims[3]];
    float * prob = new float[input_dims[2] * input_dims[3]];	
	getMaxProbabilityAndOutputClass(prob, classIDBuf, input_dims, output_layer, threshold);


	//Input Image.
	std::string input_img_path = argv[7];
	Mat input_img = imread(input_img_path);
	if(!input_img.data) std::cerr << "Unable to read input image" << std::endl;
	
	cv::Size input_geometry = cv::Size(input_dims[3], input_dims[2]);
	Mat output_img;
	output_img.create(input_geometry, CV_8UC3);
	overlayInputImage(input_img, classIDBuf, output_img, input_geometry);

	std::string opfileName = "test_out.png";
	imwrite(opfileName, output_img);

	delete output_layer;
	delete classIDBuf;
	delete prob;

	return 0;
}
