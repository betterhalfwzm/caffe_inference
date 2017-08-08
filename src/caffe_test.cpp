#include <caffe/caffe.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>

using namespace caffe;
using namespace cv;

#define MAX_FILE_NAME_LENGTH 400

//TODO:  Need to get the caffe_root as a variable.
std::string caffe_root = "";

class CaffeDump{
public:
    CaffeDump(const std::string& net_file, const std::string& weight_file, int input_dims[4]);
    ~CaffeDump();
    void dump_caffe_inference_output(const Mat& img, int input_dims[4]);

private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int_tp num_channels_;
    float threshold_;

};

// Get all available GPU devices
static void get_gpus(vector<int>* gpus) {
    int count = 0;
#ifndef CPU_ONLY
    count = Caffe::EnumerateDevices(true);
#else
    NO_GPU;
#endif
    gpus->push_back(0);
    for (int i = 0; i < count; ++i) {
        gpus->push_back(i);
    }
}

CaffeDump::CaffeDump(const std::string& net_file, const std::string& weight_file, int input_dims[4])
{
    //Set device id and mode (Set To Default CPU).
    //TODO: Need to recognize automatically and pick GPU if available.
    Caffe::set_mode(Caffe::CPU);

    // Load the network
    net_.reset(new Net<float>(net_file, TEST, Caffe::GetDefaultDevice()));
    net_->CopyTrainedLayersFrom(weight_file);

    num_channels_ = input_dims[1];
    input_geometry_ = cv::Size(input_dims[3], input_dims[2]);
}

CaffeDump::~CaffeDump() {}

void CaffeDump::dump_caffe_inference_output(const Mat& img, int input_dims[4])
{

    int total_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
    float * input_data = new float[total_size];

    // Read data from cv matrix and dump input data to the first layer in prototxt.
    int height = input_dims[2];
    int width = input_dims[3];
    cv::Mat channels[3];
    cv::Mat fImg;
    img.convertTo(fImg, CV_32FC3);
    cv::split(fImg, channels);

    memcpy(input_data, channels[0].data, width*height*sizeof(float));
    memcpy(input_data + width*height, channels[1].data, width*height*sizeof(float));
    memcpy(input_data + width*height + width*height, channels[2].data, width*height*sizeof(float));

    FILE * fInp;
    std::string input_file_path = caffe_root + "out/input.f32";
    fInp = fopen(input_file_path.c_str(),"wb");
    for(int i=0;i < total_size ; i++)
    {
        float val = input_data[i];
        fwrite(&val,sizeof(float),1,fInp);
    }
    fclose(fInp);
    delete input_data;

    //Caffe Run Forward (Inference).
    std::cout << "Inference will be executed " << std::endl;
    net_->Forward();
}

int main(int argc, char * argv[])
{
    if(argc < 2)
    {
        printf("Usage caffe_test <net.prototxt> <net.caffemodel> <inputFile_txt> <outputDirectory> <output_prefix>\n");
        return -1;
    }

    std::string  prototxt_file = argv[1];
    std::string caffemodel_file = argv[2];
    const char * input_files_path = argv[3];
    const char * output_directory = argv[4];

    int input_dims[4];
    input_dims[0] = atoi(argv[5]);
    input_dims[1] = atoi(argv[6]);
    input_dims[2] = atoi(argv[7]);
    input_dims[3] = atoi(argv[8]);
    std::string out_prefix = argv[9];

    std::cout << "Reading the given prototxt file : " << prototxt_file << std::endl;
    std::cout << "Reading the given caffemodel file: " << caffemodel_file << std::endl;

    FILE * fs;
    char * image_path = NULL;
    size_t buff_size =0;
    ssize_t read;

    fs = fopen(input_files_path, "r");
    if(!fs){
        std::cout << "Unable to open the file." << std::endl;
        return -1;
    }


    int count = 0;
    while((read = getline(&image_path,&buff_size, fs)) != -1 ){

        if( strchr(image_path,'\n') ) *strchr(image_path, '\n') = 0;
        printf("%s\n",image_path);

        //Read image in given path.
        std::string inp_image(image_path);
        Mat image;
        image = imread(inp_image);
        if(!image.data) {
            std::cout << "Unable to read image " << inp_image << std::endl;
            return -1;
        }

        //Write the input image to image_list.txt
        FILE * fImg;
        fImg = fopen("image_list.txt","wb");
        size_t len = strlen(image_path);
        fwrite(image_path, sizeof(char),len, fImg);
        fclose(fImg);

        CaffeDump dump(prototxt_file.c_str(), caffemodel_file.c_str(), input_dims);
        dump.dump_caffe_inference_output(image,input_dims);
        count++;

        //Renaming the output files.
        std::string file_path = caffe_root + "out/prob.f32";
        std::string new_filePath = caffe_root  + "out/" +  out_prefix + "_" + std::to_string(count)+ ".f32" ;
        std::string input_filePath = caffe_root + "out/input.f32";
        std::string new_inp_file = caffe_root + "out/input_"+ out_prefix + "_" + std::to_string(count) + ".f32";
        int ret = rename(file_path.c_str(), new_filePath.c_str());
        ret = rename(input_filePath.c_str(), new_inp_file.c_str());
    }
    
    return 0;

}

