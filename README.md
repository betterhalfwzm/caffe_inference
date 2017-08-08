# CAFFE INFERENCE

This module automatically dumps the input and output to <CAFFE_ROOT/out> folders.

Set up:
1. Clone : git clone -b dump_input_output https://github.com/lcskrishna/caffe.git
2. Modify the changes that are necessary for building caffe. 
3. make all.
4. Create an out directory in <CAFFE_ROOT> if not exists.

#createFileList.sh :
1. Store all the images which you intend to test into that folder and run this script.
2. It creates a text file that contains all the paths of the images.
3. Use this text file to run in the inference.

In setup use run.sh to create an executable file.
1. Change the prototxt image data file to image_list.txt
2. Run the executable and follow the instructions.
