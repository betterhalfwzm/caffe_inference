g++ -std=c++11 caffe_test.cpp `pkg-config --libs opencv` -I <CAFFE_BUILD> -I <CAFFE_BUILD>/src/  -L <CAFFE_BUILD>/lib/ -o caffe_test -lboost_system -lcaffe -lglog

