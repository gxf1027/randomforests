# HOW TO USE:
# For training: 
# $ make
# For testing:
# $ make -e runtype=test
CXX:= g++ -std=c++11
CFLAGS:= -fpermissive -O2
src:= src/*.cpp src/tinyxml2/*.cpp
mainfunc:= demo/rf_train.cpp
target:= rf_train 
runtype:= train
ifeq ($(runtype),test)
	target:= rf_test
        mainfunc:= demo/rf_test.cpp
endif
$(target):$(src)
	$(CXX) $(CFLAGS) $^ $(mainfunc) -o $@ 
