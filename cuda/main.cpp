#include <CImg.h>
#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>
<<<<<<< HEAD
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
=======
>>>>>>> 846a608ae44b9981bd2d1225a3ffa27cdbef206a

using cimg_library::CImg;
using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

// Constants
const bool displayImages = false;
<<<<<<< HEAD
const bool saveAllImages = true;
=======
const bool saveAllImages = false;
>>>>>>> 846a608ae44b9981bd2d1225a3ffa27cdbef206a
const unsigned int HISTOGRAM_SIZE = 256;
const unsigned int BAR_WIDTH = 4;
const unsigned int CONTRAST_THRESHOLD = 80;
const float filter[] = {	1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
						1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
						1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

extern void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height);
//extern void rgb2grayCuda

extern void histogram1D(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH);
//extern void histogram1DCuda 

extern void contrast1D(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD);
//extern void contrast1DCuda

extern void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter);
//extern void triangularSmoothCuda

<<<<<<< HEAD
extern void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height);

=======
>>>>>>> 846a608ae44b9981bd2d1225a3ffa27cdbef206a

int main(int argc, char *argv[]) 
{
	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( displayImages ) {
		inputImage.display("Input Image");
	}
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}

<<<<<<< HEAD
	//allocate memory on Host 
	//unsigned char *buffer = 0;
	//buffer = (unsigned char*)malloc(inputImage.width()*inputImage.height()*sizeof(unsigned char));
	
	//allocation on the Device
	/*
	if(cudaMalloc((void**)&d_input,inputImage.width()*inputImage.height()*sizeof(unsigned char)*3) != cudaSuccess){
		cout<<"error on allocate device memory"<<endl;
		exit(-1);
	};
	if(cudaMalloc((void**)&d_output,inputImage.width()*inputImage.height()*sizeof(unsigned char))!= cudaSuccess){
		printf("error on allocate device memory\n");
		exit(-1);
	};

	//memory copy from Host to Device 
	if(cudaMemcpy(d_input,inputImage.data(),inputImage.width()*inputImage.height()*sizeof(unsigned char)*3,cudaMemcpyHostToDevice)
		!= cudaSuccess){
		printf("error on memory transfer\n");
		exit(-2);
	}
	*/
	
	// Convert the input image to grayscale 
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);
	CImg< unsigned char > grayImage_gpu = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

	rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());
	rgb2grayCuda(inputImage.data(),grayImage_gpu,inputImage.width(), inputImage.height());

	//memory copy from Device to Host
	/*
	if(cudaMemcpy(grayImage_gpu,d_output,inputImage.width()*inputImage.height()*sizeof(unsigned char),cudaMemcpyDeviceToHost)
		!= cudaSuccess){
		printf("error on memory transfer\n");
		exit(-3);
	}
	*/
	//CImg< unsigned char > grayImage_gpu = CImg< unsigned char >(buffer[0])

=======
	// Convert the input image to grayscale 
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

	rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());
	//rgb2grayCuda
>>>>>>> 846a608ae44b9981bd2d1225a3ffa27cdbef206a

	if ( displayImages ) {
		grayImage.display("Grayscale Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./grayscale.bmp");
<<<<<<< HEAD
		grayImage_gpu.save("./grayImage_gpu.bmp");
	}

=======
	}
	
>>>>>>> 846a608ae44b9981bd2d1225a3ffa27cdbef206a
	// Compute 1D histogram
	CImg< unsigned char > histogramImage = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	unsigned int *histogram = new unsigned int [HISTOGRAM_SIZE];

	histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, BAR_WIDTH);
	//histogram1DCuda

	if ( displayImages ) {
		histogramImage.display("Histogram");
	}
	if ( saveAllImages ) {
		histogramImage.save("./histogram.bmp");
	}

	// Contrast enhancement
	contrast1D(grayImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, CONTRAST_THRESHOLD);
	//contrast1DCuda

	if ( displayImages ) {
		grayImage.display("Contrast Enhanced Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./contrast.bmp");
	}

	delete [] histogram;

	// Triangular smooth (convolution)
	CImg< unsigned char > smoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);

	triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter);
	//triangularSmoothCuda
	
	if ( displayImages ) {
		smoothImage.display("Smooth Image");
	}
	
	if ( saveAllImages ) {
		smoothImage.save("./smooth.bmp");
	}
<<<<<<< HEAD
	
=======
>>>>>>> 846a608ae44b9981bd2d1225a3ffa27cdbef206a

	return 0;
}

