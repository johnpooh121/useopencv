#include <jni.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "vo_features.h"

using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{

    string line;
    int i = 0;
    ifstream myfile ("/home/avisingh/Datasets/KITTI_VO/00.txt");
    double x =0, y=0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open())
    {
        while (( getline (myfile,line) ) && (i<=frame_id))
        {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j=0; j<12; j++)  {
                in >> z ;
                if (j==7) y=z;
                if (j==3)  x=z;
            }

            i++;
        }
        myfile.close();
    }

    else {
        cout << "Unable to open file";
        return 0;
    }

    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}



extern "C"
JNIEXPORT void JNICALL



Java_com_example_useopencv_MainActivity_ConvertRGBtoGray(JNIEnv *env, jobject thiz,
                                                         jlong mat_addr_input,
                                                         jlong mat_addr_result) {
    // TODO: implement ConvertRGBtoGray()
    Mat &matInput = *(Mat *)mat_addr_input;
    Mat &matResult = *(Mat *)mat_addr_result;
    cvtColor(matInput, matResult, COLOR_RGBA2GRAY);
}
extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_useopencv_MainActivity_returnarray(JNIEnv *env, jobject thiz,jint x,jint y,jint z) {
    int arr[3]={x,y,z};
    jintArray result = env->NewIntArray(10);
    env->SetIntArrayRegion( result, 0, 10, arr);
    return result;
}

extern "C"
JNIEXPORT jdoubleArray JNICALL
Java_com_example_useopencv_MainActivity_returnxyz(JNIEnv *env, jobject thiz,jlong mat_addr_input1,jlong mat_addr_input2,jlong mat_addr_input3) {
    Mat &matInput1 = *(Mat *)mat_addr_input1;
    Mat &matInput2 = *(Mat *)mat_addr_input2;
    Mat &matInput3 = *(Mat *)mat_addr_input3;
    Mat img_1, img_2;
    Mat R_f, t_f; //the final rotation and tranlation vectors containing the
    double scale = 1.00;
    ofstream myfile;
    myfile.open ("results1_1.txt");


    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    //read the first two frames from the dataset
    Mat img_1_c = matInput1;
    Mat img_2_c = matInput2;

    if ( !img_1_c.data || !img_2_c.data ) {
        std::cout<< " --(!) Error reading images " << std::endl;
    }

    // we work with grayscale images
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    // feature detection, tracking
    vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    featureDetection(img_1, points1);        //detect features in img_1
    vector<uchar> status;
    featureTracking(img_1,img_2,points1,points2, status); //track those features to img_2

    //TODO: add a fucntion to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);
    //recovering the pose and the essential matrix
    Mat E, R, t, mask;
    E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, focal, pp, mask);

    Mat prevImage = img_2;
    Mat currImage;
    vector<Point2f> prevFeatures = points2;
    vector<Point2f> currFeatures;

    char filename[100];

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();


    //for numframe =1 ; numframe<Maxframe;numframe++
    int numFrame =2;



    Mat currImage_c = matInput3;
    cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
    status.clear();
    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

    E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

    Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);


    for(int i=0;i<prevFeatures.size();i++)	{   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
        prevPts.at<double>(0,i) = prevFeatures.at(i).x;
        prevPts.at<double>(1,i) = prevFeatures.at(i).y;

        currPts.at<double>(0,i) = currFeatures.at(i).x;
        currPts.at<double>(1,i) = currFeatures.at(i).y;
    }

    scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

    //cout << "Scale is " << scale << endl;

    if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

        t_f = t_f + scale*(R_f*t);
        R_f = R*R_f;

    }

    else {
        //cout << "scale below 0.1, or incorrect translation" << endl;
    }

    // lines for printing results
    // myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

    // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
    if (prevFeatures.size() < MIN_NUM_FEAT)	{
        //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
        //cout << "trigerring redection" << endl;
        featureDetection(prevImage, prevFeatures);
        featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);

    }

    prevImage = currImage.clone();
    prevFeatures = currFeatures;

    int x = int(t_f.at<double>(0)) + 300;
    int y = int(t_f.at<double>(2)) + 100;

    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));

    double arr[3]={t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2)};
    jdoubleArray result = env->NewDoubleArray(3);
    env->SetDoubleArrayRegion( result, 0.0, 3, arr);
    return result;
}