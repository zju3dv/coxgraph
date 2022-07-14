/*
 Copyright (C) 2011  The Autonomous Systems Lab, ETH Zurich,
 Stefan Leutenegger, Simon Lynen and Margarita Chli.

 Copyright (C) 2013  The Autonomous Systems Lab, ETH Zurich,
 Stefan Leutenegger and Simon Lynen.

 BRISK - Binary Robust Invariant Scalable Keypoints
 Reference implementation of
 [1] Stefan Leutenegger,Margarita Chli and Roland Siegwart, BRISK:
 Binary Robust Invariant Scalable Keypoints, in Proceedings of
 the IEEE International Conference on Computer Vision (ICCV2011).

 This file is part of BRISK.

 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of the <organization> nor the
       names of its contributors may be used to endorse or promote products
       derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <brisk/brisk.h>
#include <brisk/brute-force-matcher.h>
#include <brisk/internal/brisk-layer.h>

#include <image_transport/image_transport.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>

using namespace std;

const int NUM_REF_IMG = 1;

///DEBUG
agast::Mat outimg;
std::vector<std::vector<cv::DMatch> > matchesdbg1;
std::vector<std::vector<cv::DMatch> > matchesdbg2;

// image with descriptors
class DescribedImage {
 public:
  agast::Mat image;
  std::vector<agast::KeyPoint> keypoints;
  agast::Mat descriptors;
  // provide a deep copy:
  DescribedImage() {
  }
  DescribedImage(const DescribedImage& dI) {
    this->image = dI.image.clone();
    this->keypoints = dI.keypoints;
    this->descriptors = dI.descriptors.clone();
  }
  DescribedImage& operator =(const DescribedImage& dI) {
    this->image = dI.image.clone();
    this->keypoints = dI.keypoints;
    this->descriptors = dI.descriptors.clone();
    return *this;
  }
};

// to be used by both threads
struct SharedData {

  DescribedImage image1;
  DescribedImage image2;

  std::vector<DescribedImage> referenceImage;
  unsigned int numRefImages;

  std::vector<std::vector<std::vector<cv::DMatch> > > matches1;
  std::vector<std::vector<std::vector<cv::DMatch> > > matches2;
  float matchingThreshold;

  int frameID;
  int frameIDuse;
  bool hasNewData;
  bool hasNewImage;
  boost::mutex mutex;

  // timing
  float timeleft;

  // display images
  agast::Mat imgRGB;  // the display matrix
  agast::Mat imgRGBmain;  // the main section of the display
  std::vector<agast::Mat> imgRGBref;  // reference images section of display
  std::vector<agast::Mat> imgRGBref2;  // reference images - not painted into
};

SharedData sharedData;
bool drawingImage;

// this is the actual ROS Node
class ImageSubscriber {
 public:
  ImageSubscriber(ros::NodeHandle &n, cv::Ptr<cv::FeatureDetector> detector,
                  cv::Ptr<cv::DescriptorExtractor> descriptorExtractor,
                  cv::Ptr<cv::DescriptorMatcher> descriptorMatcher,
                  float matchingThreshold)
      : n_(n),
        it_(n_) {
    image_sub_ = it_.subscribe("cam0/image_raw", 1,
                               &ImageSubscriber::imageCallback, this);

    // also init shared data:
    sharedData.frameID = 0;
    sharedData.hasNewData = false;
    sharedData.hasNewImage = false;
    sharedData.matchingThreshold = matchingThreshold;

    // get the detector, extractor and matcher:
    detector_ = detector;
    descriptorExtractor_ = descriptorExtractor;
    descriptorMatcher_ = descriptorMatcher;

    // time
    tNew_.tv_sec = 0;
    tNew_.tv_usec = 0;
    dT_ = 0.0f;
  }
  ~ImageSubscriber() {
  }
  void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    /*if(drawingImage)
     std::cout<<"!";*/

    // measure time
    timeval tEnd;
    tOld_ = tNew_;
    gettimeofday(&tNew_, 0);
    float dT = float(tNew_.tv_sec - tOld_.tv_sec)
        + float(tNew_.tv_usec - tOld_.tv_usec) / 1000000.0f;
    if (tOld_.tv_sec == 0 && tOld_.tv_usec == 0) {
      dT = 0.1;
    }
    if (dT == 0.0f)
      dT_ = dT;
    else
      dT_ = 0.9f * dT_ + 0.1f * dT;

    //std::cout<<":";cout.flush();
    // create the image from the message
    agast::Mat img(msg->height, msg->width, CV_8U,
                const_cast<uint8_t*>(&(msg->data[0])));

    // current image buffer
    DescribedImage* currentImage;
//		std::vector<std::vector<std::vector<cv::DMatch> > > * currentMatches;
    //std::vector<std::vector<cv::DMatch> > * pmatch;
    if (sharedData.frameID % 2 == 0) {
      currentImage = &sharedData.image1;
//			currentMatches=&sharedData.matches1;
      //pmatch=&matchesdbg1;
    } else {
      currentImage = &sharedData.image2;
//			currentMatches=&sharedData.matches2;
      //pmatch=&matchesdbg2;
    }

    if (sharedData.frameID - sharedData.frameIDuse > 1) {
      ROS_WARN("Framerate overrun");
      return;
    }

    // copy image
    boost::mutex::scoped_lock l(sharedData.mutex);
    currentImage->image = img.clone();

    gettimeofday(&tEnd, 0);
    sharedData.timeleft = dT_ - float(tEnd.tv_sec - tNew_.tv_sec)
        - float(tEnd.tv_usec - tNew_.tv_usec) / 1000000.0f;

    //if(sharedData.timeleft<0.0f) std::cout<<"!"<<std::endl;

    //std::cout<<sharedData.timeleft<<std::endl;
    //cv::imshow("test",sharedData.image);
    //char key=cv::waitKey();

    // flag that there is new data available
    sharedData.frameID++;
    sharedData.hasNewImage = true;
  }
 protected:
  ros::NodeHandle n_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;

  // frame rate and overrun detection
  timeval tOld_, tNew_;
  float dT_;  // 1/framerate

  // feature related stuff
  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorExtractor> descriptorExtractor_;
  cv::Ptr<cv::DescriptorMatcher> descriptorMatcher_;
};

// visualizer that will run in another thread
class BriskVisualizer {
 public:
  BriskVisualizer(unsigned int numRefImages = NUM_REF_IMG) {
    sharedData.numRefImages = numRefImages;
    stoprequested_ = false;
    if (numRefImages > 2)
      ros::Exception("More than two reference images "
                     "are not supported. "
                     "You wouldn't see anything anymore, right?");
    if (numRefImages < 1)
      ros::Exception("You want zero reference images. "
                     "This doesn't make sense for a matcher demo...");
  }
  // Create the thread and start work
  void start() {
    assert(!thread_);
    thread_ = boost::shared_ptr < boost::thread
        > (new boost::thread(boost::bind(&BriskVisualizer::visualize, this)));
  }

  void stop()  // Note 1
  {
    assert(thread_);
    stoprequested_ = true;
    thread_->join();
  }
  bool isrunning() {
    return !stoprequested_;
  }

 private:

  void visualize() {
    std::vector < agast::Mat > tmp;
    char key = 0;
    long int frameCounter = 0;
    cv::namedWindow("BRISK Demo");

    // wait for the first image
    while (true) {
      boost::mutex::scoped_try_lock l(sharedData.mutex);
      if (l.owns_lock()) {
        if (sharedData.hasNewData) {
          break;
        }
      }
    }
    // make the display matrix have the right size:
    agast::Mat* image;
    DescribedImage* descImage;
    if (sharedData.frameID % 2 == 0) {
      image = &sharedData.image2.image;
      descImage = &sharedData.image2;
    } else {
      image = &sharedData.image1.image;
      descImage = &sharedData.image1;
    }

    cv::cvtColor((*image), sharedData.imgRGBmain, CV_GRAY2RGB);
    cv::imshow("BRISK Demo", sharedData.imgRGB);

    int captureIDremember = -1;
    bool tryCapture = false;
    char lastkey = 0;
    while (!stoprequested_ && !ros::isShuttingDown()) {
      // consider timing
      timeval start, end;
      gettimeofday(&start, 0);

      // wait for the new image, keypoints, and matches
      while (true) {
        //std::cout<<".";
        // handle escape button
        if (key == 27) {
          stoprequested_ = true;
          ros::shutdown();
          return;
        }
        // handle capture frame
        if ((lastkey != key) && (key == '\n' || key == 'a' || key == '\r')) {
          if (sharedData.frameID != captureIDremember) {
            tryCapture = true;
          } else
            tryCapture = false;
        } else {
          tryCapture = false;
        }

        // try capturing if required
        boost::mutex::scoped_try_lock l(sharedData.mutex);
        if (l.owns_lock()) {
          if (tryCapture
              && (sharedData.referenceImage.size() < sharedData.numRefImages)) {

            if (sharedData.frameID % 2 == 0) {
              image = &sharedData.image2.image;
              descImage = &sharedData.image2;
            } else {
              image = &sharedData.image1.image;
              descImage = &sharedData.image1;
            }
            // copy
            //std::cout<<"capture"<<std::endl;
            // save into shared data
            sharedData.referenceImage.push_back(*descImage);
            // adapt display
            tmp.push_back(agast::Mat(image->rows / 2, image->cols / 2, CV_8U));
            if (sharedData.numRefImages == 2)
              brisk::Halfsample8(*image, tmp.back());
            else
              tmp.push_back(*image);
            //cv::imshow("BRISK Demo",tmp);
            //cv::waitKey();
            cv::cvtColor(
                tmp.back(),
                sharedData.imgRGBref2[sharedData.referenceImage.size() - 1],
                CV_GRAY2RGB);

            // set up the correct size for the matches
            sharedData.matches1.resize(sharedData.referenceImage.size());
            sharedData.matches2.resize(sharedData.referenceImage.size());

            // remember that we already captured the frame
            captureIDremember = sharedData.frameID;
            //tryCapture=false;
          }

          if (sharedData.hasNewData) {
            sharedData.hasNewData = false;
            frameCounter++;
            break;

          }
        }
        lastkey = key;
        key = cv::waitKey(2);
      }

      //agast::Mat t1,t2;
      //cv::cvtColor(p_descImage->image, t1, CV_GRAY2RGB);
      //cv::cvtColor(sharedData.referenceImage[0].image, t2, CV_GRAY2RGB);
      /*cv::drawMatches(
       t2,
       sharedData.referenceImage[0].keypoints,
       t1,
       p_descImage->keypoints,
       pmatch//p_matches->at(0),
       outimg);*/

      /*cv::namedWindow("Matches");
       if(outimg.cols!=0)
       cv::imshow("Matches", outimg);*/
      //cv::imshow("BRISK Demo",imgRGB);
      //cv::waitKey();

      cv::imshow("BRISK Demo", sharedData.imgRGB);

      // done. calculate how much time is left
      gettimeofday(&end, 0);
      sharedData.timeleft -= float(end.tv_sec - start.tv_sec)
          + float(end.tv_usec - start.tv_usec) / 1000000.0f;

      //if(sharedData.timeleft<0.0f) std::cout<<"!!"<<std::endl;
      //std::cout<<sharedData.timeleft<<std::endl;

      //drawingImage=false;

      key = cv::waitKey(std::max(int(sharedData.timeleft * 1000.0f) - 1, 2));
    }
  }

  boost::shared_ptr<boost::thread> thread_;
  bool stoprequested_;
};

void help(char** argv) {
  std::cout << "This command line tool lets you evaluate different keypoint "
      << "detectors, descriptors and matchers." << std::endl << "usage:"
      << std::endl << argv[0]
      << " <detector> <descriptor> <matcher> <threshold> <noRefFrame> [videoFname]"
      << std::endl << "    "
      << "detector:   Feature detector, e.g. FAST, or BRISK. You can add the "
      << std::endl << "    "
      << "            threshold, e.g. BRISK80 or SURF2000" << std::endl
      << "    "
      << "descriptor: Feature descriptor, e.g. SURF, BRIEF, BRISK or U-BRISK."
      << std::endl << "    "
      << "matcher:    Can be FlannBased, BruteForce, BruteForce-L1,"
      << std::endl << "    "
      << "            BruteForce-Hamming, BruteForce-HammingLUT, " << std::endl
      << "    " << "            or BruteForce-HammingSse (fastest for Hamming)."
      << std::endl << "    " << "threshold:  Matching threshold." << std::endl
      << "    " << "noRefFrame: Number of reference frames, 1 or 2."
      << std::endl << "    " << "videoFname: Optional raw video output file."
      << std::endl;
}

int main(int argc, char ** argv) {

  // process command line args
  if (argc != 6 && argc != 7) {
    help(argv);
    return 1;
  }

  std::string videoFname;
  bool videoOut = false;

  if (argc == 7) {
    videoFname = argv[6];
    videoOut = true;
  } else
    videoFname = "";

  // run FAST in first image
  std::vector<agast::KeyPoint> keypoints, keypoints2;
  int threshold;

  // create the detector:
  cv::Ptr < cv::FeatureDetector > detector;
  if (strncmp("FAST", argv[1], 4) == 0) {
    threshold = atoi(argv[1] + 4);
    if (threshold == 0)
      threshold = 30;
    detector = cv::FastFeatureDetector::create(threshold, true);
  } else if (strncmp("BRISK_old", argv[1], 9) == 0) {
    threshold = atoi(argv[1] + 9);
    if (threshold == 0)
      threshold = 30;
    detector = cv::BRISK::create(threshold, 4);
  } else if (strncmp("BRISK", argv[1], 5) == 0) {
    threshold = atoi(argv[1] + 5);
    if (threshold == 0)
      threshold = 30;
    detector =
        new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
            4, threshold, 20);
  } else if (strncmp("SURF", argv[1], 4) == 0) {
    threshold = atoi(argv[1] + 4);
    if (threshold == 0)
      threshold = 400;
    detector = cv::xfeatures2d::SurfFeatureDetector::create(threshold);
  } else if (strncmp("SIFT", argv[1], 4) == 0) {
    float edgeThreshold = atof(argv[1] + 4);
    detector =
        cv::xfeatures2d::SiftFeatureDetector::create(0, 3, 0.04, edgeThreshold);
    //I=1; // save time, this is so slow anyways...
  }
  if (detector.empty()) {
    std::cout << "Detector " << argv[1] << " not recognized. Check spelling!"
        << std::endl;
    return 2;
  }

  // now the extractor:
  cv::Ptr < cv::DescriptorExtractor > descriptorExtractor;
  if (std::string(argv[2]) == "BRISK") {
    descriptorExtractor = new cv::BriskDescriptorExtractor(true, true);
  } else if (std::string(argv[2]) == "U-BRISK") {
    descriptorExtractor = new cv::BriskDescriptorExtractor(true, false);
  } else if (std::string(argv[2]) == "SURF") {
    descriptorExtractor = cv::xfeatures2d::SurfDescriptorExtractor::create();
  } else if (std::string(argv[2]) == "SIFT") {
    descriptorExtractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
  } else if (std::string(argv[2]) == "BRIEF") {
    descriptorExtractor = cv::xfeatures2d::BriefDescriptorExtractor::create(64);
  }
  if (descriptorExtractor.empty()) {
    std::cout << "Descriptor " << argv[2] << " not recognized. Check spelling!"
        << std::endl;
    return 3;
  }

  // get the descriptors

  std::vector < cv::DMatch > indices;

  // match descriptors:
  cv::Ptr < cv::DescriptorMatcher > descriptorMatcher;
  if (std::string(argv[3]) == "BruteForce-HammingSse") {
    descriptorMatcher = new brisk::BruteForceMatcherSse;
  } else {
    descriptorMatcher = new cv::BFMatcher(cv::L2<float>::normType);
  }
  if (descriptorMatcher.empty()) {
    std::cout << "Matcher " << argv[3] << " not recognized. Check spelling!"
        << std::endl;
    return 5;
  }

  // matching threshold
  float matchingThreshold = atof(argv[4]);

  // set up the node and the visualization
  ros::init(argc, argv, "demo");
  ros::NodeHandle nh;
  ImageSubscriber imageSubscriber(nh, detector, descriptorExtractor,
                                  descriptorMatcher, matchingThreshold);
  BriskVisualizer visualizer(atoi(argv[5]));

  // start the application
  visualizer.start();
  //cv::waitKey(1000);
  //ros::MultiThreadedSpinner spinner(2);
  //spinner.spin();
  //ros::spin();

  // do we have to write a video?
  //video writer
  cv::VideoWriter writer;
  bool writerIsOpen = false;
  bool imagesInitialized = false;

  while (visualizer.isrunning() && !ros::isShuttingDown()) {
    // main event loop:
    // wait for acquisition
    sharedData.hasNewImage = false;
    while (visualizer.isrunning()) {
      //boost::mutex::scoped_try_lock l(sharedData.mutex);
      //if(l.owns_lock()){
      if (sharedData.hasNewImage) {
        break;
      }
      //}
      //l.unlock();
      ros::spinOnce();
      ros::Duration(0.001).sleep();
    }
    if (!visualizer.isrunning())
      break;
    //std::cout<<":";
    sharedData.hasNewData = false;

    //while(!sharedData.hasNewImage);
    sharedData.frameIDuse = sharedData.frameID;

    // care about time
    timeval start, end;
    gettimeofday(&start, 0);

    // current image buffer
    DescribedImage* currentImage;
    std::vector < std::vector<std::vector<cv::DMatch> > > *currentMatches;
    //std::vector<std::vector<cv::DMatch> > * pmatch;
    if (sharedData.frameID % 2 == 1) {
      currentImage = &sharedData.image1;
      currentMatches = &sharedData.matches1;
      //pmatch=&matchesdbg1;
    } else {
      currentImage = &sharedData.image2;
      currentMatches = &sharedData.matches2;
      //pmatch=&matchesdbg2;
    }

    if (!imagesInitialized) {
      // init shared images
      sharedData.imgRGB = agast::Mat::zeros(
          currentImage->image.rows
              + currentImage->image.rows / sharedData.numRefImages,
          currentImage->image.cols, CV_8UC3);

      // display it:
      sharedData.imgRGBmain = sharedData.imgRGB.rowRange(
          currentImage->image.rows / sharedData.numRefImages,
          currentImage->image.rows
              + currentImage->image.rows / sharedData.numRefImages);
      // create the other submatrices as well
      for (unsigned int i = 0; i < sharedData.numRefImages; i++) {
        sharedData.imgRGBref.push_back(
            sharedData.imgRGB(
                cv::Range(0,
                          currentImage->image.rows / sharedData.numRefImages),
                cv::Range(
                    i * currentImage->image.cols / sharedData.numRefImages,
                    (i + 1) * currentImage->image.cols
                        / sharedData.numRefImages)));
        sharedData.imgRGBref2.push_back(
            agast::Mat::zeros(currentImage->image.rows / sharedData.numRefImages,
                           currentImage->image.cols / sharedData.numRefImages,
                           CV_8UC3));
      }
      imagesInitialized = true;
    }

    // video writing?
    if (videoOut && !writerIsOpen) {
      writerIsOpen = writer.open(
          videoFname, CV_FOURCC('I', 'Y', 'U', 'V'), 25.0,
          cv::Size(sharedData.imgRGB.cols, sharedData.imgRGB.rows));
    }

    // extract features
    boost::mutex::scoped_lock l(sharedData.mutex);
    //static_cast<cv::Ptr<brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> > >
    (detector)->detect(currentImage->image, currentImage->keypoints);
    // get the descriptors
    descriptorExtractor->compute(currentImage->image, currentImage->keypoints,
                                 currentImage->descriptors);

    // do the matches if reference images exist:
    for (unsigned int i = 0; i < sharedData.referenceImage.size(); i++) {
      currentMatches->at(i).resize(0);  // incredible that the matcher does not do this on it's own...
      descriptorMatcher->radiusMatch(sharedData.referenceImage[i].descriptors,
                                     currentImage->descriptors,
                                     currentMatches->at(i)/**pmatch*/,
                                     sharedData.matchingThreshold);
    }
    l.unlock();

    //convert to a color image
    cv::cvtColor(currentImage->image, sharedData.imgRGBmain, CV_GRAY2RGB);

    // display keypoints
    const unsigned int ksize = currentImage->keypoints.size();
    // before we have reference images, let's just draw blue circles
    if (sharedData.referenceImage.empty()) {
      for (unsigned int i = 0; i < ksize; i++) {
        cv::circle(sharedData.imgRGBmain, currentImage->keypoints[i].pt,
                   currentImage->keypoints[i].size / 4.0, cv::Scalar(255, 0, 0),
                   2);
      }
    } else {
      // draw keypoints in new image:
      const unsigned int ksize2 = currentImage->keypoints.size();
      for (unsigned int i = 0; i < ksize2; i++) {
        cv::circle(sharedData.imgRGBmain, currentImage->keypoints[i].pt,
                   currentImage->keypoints[i].size / 4.0, cv::Scalar(0, 0, 255),
                   2);
      }

      // clean
      const float scaling = 1.0f / float(sharedData.numRefImages);
      for (unsigned int r = 0; r < sharedData.numRefImages; r++) {
        // copy the unpainted reference images
        sharedData.imgRGBref2[r].copyTo(sharedData.imgRGBref[r]);
      }
      // we want to draw the matches as well...
      const unsigned int rSize = sharedData.referenceImage.size();
      for (unsigned int r = 0; r < rSize; r++) {
        // draw keypoints in reference image:
        const unsigned int ksize =
            sharedData.referenceImage[r].keypoints.size();
        for (unsigned int i = 0; i < ksize; i++) {
          cv::circle(
              sharedData.imgRGBref[r],
              sharedData.referenceImage[r].keypoints[i].pt * scaling,
              sharedData.referenceImage[r].keypoints[i].size / 4.0 * scaling,
              cv::Scalar(0, 0, 255), 2.0f / float(sharedData.numRefImages));
        }

        // draw match points:
        const unsigned int msize = currentMatches->at(r).size();
        //const int x_offset=currentImage->image.cols/sharedData.numRefImages;
        //const int y_offset=currentImage->image.rows/sharedData.numRefImages;
        for (unsigned int m = 0; m < msize; m++) {
          if ((currentMatches->at(r))[m].size() == 0)
            continue;
          const cv::DMatch& match = currentMatches->at(r)[m][0];
          const int ind1 = match.trainIdx;
          const int ind2 = match.queryIdx;
          cv::circle(sharedData.imgRGBmain, currentImage->keypoints[ind1].pt,
                     currentImage->keypoints[ind1].size / 4.0,
                     cv::Scalar(0, 255, 0), 2);
          cv::circle(
              sharedData.imgRGBref.at(r),
              sharedData.referenceImage.at(r).keypoints.at(ind2).pt * scaling,
              sharedData.referenceImage.at(r).keypoints.at(ind2).size / 4.0
                  * scaling,
              cv::Scalar(0, 255, 0), 2.0f / float(sharedData.numRefImages));
          /*cv::line(sharedData.imgRGB,currentImage->keypoints[ind1].pt+cv::Point2f(0,y_offset),
           sharedData.referenceImage[r].keypoints[ind2].pt*scaling+cv::Point2f(r*x_offset,0),
           cv::Scalar(0,255,0),2);*/
        }
      }
      // draw match lines:
      for (unsigned int r = 0; r < rSize; r++) {
        const unsigned int msize = currentMatches->at(r).size();
        const int x_offset = currentImage->image.cols / sharedData.numRefImages;
        const int y_offset = currentImage->image.rows / sharedData.numRefImages;
        for (unsigned int m = 0; m < msize; m++) {
          if ((currentMatches->at(r))[m].size() == 0)
            continue;
          const cv::DMatch& match = currentMatches->at(r)[m][0];
          const int ind1 = match.trainIdx;
          const int ind2 = match.queryIdx;
          cv::line(
              sharedData.imgRGB,
              currentImage->keypoints[ind1].pt + cv::Point2f(0, y_offset),
              sharedData.referenceImage[r].keypoints[ind2].pt * scaling
                  + cv::Point2f(r * x_offset, 0),
              cv::Scalar(0, 255, 0), 2);
        }
      }
    }

    // video writing?
    if (videoOut && writerIsOpen)
      writer.write(sharedData.imgRGB);

    gettimeofday(&end, 0);
    sharedData.timeleft -= float(end.tv_sec - start.tv_sec)
        + float(end.tv_usec - start.tv_usec) / 1000000.0f;
    if (sharedData.timeleft < 0.0f)
      ROS_WARN("Framerate overrun!");
    else
      sharedData.hasNewData = true;
  }
  visualizer.stop();

  /*/ the image:
   agast::Mat imgGray;
   agast::Mat imgGrayKey;
   agast::Mat imgRGB;

   imgGray.create(480,752,CV_8U);

   //video writer
   cv::VideoWriter writer;
   if (videoOut)
   writer.open(videoFname, CV_FOURCC('I', 'Y', 'U', 'V'),25.0,cv::Size(752,965));


   ros::init(argc, argv, "demo");
   ros::NodeHandle nh;
   ImageSubscriber imageSubscriber(nh);
   cv::namedWindow("demo");
   cvStartWindowThread();
   ros::AsyncSpinner spinner(0);
   spinner.start();
   agast::Mat descriptors;
   char k=0;
   for(int i=0; i<NUM_REF_IMG; i++){
   k=0;
   while(k!='n'){
   imageSubscriber.getNextImg(imgGray,k);
   detector->detect(imgGray,keypoints);
   cv::cvtColor(imgGray, imgRGB, CV_GRAY2RGB);
   const unsigned int ksize=keypoints.size();
   for(unsigned int i=0; i<ksize; i++){
   cv::circle(imgRGB,keypoints[i].pt,keypoints[i].size/4.0,cv::Scalar(255,0,0),2);
   }
   //imgGray.resize(965);
   cv::imshow("demo",imgRGB);
   if(videoOut){
   const int max=imageSubscriber.ctr+1;
   imgRGB.resize(965);
   for(int f=0; f<max; f++){
   writer.write(imgRGB);
   }
   }
   }
   descriptorExtractor->compute(imgGray,keypoints,descriptors);
   imgGrayKey=imgGray.clone();
   imgGrayKey.resize(480);
   }
   agast::Mat someZeros=agast::Mat::zeros(5,imgGray.cols,CV_8U);
   const unsigned int ksize=keypoints.size();
   while(k!=27){
   agast::Mat descriptors2;
   //if(imageSubscriber.ctr>0) std::cout<<imageSubscriber.ctr<<" frame(s) lost."<<std::endl;
   imageSubscriber.getNextImg(imgGray,k);

   std::vector<std::vector<cv::DMatch> > matches;
   // detect, extract and match:
   detector->detect(imgGray,keypoints2);
   descriptorExtractor->compute(imgGray,keypoints2,descriptors2);
   descriptorMatcher->radiusMatch(descriptors2,descriptors,matches,matchingThreshold);

   imgGray.push_back(someZeros); // some black lines
   imgGray.push_back(imgGrayKey);
   cv::cvtColor(imgGray, imgRGB, CV_GRAY2RGB);

   // draw keypoints in reference image:
   for(unsigned int i=0; i<ksize; i++){
   cv::circle(imgRGB,keypoints[i].pt+cv::Point2f(0,485),keypoints[i].size/4.0,cv::Scalar(0,0,255),2);
   }
   // draw keypoints in new image:
   const unsigned int ksize2=keypoints2.size();
   for(unsigned int i=0; i<ksize2; i++){
   cv::circle(imgRGB,keypoints2[i].pt,keypoints2[i].size/4.0,cv::Scalar(0,0,255),2);
   }

   // draw match lines:
   const unsigned int msize=matches.size();
   for(unsigned int m=0; m<msize; m++){
   if(matches[m].size()==0) continue;
   const cv::DMatch match=matches[m][0];
   const int ind1=match.trainIdx;
   const int ind2=match.queryIdx;
   cv::circle(imgRGB,keypoints[ind1].pt+cv::Point2f(0,485),keypoints[ind1].size/4.0,cv::Scalar(0,255,0),2);
   cv::circle(imgRGB,keypoints2[ind2].pt,keypoints2[ind2].size/4.0,cv::Scalar(0,255,0),2);
   cv::line(imgRGB,keypoints[ind1].pt+cv::Point2f(0,485),keypoints2[ind2].pt,cv::Scalar(0,255,0),2);
   }

   cv::imshow("demo",imgRGB);
   if(videoOut){
   const int max=imageSubscriber.ctr+1;
   for(int f=0; f<max; f++){
   writer.write(imgRGB);
   }
   }
   }
   spinner.stop();
   cvDestroyWindow("demo");*/

  return 0;
}
