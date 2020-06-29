//
// Created by sxy on 2020/6/26.
//

#ifndef SRC_POINTCLOUDMAPPING_H
#define SRC_POINTCLOUDMAPPING_H

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "Pointcloude.h"
//#include "System.h"
#include "KeyFrame.h"
#include<thread>
using namespace std;
namespace ORB_SLAM2
{
    class PointCloudMapping
    {
    public:
        typedef pcl::PointXYZRGBA PointT;
        typedef pcl::PointCloud<PointT> PointCloud;

        PointCloudMapping(double resolution_,double meank_,double thresh_,int mbpointcloud_);
        void save();
        //插入一个关键帧,会更新一次地图
        void insertKeyFrame(KeyFrame* kf,cv::Mat& color,cv::Mat& depth,int idk,vector<KeyFrame*> vpKFs);
        void shutdown();
        void viewer();
        int loopcount=0;
        vector<KeyFrame*> currentvpKFs;
        bool cloudbusy;
        bool loopbusy;
        void updatecloud();
        bool bStop = false;
    protected:
        PointCloud::Ptr generationPointCloud(KeyFrame* kf,cv::Mat& color,cv::Mat& depth);
        PointCloud::Ptr globalMap;
        shared_ptr<thread>  viewerThread;
        bool    shutDownFlag    =false;
        mutex   shutDownMutex;
        condition_variable  keyFrameUpdated;
        mutex               keyFrameUpdateMutex;
        vector<PointCloude>     pointcloud;
        vector<KeyFrame*>       keyframes;
        mutex                   keyframeMutex;
        uint16_t                lastKeyframeSize =0;
        double resolution = 0.04;
        double meank = 50;
        double thresh = 1;
        pcl::VoxelGrid<PointT>  voxel;
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        int mbpointcloud;
    };
}
















#endif //SRC_POINTCLOUDMAPPING_H
