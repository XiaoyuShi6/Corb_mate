//
// Created by sxy on 2020/6/26.
//

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "Pointcloude.h"
#include "Pointcloudmapping.h"
#include <KeyFrame.h>
#include "Converter.h"

namespace ORB_SLAM2
{
PointCloudMapping::PointCloudMapping(double resolution_, double meank_, double thresh_,int mbpointcloud_) {
    resolution=resolution_;
    meank=meank_;
    thresh=thresh_;
    mbpointcloud=mbpointcloud_;
    statistical_filter.setMeanK(meank);
    statistical_filter.setStddevMulThresh(thresh);
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );
    if(mbpointcloud==1)
        viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown() {
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}
void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth,int idk,vector<KeyFrame*> vpKFs)
{
    cout<<"receive a keyframe, id = "<<idk<<" 第"<<kf->mnId<<"个"<<endl;
    //cout<<"vpKFs数量"<<vpKFs.size()<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    currentvpKFs = vpKFs;
    //colorImgs.push_back( color.clone() );
    //depthImgs.push_back( depth.clone() );
    PointCloude pointcloude;
    pointcloude.pcID = idk;
    pointcloude.T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    pointcloude.pcE = generationPointCloud(kf,color,depth);
    pointcloud.push_back(pointcloude);
    keyFrameUpdated.notify_one();
}
//将某一帧关键帧转化为点云
pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::generationPointCloud(ORB_SLAM2::KeyFrame *kf,
                                                                                        cv::Mat &color,
                                                                                        cv::Mat &depth)
{
    PointCloud::Ptr tmp(new PointCloud());
    for (int m = 0; m <depth.rows ;m+=3)
    {
        for (int n = 0; n <depth.cols ;n=n+3)
        {
            float d=depth.ptr<float>(m)[n];
            if (d<0.01||d>5)
                continue;
            PointT p;
            p.z=d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;
            p.b=color.ptr<uchar>(m)[n*3];
            p.g=color.ptr<uchar>(m)[n*3+1];
            p.r=color.ptr<uchar>(m)[n*3+2];
            tmp->points.push_back(p);
        }

    }
    return tmp;
}
void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("PointCloud viewer");
    while (1)
    {
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if(shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
            keyFrameUpdated.wait(lck_keyframeUpdated);
        }
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }
        if(loopbusy || bStop)
        {
            //cout<<"loopbusy || bStop"<<endl;
            continue;
        }
        cloudbusy = true;
        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {


            PointCloud::Ptr p (new PointCloud);
            pcl::transformPointCloud( *(pointcloud[i].pcE), *p, pointcloud[i].T.inverse().matrix());
            //cout<<"处理好第i个点云"<<i<<endl;
            *globalMap += *p;
            //PointCloud::Ptr tmp(new PointCloud());
            //voxel.setInputCloud( globalMap );
            // voxel.filter( *tmp );
            //globalMap->swap( *tmp );
        }
        // depth filter and statistical removal
        PointCloud::Ptr tmp1 ( new PointCloud );
        //去除点云中的离群点
        statistical_filter.setInputCloud(globalMap);
        statistical_filter.filter( *tmp1 );
        //体素滤波器进行下采样
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud( tmp1 );
        voxel.filter( *globalMap );
        //globalMap->swap( *tmp );
        viewer.showCloud( globalMap );
        cout<<"show global map, size="<<N<<"   "<<globalMap->points.size()<<endl;
        lastKeyframeSize = N;
        cloudbusy = false;
    }
}
void PointCloudMapping::save()
{
    pcl::io::savePCDFile( "result.pcd", *globalMap );
    cout<<"globalMap save finished"<<endl;
}
void PointCloudMapping::updatecloud()
{
    if(!cloudbusy)
    {
        loopbusy = true;
        cout<<"startloopmappoint"<<endl;
        PointCloud::Ptr tmp1(new PointCloud);
        for (int i=0;i<currentvpKFs.size();i++)
        {
            for (int j=0;j<pointcloud.size();j++)
            {
                if(pointcloud[j].pcID==currentvpKFs[i]->mnFrameId)
                {
                    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(currentvpKFs[i]->GetPose() );
                    PointCloud::Ptr cloud(new PointCloud);
                    pcl::transformPointCloud( *pointcloud[j].pcE, *cloud, T.inverse().matrix());
                    *tmp1 +=*cloud;

                    //cout<<"第pointcloud"<<j<<"与第vpKFs"<<i<<"匹配"<<endl;
                    continue;
                }
            }
        }
        cout<<"finishloopmap"<<endl;
        PointCloud::Ptr tmp2(new PointCloud());
        voxel.setInputCloud( tmp1 );
        voxel.filter( *tmp2 );
        globalMap->swap( *tmp2 );
        //viewer.showCloud( globalMap );
        loopbusy = false;
        //cloudbusy = true;
        loopcount++;

        //*globalMap = *tmp1;
    }
}

}
