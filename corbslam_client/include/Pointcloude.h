//
// Created by sxy on 2020/6/26.
//

#ifndef SRC_POINTCLOUDE_H
#define SRC_POINTCLOUDE_H

//#include <pcl/common/transforms.h>
//#include <pcl/point_types.h>
//#include <pcl/filters/voxel_grid.h>
//#include <condition_variable>
//#include <pcl/io/pcd_io.h>
//#include <pcl/filters/statistical_outlier_removal.h>


namespace ORB_SLAM2
{

    class PointCloude
    {
        typedef pcl::PointXYZRGBA PointT;
        typedef pcl::PointCloud<PointT> PointCloud;
    public:
        PointCloude(){}
        PointCloud::Ptr pcE;
    public:
        Eigen::Isometry3d T;
        int pcID;
//protected:

    };

} //namespace ORB_SLAM







#endif //SRC_POINTCLOUDE_H
