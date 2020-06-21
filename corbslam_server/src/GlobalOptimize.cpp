//
// Created by lifu on 6/29/17.
//


#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <Eigen/StdVector>

#include "ORBmatcher.h"
#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"

#include "GlobalOptimize.h"

namespace CORBSLAM_SERVER{

    GlobalOptimize::GlobalOptimize(ServerMap *tgm) {

        this->globalMAp = tgm;

        this->mpCacher = tgm->pCacher;

        this->mbRunningGBA = false;

        this->mbFixScale = true;

        this->mnCovisibilityConsistencyTh = 3;

    }

    void GlobalOptimize::setCurentKeyFrame(KeyFrame *pCurKF) {

        this->mpCurrentKF = pCurKF;

    }

    void GlobalOptimize::setCandidates(std::vector<KeyFrame *> pCandites) {

        this->mvpEnoughConsistentCandidates = pCandites;

    }

    bool GlobalOptimize::ComputeSim3()
    {
        // For each consistent loop candidate we try to compute a Sim3

        cout << "ComputeSim3 start!\n";
        const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

        // We compute first ORB matches for each candidate
        // If enough matches are found, we setup a Sim3Solver
        ORBmatcher matcher(0.75,true);

        vector<Sim3Solver*> vpSim3Solvers;         // 每个候选帧都有一个Sim3Solver
        vpSim3Solvers.resize(nInitialCandidates);

        vector<vector<MapPoint*> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nInitialCandidates);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nInitialCandidates);

        int nCandidates=0; //candidates with enough matches

        for(int i=0; i<nInitialCandidates; i++)
        {
            // 步骤1：从筛选的闭环候选帧中取出一帧关键帧pKF
            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // avoid that local mapping erase it while it is being processed in this thread
            pKF->SetNotErase();

            if(pKF->isBad())
            {
                vbDiscarded[i] = true;
                continue;
            }
            // 步骤2：将当前帧mpCurrentKF与闭环候选关键帧pKF匹配
            // 通过bow加速得到mpCurrentKF与pKF之间的匹配特征点，vvpMapPointMatches是匹配特征点对应的MapPoints
            int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
                // 构造Sim3求解器
                // 如果mbFixScale为true，则是6DoFf优化（双目 RGBD），如果是false，则是7DoF优化（单目）
                Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
                pSolver->SetRansacParameters(0.99,20,300);
                vpSim3Solvers[i] = pSolver;
            }

            nCandidates++;
        }

        bool bMatch = false;// 用于标记是否有一个候选帧通过Sim3的求解与优化

        // Perform alternatively RANSAC iterations for each candidate
        // until one is succesful or all fail
        // 一直循环所有的候选帧，每个候选帧迭代5次，如果5次迭代后得不到结果，就换下一个候选帧
        // 直到有一个候选帧首次迭代成功bMatch为true，或者某个候选帧总的迭代次数超过限制，直接将它剔除
        while(nCandidates>0 && !bMatch)
        {
            for(int i=0; i<nInitialCandidates; i++)
            {
                if(vbDiscarded[i])
                    continue;

                KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;
                // 步骤3：对步骤2中有较好的匹配的关键帧求取Sim3变换
                Sim3Solver* pSolver = vpSim3Solvers[i];
                // 最多迭代5次，返回的Scm是候选帧pKF到当前帧mpCurrentKF的Sim3变换（T12）
                cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);
                // 经过n次循环，每次迭代5次，总共迭代 n*5 次
                // 总迭代次数达到最大限制还没有求出合格的Sim3变换，该候选帧剔除
                // If Ransac reachs max. iterations discard keyframe
                if(bNoMore)
                {
                    vbDiscarded[i]=true;
                    nCandidates--;
                }

                // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
                if(!Scm.empty())
                {
                    vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                    for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                    {
                        if(vbInliers[j])
                            vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                    }

                    cv::Mat R = pSolver->GetEstimatedRotation();// 候选帧pKF到当前帧mpCurrentKF的R（R12）
                    cv::Mat t = pSolver->GetEstimatedTranslation();// 候选帧pKF到当前帧mpCurrentKF的t（t12），当前帧坐标系下，方向由pKF指向当前帧
                    const float s = pSolver->GetEstimatedScale();// 候选帧pKF到当前帧mpCurrentKF的变换尺度s（s12）
                    // 查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数，之前使用SearchByBoW进行特征点匹配时会有漏匹配）
                    // 通过Sim3变换，确定pKF1的特征点在pKF2中的大致区域，同理，确定pKF2的特征点在pKF1中的大致区域
                    // 在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新匹配vpMapPointMatches
                    matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);
                    // 步骤5：Sim3优化，只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
                    // OpenCV的Mat矩阵转成Eigen的Matrix类型
                    g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                    // 优化mpCurrentKF与pKF对应的MapPoints间的Sim3，得到优化后的量gScm
                    const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                    // If optimization is succesful stop ransacs and continue
                    if(nInliers>=20)
                    {
                        bMatch = true;
                        // mpMatchedKF就是最终闭环检测出来与当前帧形成闭环的关键帧
                        mpMatchedKF = pKF;
                        // 得到从世界坐标系到该候选帧的Sim3变换，Scale=1
                        g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                        // 得到g2o优化后从世界坐标系到当前帧的Sim3变换
                        mg2oScw = gScm*gSmw;
                        mScw = Converter::toCvMat(mg2oScw);

                        mvpCurrentMatchedPoints = vpMapPointMatches;
                        break;// 只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
                    }
                }
            }

            cout << "nCandidates: " << nCandidates << " match: " << bMatch << endl;
         }
        // 没有一个闭环匹配候选帧通过Sim3的求解与优化
        if(!bMatch)
        {
            // 清空mvpEnoughConsistentCandidates
            cout << "computeSim3 end! Failure" << endl;
            for(int i=0; i<nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            mpCurrentKF->SetErase();
            return false;
        }
        // 步骤6：取出闭环匹配上关键帧的相连关键帧，得到它们的MapPoints放入mvpLoopMapPoints
        // 注意是匹配上的那个关键帧：mpMatchedKF
        // 将mpMatchedKF相连的关键帧全部取出来放入vpLoopConnectedKFs
        // 将vpLoopConnectedKFs的MapPoints取出来放入mvpLoopMapPoints
        // Retrieve MapPoints seen in Loop Keyframe and neighbors
        vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
        vpLoopConnectedKFs.push_back(mpMatchedKF);
        mvpLoopMapPoints.clear();
        for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
        {
            KeyFrame* pKF = *vit;
            vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
            for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
            {
                MapPoint* pMP = vpMapPoints[i];
                if(pMP)
                {
                    if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                    {
                        mvpLoopMapPoints.push_back(pMP);
                        pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                    }
                }
            }
        }
        // 步骤7：将闭环匹配上关键帧以及相连关键帧的MapPoints投影到当前关键帧进行投影匹配
        // 根据投影查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数）
        // 根据Sim3变换，将每个mvpLoopMapPoints投影到mpCurrentKF上，并根据尺度确定一个搜索区域，
        // 根据该MapPoint的描述子与该区域内的特征点进行匹配，如果匹配误差小于TH_LOW即匹配成功，更新mvpCurrentMatchedPoints
        // mvpCurrentMatchedPoints将用于SearchAndFuse中检测当前帧MapPoints与匹配的MapPoints是否存在冲突

        // Find more matches projecting with the computed Sim3
        matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

        // If enough matches accept Loop
        int nTotalMatches = 0;
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
                nTotalMatches++;
        }

        cout << "computeSim3 end! nTotalMatches: " << nTotalMatches << endl;

        if(nTotalMatches>=40)
        {
            for(int i=0; i<nInitialCandidates; i++)
                if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                    mvpEnoughConsistentCandidates[i]->SetErase();
            return true;
        }
        else
        {
            for(int i=0; i<nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            mpCurrentKF->SetErase();
            return false;
        }

    }

    void GlobalOptimize::CorrectLoop()
    {
        cout << "Loop detected!" << endl;

        // Send a stop signal to Local Mapping
        // Avoid new keyframes are inserted while correcting the loop

        // If a Global Bundle Adjustment is running, abort it
        if(isRunningGBA())
        {
            unique_lock<mutex> lock(mMutexGBA);
            mbStopGBA = true;

            mnFullBAIdx++;

            if(mpThreadGBA)
            {
                mpThreadGBA->detach();
                delete mpThreadGBA;
            }
        }

        // Ensure current keyframe is updated
        mpCurrentKF->UpdateConnections();
        // 步骤2：通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的MapPoints
        // 当前帧与世界坐标系之间的Sim变换在ComputeSim3函数中已经确定并优化，
        // 通过相对位姿关系，可以确定这些相连的关键帧与世界坐标系之间的Sim3变换
        // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation

        // 取出与当前帧相连的关键帧，包括当前关键帧
        mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
        mvpCurrentConnectedKFs.push_back(mpCurrentKF);

        KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
        CorrectedSim3[mpCurrentKF]=mg2oScw;
        cv::Mat Twc = mpCurrentKF->GetPoseInverse();

        {
            // Get Map Mutex
            unique_lock<mutex> lock(mpCacher->getMpMap()->mMutexMapUpdate);

            for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
            {
                KeyFrame* pKFi = *vit;

                cv::Mat Tiw = pKFi->GetPose();

                if(pKFi!=mpCurrentKF)
                {
                    cv::Mat Tic = Tiw*Twc;
                    cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                    cv::Mat tic = Tic.rowRange(0,3).col(3);
                    g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                    g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                    //Pose corrected with the Sim3 of the loop closure
                    CorrectedSim3[pKFi]=g2oCorrectedSiw;
                }

                cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
                cv::Mat tiw = Tiw.rowRange(0,3).col(3);
                g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
                //Pose without correction
                NonCorrectedSim3[pKFi]=g2oSiw;
            }

            // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
            for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;
                g2o::Sim3 g2oCorrectedSiw = mit->second;
                g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

                g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

                vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
                for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
                {
                    MapPoint* pMPi = vpMPsi[iMP];
                    if(!pMPi)
                        continue;
                    if(pMPi->isBad())
                        continue;
                    if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                        continue;

                    // Project with non-corrected pose and project back with corrected pose
                    cv::Mat P3Dw = pMPi->GetWorldPos();
                    Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                    Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                    pMPi->SetWorldPos(cvCorrectedP3Dw);
                    mpCacher->addUpdateMapPoint( pMPi );

                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;
                    pMPi->UpdateNormalAndDepth();
                }

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
                Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
                double s = g2oCorrectedSiw.scale();

                eigt *=(1./s); //[R t/s;0 1]

                cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

                pKFi->SetPose(correctedTiw);

                mpCacher->addUpdateKeyframe( pKFi );

                // Make sure connections are updated
                pKFi->UpdateConnections();
            }

            // Start Loop Fusion
            // Update matched map points and replace if duplicated
            for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
            {
                if(mvpCurrentMatchedPoints[i])
                {
                    MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                    MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                    if(pCurMP)
                        pCurMP->Replace(pLoopMP);
                    else
                    {
                        mpCurrentKF->AddMapPoint(pLoopMP,i);
                        pLoopMP->AddObservation(mpCurrentKF,i);
                        pLoopMP->ComputeDistinctiveDescriptors();
                    }
                }
            }

        }

        // Project MapPoints observed in the neighborhood of the loop keyframe
        // into the current keyframe and neighbors using corrected poses.
        // Fuse duplications.
        SearchAndFuse(CorrectedSim3);

        // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
        map<KeyFrame*, set<KeyFrame*> > LoopConnections;

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;
            vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

            // Update connections. Detect new links.
            pKFi->UpdateConnections();
            LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
            for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
            {
                LoopConnections[pKFi].erase(*vit_prev);
            }
            for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
            {
                LoopConnections[pKFi].erase(*vit2);
            }
        }

        // Optimize graph
        //TODO: chage the mpmap to cache
        Optimizer::OptimizeEssentialGraph(mpCacher, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

        // Add loop edge
        mpMatchedKF->AddLoopEdge(mpCurrentKF);
        mpCurrentKF->AddLoopEdge(mpMatchedKF);

        // Launch a new thread to perform Global Bundle Adjustment
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&GlobalOptimize::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

        cout << "Loop Closed!" << endl;

        mLastLoopKFid = mpCurrentKF->mnId;
    }

    void GlobalOptimize::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
    {
        ORBmatcher matcher(0.8);

        for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
        {
            KeyFrame* pKF = mit->first;

            g2o::Sim3 g2oScw = mit->second;
            cv::Mat cvScw = Converter::toCvMat(g2oScw);

            vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
            matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

            // Get Map Mutex
            unique_lock<mutex> lock(mpCacher->getMpMap()->mMutexMapUpdate);
            const int nLP = mvpLoopMapPoints.size();
            for(int i=0; i<nLP;i++)
            {
                MapPoint* pRep = vpReplacePoints[i];
                if(pRep)
                {
                    pRep->Replace(mvpLoopMapPoints[i]);
                }
            }
        }
    }


    void GlobalOptimize::RunGlobalBundleAdjustment(unsigned long nLoopKF)
    {
        cout << "Starting Global Bundle Adjustment" << endl;

        time_t start_t = clock();


        int idx =  mnFullBAIdx;
        //TODO:change mpMap to cache
        Optimizer::GlobalBundleAdjustemnt(mpCacher,10,&mbStopGBA,nLoopKF,false);

        // Update all MapPoints and KeyFrames
        // Local Mapping was active during BA, that means that there might be new keyframes
        // not included in the Global BA and they are not consistent with the updated map.
        // We need to propagate the correction through the spanning tree
        {
            unique_lock<mutex> lock(mMutexGBA);
            if(idx!=mnFullBAIdx)
                return;

            if(!mbStopGBA)
            {
                cout << "Global Bundle Adjustment finished" << endl;
                cout << "Updating map ..." << endl;

                // Get Map Mutex
                unique_lock<mutex> lock(mpCacher->getMpMap()->mMutexMapUpdate);

                // Correct keyframes starting at map first keyframe
                list<KeyFrame*> lpKFtoCheck(mpCacher->getMpMap()->mvpKeyFrameOrigins.begin(),mpCacher->getMpMap()->mvpKeyFrameOrigins.end());

                while(!lpKFtoCheck.empty())
                {
                    KeyFrame* pKF = lpKFtoCheck.front();
                    const set<KeyFrame*> sChilds = pKF->GetChilds();
                    cv::Mat Twc = pKF->GetPoseInverse();
                    for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                    {
                        KeyFrame* pChild = *sit;
                        if(pChild->mnBAGlobalForKF!=nLoopKF)
                        {
                            cv::Mat Tchildc = pChild->GetPose()*Twc;
                            pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                            pChild->mnBAGlobalForKF=nLoopKF;

                        }
                        lpKFtoCheck.push_back(pChild);
                    }

                    pKF->mTcwBefGBA = pKF->GetPose();
                    pKF->SetPose(pKF->mTcwGBA);
                    mpCacher->addUpdateKeyframe( pKF );
                    lpKFtoCheck.pop_front();
                }

                cout << "Correct MapPoints\n";
                // Correct MapPoints
                const vector<MapPoint*> vpMPs = mpCacher->GetAllMapPointsFromMap();

                for(size_t i=0; i<vpMPs.size(); i++)
                {
                    MapPoint* pMP = vpMPs[i];

                    if(pMP->isBad())
                        continue;

                    if(pMP->mnBAGlobalForKF==nLoopKF)
                    {
                        // If optimized by Global BA, just update
                        pMP->SetWorldPos(pMP->mPosGBA);
                        mpCacher->addUpdateMapPoint( pMP );
                    }
                    else
                    {
                        // Update according to the correction of its reference keyframe
                        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                        if( pRefKF ) {

                            if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                                continue;

                            // Map to non-corrected camera
                            cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                            cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                            cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                            // Backproject using corrected camera
                            cv::Mat Twc = pRefKF->GetPoseInverse();
                            cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                            cv::Mat twc = Twc.rowRange(0,3).col(3);

                            pMP->SetWorldPos(Rwc*Xc+twc);
                            mpCacher->addUpdateMapPoint( pMP );
                        }

                    }
                }

                cout << "Map updated!" << endl;
            }

            mbFinishedGBA = true;
            mbRunningGBA = false;
        }
        time_t end_t = clock();

        fstream GBAf;
        GBAf.open( "GBA.txt");
        GBAf << (double) (end_t - start_t ) / (double) CLOCKS_PER_SEC << endl;
        GBAf.close();

    }

}