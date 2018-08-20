#include "../include/optic_flow/utilityFunctions.h"
#include <ros/ros.h>

void rotate2d(double &x, double &y, double alpha) {
  double cs = cos(alpha);  // cos and sin of rot maxtrix
  double sn = sin(alpha);

  double x_n = x * cs - y * sn;
  double y_n = x * sn + y * cs;
  x          = x_n;
  y          = y_n;
}

void rotate2d(cv::Point2f &pt, double alpha) {
  double cs = cos(alpha);  // cos and sin of rot maxtrix
  double sn = sin(alpha);

  double x_n = pt.x * cs - pt.y * sn;
  double y_n = pt.x * sn + pt.y * cs;
  pt.x       = x_n;
  pt.y       = y_n;
}


cv::Point2f pointMean(std::vector<cv::Point2f> pts) {
  float mx     = 0;
  float my     = 0;
  int   numPts = 0;
  for (uint i = 0; i < pts.size(); i++) {
    if (!std::isnan(pts[i].x) && !std::isnan(pts[i].y)) {
      mx += pts[i].x;
      my += pts[i].y;
      numPts++;
    }
  }

  if (numPts > 0) {
    // now we're talking..
    return cv::Point2f(mx / (float)numPts, my / (float)numPts);
  }

  // what do you want me to do with this?
  return cv::Point2f(nanf(""), nanf(""));
}

float getDistSq(cv::Point2f p1, cv::Point2f p2) {
  return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
}

float getNormSq(cv::Point2f p1) {
  return pow(p1.x, 2) + pow(p1.y, 2);
}

cv::Point2f twoPointMean(cv::Point2f p1, cv::Point2f p2) {
  return cv::Point2f((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
}

cv::Point2f allsacMean(std::vector<cv::Point2f> pts, float thresholdRadius_sq, int *chosen) {
  // For every two points get their mean and do the typical RANSAC things...
  if (pts.size() <= 2) {  // weve got less or same number (or zero?) of points as number to choose
    return pointMean(pts);
  }

  cv::Point2f              currMean;
  std::vector<cv::Point2f> currIter;

  int         bestIter_num = 0;
  cv::Point2f bestIter;

  for (uint i = 0; i < pts.size(); i++) {
    for (uint j = i; j < pts.size(); j++) {  // iterate over all pairs

      currIter.clear();

      currMean = twoPointMean(pts[i], pts[j]);  // calc mean

      for (uint k = 0; k < pts.size(); k++) {  // choose those in threshold
        if (getDistSq(currMean, pts[k]) < thresholdRadius_sq) {
          currIter.push_back(pts[k]);
        }
      }

      if (currIter.size() > bestIter_num) {
        bestIter_num = currIter.size();
        bestIter     = pointMean(currIter);
        *chosen      = bestIter_num;
        if (bestIter_num >= pts.size()) {
          return bestIter;
        }
      }
    }
  }

  return bestIter;
}

double calcMean(std::vector<double> pts) {
  double sum = 0;
  for (int i = 0; i < pts.size(); i++) {
    sum += pts[i];
  }
  return sum / pts.size();
}

double allsacMean(std::vector<double> pts, float thresholdRadius, int *chosen) {
  // For every two points get their mean and do the typical RANSAC things...
  if (pts.size() <= 2) {  // weve got less or same number (or zero?) of points as number to choose
    return calcMean(pts);
  }

  double              currMean;
  std::vector<double> currIter;

  int    bestIter_num = 0;
  double bestIter;

  for (uint i = 0; i < pts.size(); i++) {
    for (uint j = i; j < pts.size(); j++) {  // iterate over all pairs

      currIter.clear();

      currMean = (pts[i] + pts[j]) / 2;  // calc mean

      for (uint k = 0; k < pts.size(); k++) {  // choose those in threshold
        if (absd(pts[k] - currMean) < thresholdRadius) {
          currIter.push_back(pts[k]);
        }
      }

      if (currIter.size() > bestIter_num) {
        bestIter_num = currIter.size();
        bestIter     = calcMean(currIter);
        *chosen      = bestIter_num;
        if (bestIter_num >= pts.size()) {
          return bestIter;
        }
      }
    }
  }

  return bestIter;
}

void multiplyAllPts(std::vector<cv::Point2f> &v, float mulx, float muly) {
  for (uint i = 0; i < v.size(); i++) {
    v[i].x *= mulx;
    v[i].y *= muly;
  }
}

void multiplyAllPts(std::vector<double> &v, float mul) {
  for (uint i = 0; i < v.size(); i++) {
    v[i] *= mul;
  }
}

void rotateAllPts(std::vector<cv::Point2f> &v, double alpha) {
  for (uint i = 0; i < v.size(); i++) {
    rotate2d(v[i], alpha);
  }
}

void addToAll(std::vector<cv::Point2f> &v, float adx, float ady) {
  for (uint i = 0; i < v.size(); i++) {
    v[i].x += adx;
    v[i].y += ady;
  }
}

cv::Point2f ransacMean(std::vector<cv::Point2f> pts, int numOfChosen, float thresholdRadius_sq, int numOfIterations) {
  if (pts.size() <= numOfChosen) {  // weve got less or same number (or zero?) of points as number to choose
    return pointMean(pts);
  }

  cv::Point2f              bestIter;          // save the best mean here
  uint                     bestIter_num = 0;  // save number of points in best mean
  std::vector<cv::Point2f> currIter;          // here goes current iteration
  cv::Point2f              currMean;

  for (uint i = 0; i < numOfIterations; i++) {  // ITERATE!!!
    currIter.clear();

    for (uint j = 0; j < numOfChosen; j++) {  // choose some points (one point can be chosen more times...)
      currIter.push_back(pts[rand() % pts.size()]);
    }

    currMean = pointMean(currIter);  // get the mean

    currIter.clear();  // clear this array

    for (uint k = 0; k < pts.size(); k++) {  // choose those in threshold
      if (getDistSq(currMean, pts[k]) < thresholdRadius_sq) {
        currIter.push_back(pts[k]);
      }
    }

    if (currIter.size() > bestIter_num) {
      bestIter_num = currIter.size();
      bestIter     = pointMean(currIter);
    }
  }

  return bestIter;
}

std::vector<cv::Point2f> getOnlyInAbsBound(std::vector<cv::Point2f> v, float up) {
  std::vector<cv::Point2f> ret;
  float                    upSq = up * up;
  float                    n;
  for (int i = 0; i < v.size(); i++) {
    n = getNormSq(v[i]);
    if (n < upSq) {
      ret.push_back(v[i]);
    }
  }
  return ret;
}


std::vector<double> getOnlyInAbsBound(std::vector<double> v, double up) {
  std::vector<double> ret;
  for (int i = 0; i < v.size(); i++) {
    if (absd(v[i]) < up) {
      ret.push_back(v[i]);
    }
  }
  return ret;
}

std::vector<cv::Point2f> removeNanPoints(std::vector<cv::Point2f> v) {
  std::vector<cv::Point2f> ret;
  for (int i = 0; i < v.size(); i++) {
    if (!isnanf(v[i].x) && !isnanf(v[i].y)) {
      ret.push_back(v[i]);
    }
  }
  return ret;
}

std::vector<double> removeNanPoints(std::vector<double> v) {
  std::vector<double> ret;
  for (int i = 0; i < v.size(); i++) {
    if (!std::isnan(v[i])) {
      ret.push_back(v[i]);
    }
  }
  return ret;
}

std::vector<cv::Point2f> getOnlyInRadiusFromTruth(cv::Point2f truth, std::vector<cv::Point2f> v, float rad) {
  std::vector<cv::Point2f> ret;
  float                    radSq = rad * rad;
  float                    n;
  for (int i = 0; i < v.size(); i++) {
    n = getDistSq(truth, v[i]);
    if (n < radSq) {
      ret.push_back(v[i]);
    }
  }
  return ret;
}

float absf(float x) {
  return x > 0 ? x : -x;
}

double absd(double x) {
  return x > 0 ? x : -x;
}

StatData analyzeSpeeds(ros::Time fromTime, std::vector<SpeedBox> speeds) {
  SpeedBox sb;

  float sum   = 0;
  float sumsq = 0;
  uint  num   = 0;

  float dif;

  float sumx   = 0;
  float sumxsq = 0;

  float sumy   = 0;
  float sumysq = 0;

  for (uint i = 0; i < speeds.size(); i++) {
    sb = speeds[i];

    if (sb.time > fromTime) {
      num++;
      dif = getDistSq(sb.speed, sb.odomSpeed);
      sumsq += dif;
      sum += sqrt(dif);


      dif = absf(sb.odomSpeed.x - sb.speed.x);
      sumx += dif;
      sumxsq += dif * dif;


      dif = absf(sb.odomSpeed.y - sb.speed.y);
      sumy += dif;
      sumysq += dif * dif;
    }
  }


  float numf = (float)num;
  float exx  = sumsq / numf;  // E(X^2)
  float ex   = sum / numf;

  StatData sd;
  sd.mean   = ex;
  sd.stdDev = sqrt(exx - ex * ex);
  sd.num    = num;

  sd.meanX   = sumx / numf;
  sd.stdDevX = sqrt(sumxsq / numf - sd.meanX * sd.meanX);

  sd.meanY   = sumy / numf;
  sd.stdDevY = sqrt(sumysq / numf - sd.meanY * sd.meanY);

  return sd;
}


std::vector<cv::Point2f> estimateTranRotVvel(std::vector<cv::Point2f> vectors, double a, double fx, double fy, double range, double allsac_radiusSQ,
                                             double duration, double max_vert_speed, double max_yaw_speed) {
  // a is the distance of points from origin (typ. 40px)
  std::vector<cv::Point2f> ret;
  if (vectors.size() != 9) {
    ret.push_back(cv::Point2f(0, 0));
    ret.push_back(cv::Point2f(0, 0));
    return ret;
  }

  multiplyAllPts(vectors, 1, -1);

  cv::Point2f r1 = vectors[0];
  cv::Point2f r4 = vectors[1];
  cv::Point2f r7 = vectors[2];
  cv::Point2f r2 = vectors[3];
  cv::Point2f r5 = vectors[4];
  cv::Point2f r8 = vectors[5];
  cv::Point2f r3 = vectors[6];
  cv::Point2f r6 = vectors[7];
  cv::Point2f r9 = vectors[8];

  // estimate translation
  std::vector<cv::Point2f> t_est;
  t_est.push_back((r1 + r9));
  t_est.push_back((r3 + r7));
  t_est.push_back((r2 + r8));
  t_est.push_back((r4 + r6));
  multiplyAllPts(t_est, 0.5, 0.5);
  t_est.push_back(r5);


  /*ROS_INFO("[OpticFlow]: Translation estimates");
  for(int i = 0;i < t_est.size();i++){
      ROS_INFO("[OpticFlow]: %f %f",t_est[i].x,t_est[i].y);
  }*/

  // allsac_radius /=  range/(fx*duration); // recalc allsac radius to pixels

  // ROS_INFO_THROTTLE(0.5,"[OpticFlow]: Allsac radius: %f px.",allsac_radius);
  t_est = removeNanPoints(t_est);

  multiplyAllPts(t_est, range / (fx * duration), range / (fy * duration));

  int ch;

  cv::Point2f tr = allsacMean(t_est, allsac_radiusSQ, &ch);

  // ROS_INFO_THROTTLE(0.5,"[OpticFlow]: Translation est %f,%f m/s - chosen: %d",tr.x,tr.y,ch);

  tr.x *= (fx * duration) / range;
  tr.y *= -(fy * duration) / range;

  // ROS_INFO_THROTTLE(0.5,"[OpticFlow]: Translation est %f,%f px - chosen: %d",tr.x,tr.y,ch);

  ret.push_back(tr);

  // get estimates of R and S
  std::vector<double> r_est;
  std::vector<double> s_est;

  r1 -= tr;
  r2 -= tr;
  r3 -= tr;
  r4 -= tr;

  r6 -= tr;
  r7 -= tr;
  r8 -= tr;
  r9 -= tr;

  // ROS_INFO("[OpticFlow]: Vectors: \n(%f,%f) , (%f,%f) , (%f,%f)\n(%f,%f) , (%f,%f) , (%f,%f)\n(%f,%f) , (%f,%f) , (%f,%f)\n",
  // r1.x,r1.y,r2.x,r2.y,r3.x,r3.y,r4.x,r4.y,r5.x,r5.y,r6.x,r6.y,r7.x,r7.y,r8.x,r8.y,r9.x,r9.y);

  r_est.push_back((r1.y + r1.x) / 2);  // diagonal vectors
  s_est.push_back((r1.y - r1.x) / 2);

  r_est.push_back((-r9.x - r9.y) / 2);
  s_est.push_back((-r9.y + r9.x) / 2);

  r_est.push_back((r3.x - r3.y) / 2);
  s_est.push_back((r3.y + r3.x) / 2);

  r_est.push_back((r7.y - r7.x) / 2);
  s_est.push_back((-r7.y - r7.x) / 2);

  r_est.push_back(r4.y);  // cross vectors
  s_est.push_back(-r4.x);

  r_est.push_back(-r6.y);
  s_est.push_back(r6.x);

  r_est.push_back(r2.x);
  s_est.push_back(r2.y);

  r_est.push_back(-r8.x);
  s_est.push_back(-r8.y);

  /*ROS_INFO("[OpticFlow]: Rotation and scale estimates");
  for(int i = 0;i < r_est.size();i++){
      ROS_INFO("[OpticFlow]: %f %f",r_est[i],s_est[i]);
  }*/

  r_est = removeNanPoints(r_est);
  s_est = removeNanPoints(s_est);


  // Yaw velocity
  multiplyAllPts(r_est, 1.0 / (duration * a));  // recalc to rad/s

  // r_est = getOnlyInAbsBound(r_est,max_yaw_speed); // abs. bounding
  double rot = 0;
  for (int i = 0; i < r_est.size(); i++) {
    rot += r_est[i];
  }
  rot /= ((double)r_est.size());

  // ROS_INFO_THROTTLE(0.5,"[OpticFlow]: Rotation est %f px - chosen: %d",rot,ch);

  // Vertical speed
  multiplyAllPts(s_est, range / (duration * a));  // recalc to m/s
  // s_est = getOnlyInAbsBound(s_est,max_vert_speed); // absolute bounding

  // average
  double vert = 0;
  for (int i = 0; i < s_est.size(); i++) {
    vert += s_est[i];
  }
  vert /= ((double)s_est.size());

  /*ROS_INFO("[OpticFlow]: Rotation and scale estimates after");
  for(int i = 0;i < r_est.size();i++){
      ROS_INFO("[OpticFlow]: %f %f",r_est[i],s_est[i]);
  }*/


  if (absd(rot) > max_yaw_speed) {
    rot = nan("");
  }

  if (absd(vert) > max_vert_speed) {
    vert = nan("");
  }

  // ROS_INFO_THROTTLE(0.5,"[OpticFlow]: Rot est %f rad/s, VVel est %f m/s",rot,vert);

  ret.push_back(cv::Point2f(rot, vert));
  return ret;
}
