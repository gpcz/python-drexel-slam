import scipy.io
import numpy
import pylab
import math
from slamutil import *

# Constants
GPS_FRAME = 1
MOVEMENT_FRAME = 2
LASER_FRAME = 3

# Load B.Green's dataset.
trueLocs = scipy.io.loadmat('beac_juan3.mat')
dataset = scipy.io.loadmat('data_set.mat')

# Copied straight from B.Green's SLAM.m
L=2.83; # distance between front and rear axles
h=0.76; # distance between center of rear axle and encoder
b=0.5;  # vertical distance from rear axle to laser
a=3.78; # horizontal distance from rear axle to laser
startLong = dataset['GPSLon'][0][0]
startLat = dataset['GPSLat'][0][0]
PI = math.pi
init_est = numpy.ndarray(shape=(3,1), buffer=numpy.array([[float(startLong)],
                                                          [float(startLat)],
                                                          [-126.0*PI/180.0]]))

# The dataset starts doing weird things, so we stop the simulation before
# the end.
timeLimit = 100;

##
# This computes an array of points that correspond to the robot's true path.
#
# @param dataset The Matlab data set in question.
# @return An array of points corresponding to true positions (from DGPS).
def TruePath(dataset):
  result = numpy.array([[],[]])
  longitudeRet = MakeArrayIterator(dataset['GPSLon'][0])
  latitudeRet = MakeArrayIterator(dataset['GPSLat'][0])
  for i in range(len(dataset['Time'][0])-1):
    if dataset['Time'][0][i] >= dataset['Time'][0][0]+timeLimit:
      break
    if dataset['Sensor'][0][i] == GPS_FRAME:
      result = numpy.append(result,[[longitudeRet()],[latitudeRet()]],1)
  return result

##
# This computes an array of points that correspond to blind dead reckoning.
#
# @param dataset The Matlab data set in question.
# @return An array of points corresponding to dead reckoning movement.
def DeadReckoning(dataset):
  dtRet = MakeArrayDiffIterator(dataset['Time'][0])
  velRet = MakeArrayIterator(dataset['Velocity'][0])
  steerRet = MakeArrayIterator(dataset['Steering'][0])
  vehicleModel = AckermanVehicle(L,h,b,a,init_est,velRet(),steerRet())

  result = init_est
  for i in range(1,len(dataset['Time'][0])-1):
    dt = dtRet()
    vehicleModel.predict(dt)
    result = numpy.append(result,numpy.zeros((result.shape[0],1)),1)
    result[0:2,i] = vehicleModel.state[0:2,0]
    if dataset['Sensor'][0][i] == MOVEMENT_FRAME:
      vehicleModel.new_steering(velRet(),steerRet())
  return result

##
# This computes a new state and probability estimate for the robot using the
# Extended Kalman Filter (EKF).
#
# @param state The current state vector.
# @param Pest The current probability vector.
# @param markID The ID of the observed landmark.
# @param z The observation vector.
# @return A vector containing a new state vector and a new probability vector.
def EKFUpdate(state,Pest,markID,z):
  subt = numpy.subtract
  ndot = numpy.dot
  state = numpy.transpose(numpy.matrix(state))
  Jh = ObservationJacobian(state,markID)
  Jht = numpy.transpose(Jh)
  dx = state[markID*2+3,0]-state[0,0]
  dy = state[markID*2+4,0]-state[1,0]
  r = math.sqrt(dx*dx+dy*dy)
  H = [[r],[math.atan2(dy,dx)-state[2,0]+PI/2.0]]
  inno = numpy.subtract(z,H)
  inno[1,0] = NormalizeAngle(inno[1,0])
  V = numpy.identity(2)
  Vt = numpy.transpose(V)
  R = numpy.array([[0.1, 0],[0, (PI/180.0)*(PI/180.0)]])
  S = numpy.dot(Jh,numpy.dot(Pest,Jht))+numpy.dot(V,numpy.dot(R,Vt))
  K = numpy.dot(Pest,numpy.dot(numpy.transpose(Jh),numpy.linalg.inv(S)))
  wtf = numpy.dot(K,inno)
  state = state + wtf
  Pest = numpy.matrix(ndot(subt(numpy.identity(K.shape[0]),ndot(K,Jh)),Pest))
  return [state,Pest]

def SLAM(dataset):
  dtRet = MakeArrayDiffIterator(dataset['Time'][0])
  velRet = MakeArrayIterator(dataset['Velocity'][0])
  steerRet = MakeArrayIterator(dataset['Steering'][0])
  laserRet = MakeArrayIterator(dataset['Laser'])
  intensityRet = MakeArrayIterator(dataset['Intensity'])
  vehicleModel = AckermanVehicle(L,h,b,a,init_est,velRet(),steerRet())
  factor = (15.0*PI/180.0)
  Pest = numpy.array([[0.01,0,0],[0,0.01,0],[0,0,factor]])
  Q = numpy.array([[0.5,0.0,0.0],[0,0.5,0.0],[0.0,0.0,factor*factor]])
  W = numpy.array([[1,0,0],[0,1,0],[0,0,1]])

  result = init_est
  for i in range(1,len(dataset['Time'][0])-1):
    if dataset['Time'][0][i] >= dataset['Time'][0][0]+timeLimit:
      break
    dt = dtRet()
    vehicleModel.predict(dt)
    result = numpy.append(result,numpy.zeros((result.shape[0],1)),1)
    result[:,i] = result[:,i-1]
    result[0:3,i] = vehicleModel.state[0:3,0]
    A = vehicleModel.jacobian(dt,(result.shape[0]-3)/2)
    At = numpy.transpose(A)
    Wt = numpy.transpose(W)
    Pest = numpy.dot(A,numpy.dot(Pest,At))-numpy.dot(W,numpy.dot(Q,Wt))
    if dataset['Sensor'][0][i] == MOVEMENT_FRAME:
      vehicleModel.new_steering(velRet(),steerRet())
    if dataset['Sensor'][0][i] == LASER_FRAME:
      laserMeasure = laserRet()
      intensityMeasure = intensityRet()
      clumps = FindClumps(laserMeasure,intensityMeasure)
      if len(clumps) > 0:
        [clumpRange,clumpBearings] = ClumpsToRangeBearing(clumps)
        for count in range(len(clumpRange)):
          closest = FindClosestLandmark(result[0:3,i],
                                        clumpRange[count],
                                        clumpBearings[count],
                                        MakeLandmarkArray(result))
          landmark = FindGlobalLaserCoord(result[0:3,i],
                                          clumpRange[count],
                                          clumpBearings[count])
          if closest != False:
            [closeDist,theAddr] = closest
            if closeDist > 2.0:
              [result,Pest,theAddr,Q,W] = AddNewLandmark(result,
                                                         Pest,
                                                         landmark,Q,W)
          else:
            [result,Pest,theAddr,Q,W] = AddNewLandmark(result,
                                                       Pest,
                                                       landmark,Q,W)
          [newState,Pest] = EKFUpdate(result[:,i],
                                      Pest,
                                      theAddr,
                                      numpy.array([[clumpRange[count]],
                                                   [clumpBearings[count]]]))
          result[:,i] = numpy.transpose(newState[:,0])
          vehicleModel.plantState(newState[0:3,0])
  return result

truep = TruePath(dataset)
deadr = SLAM(dataset)
pylab.hold(True)
pylab.plot(truep[0,:],truep[1,:],'.',deadr[0,:],deadr[1,:],'-')
[estLocsX,estLocsY] = MakeScatterplotArrays(deadr)
pylab.scatter(trueLocs['estbeac'][:,0],trueLocs['estbeac'][:,1],s=40)
pylab.scatter(estLocsX,estLocsY,s=10,c='r')
pylab.hold(False)
pylab.show()
