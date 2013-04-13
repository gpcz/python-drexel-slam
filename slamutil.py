import math
import numpy

## Class that simulates a four-wheeled vehicle with Ackerman steering.
class AckermanVehicle:
  ## Constructor.  Requires specifying intrinsic knowledge of vehicle.
  # @param _L Distance between front and rear axles.
  # @param _h Distance between center of rear axle and encoder.
  # @param _b Vertical distance from rear axle to laser.
  # @param _a Horizontal distance from rear axle to laser.
  # @param _init_state Initial position and heading.
  # @param _init_Ve Initial forward (or backward) velocity.
  # @param _init_steer_a Initial steering angle.
  def __init__(self,_L,_h,_b,_a,_init_state,_init_Ve,_init_steer_a):
    self.L = _L
    self.h = _h
    self.b = _b
    self.a = _a
    self.state = _init_state
    self.Ve = _init_Ve
    self.steer_a = _init_steer_a
    self.Vc = self.measuredToVehicleCenter()

  ## Plants a new state into the simulator.  Useful for resetting.
  # @param newState The new state variable that replaces self.state.
  def plantState(self,newState):
    self.state = newState

  def predict(self,dt):
    phi = self.state[2,0]
    self.state = numpy.array([
        [self.state[0,0] + dt*self.Vc*math.cos(phi)-dt*self.Vc/self.L*math.tan(self.steer_a)*(self.a*math.sin(phi)+self.b*math.cos(phi))],
        [self.state[1,0] + dt*self.Vc*math.sin(phi)+dt*self.Vc/self.L*math.tan(self.steer_a)*(self.a*math.cos(phi)-self.b*math.sin(phi))],
        [NormalizeAngle(phi + dt*self.Vc/self.L*math.tan(self.steer_a))]])

  def jacobian(self,dt,numLandmarks):
    phi = self.state[2,0]
    result = numpy.identity(3+numLandmarks*2)
    result[0,2] = -dt*self.Vc*math.sin(phi)-dt*self.Vc/self.L*math.tan(self.steer_a)*(self.a*math.cos(phi)-self.b*math.sin(phi))
    result[1,2] = dt*self.Vc*math.cos(phi)-dt*self.Vc/self.L*math.tan(self.steer_a)*(self.a*math.sin(phi)+self.b*math.cos(phi))
    return result

  def measuredToVehicleCenter(self):
    return self.Ve / (1-(self.h/self.L)*math.tan(self.steer_a))

  def new_steering(self,_Ve,_steer_a):
    self.Ve = _Ve
    self.steer_a = _steer_a
    self.Vc = self.measuredToVehicleCenter()

def DistanceFormula(a,b):
  return math.sqrt((a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1]))

def FindClosestLandmark(robotState,lasrange,bearing,landmarks):
  if len(landmarks) == 0:
    return False
  measurementPoint = FindGlobalLaserCoord(robotState,lasrange,bearing)
  maxDist = DistanceFormula(measurementPoint,landmarks[0])
  maxAddress = 0
  for i in range(0,len(landmarks)):
    dist = DistanceFormula(measurementPoint,landmarks[i])
    if dist < maxDist:
      maxDist = dist
      maxAddress = i
  return [maxDist,maxAddress]

def NormalizeAngle(angle):
  if angle > math.pi:
    return angle-math.pi*2
  if angle < -math.pi:
    return angle+math.pi*2
  return angle

def MakeArrayDiffIterator(theArray):
  def Ret():
    Ret.theCount += 1
    return theArray[Ret.theCount]-theArray[Ret.theCount-1]
  Ret.theCount = 0
  return Ret

def MakeArrayIterator(theArray):
  def Ret():
    result = theArray[Ret.theCount]
    Ret.theCount += 1
    return result
  Ret.theCount = 0
  return Ret

def MakeScatterplotArrays(result):
  Xs = []
  Ys = []
  for i in range((result.shape[0]-3)/2):
    Xs.append(result[i*2+3,result.shape[1]-1])
    Ys.append(result[i*2+4,result.shape[1]-1])
  return [Xs,Ys]

def MakeLandmarkArray(result):
  #print result.shape[0]
  numMarks = (result.shape[0]-3)/2
  output = []
  for i in range(numMarks):
    #print "i: " + str(i)
    output.append([result[i*2+3,result.shape[1]-1],result[i*2+4,result.shape[1]-1]])
#  print output
  return output

def TackOnNewZeroRows(mat,num):
  mat = numpy.append(mat,numpy.zeros((mat.shape[0],num)),1)
  mat = numpy.append(mat,numpy.zeros((num,mat.shape[1])),0)
  return mat

def AddNewLandmark(result,Pest,newMark,Q,W):
  result = numpy.append(result,numpy.zeros((2,result.shape[1])),0)
  result[result.shape[0]-2,result.shape[1]-1] = newMark[0]
  result[result.shape[0]-1,result.shape[1]-1] = newMark[1]
  Pest = TackOnNewZeroRows(Pest,2)
  Pest[Pest.shape[0]-2,Pest.shape[1]-2] = 1000000.0
  Pest[Pest.shape[0]-1,Pest.shape[1]-1] = 1000000.0
  theAddr = (Pest.shape[0]-3)/2-1
  return [result,Pest,theAddr,TackOnNewZeroRows(Q,2),TackOnNewZeroRows(W,2)]

def ObservationZMatrix(v_state,markID):
  ximxv = v_state[3+2*markID]-v_state[0]
  yimyv = v_state[4+2*markID]-v_state[1]
  xvmxi = v_state[0]-v_state[3+2*markID]
  yvmyi = v_state[1]-v_state[4+2*markID]
  r = math.sqrt(ximxv*ximxv+yimyv*yimyv)
  theMatrix = numpy.array([[r],
                           [math.atan2(yimyv,ximxv)-v_state[2]+math.pi/2.0]])
  return theMatrix

def ObservationJacobian(v_state,markID):
  ximxv = v_state[3+2*markID,0]-v_state[0,0]
  yimyv = v_state[4+2*markID,0]-v_state[1,0]
  xvmxi = v_state[0,0]-v_state[3+2*markID,0]
  yvmyi = v_state[1,0]-v_state[4+2*markID,0]
  r = math.sqrt(ximxv*ximxv+yimyv*yimyv)
  theMatrix = numpy.array([[xvmxi/r,yvmyi/r,0],
                           [yimyv/(r*r),xvmxi/(r*r),-1.0]])
  theMatrix = numpy.append(theMatrix,numpy.zeros((2,v_state.shape[0]-3)),1)
  theMatrix[0,3+2*markID] = ximxv/r
  theMatrix[0,4+2*markID] = yimyv/r
  theMatrix[1,3+2*markID] = yvmyi/(r*r)
  theMatrix[1,4+2*markID] = ximxv/(r*r)
  return theMatrix

def FindGlobalLaserCoord(v_state,r,bearing):
  phi = v_state[2]
  laserBear = phi+bearing-math.pi/2.0
  return [v_state[0]+r*math.cos(laserBear),v_state[1]+r*math.sin(laserBear)]

def ClumpsToRangeBearing(clumps):
  ranges = []
  bearings = []
  for i in clumps:
    theRange = float(sum(i[1]))/float(len(i[1]))
    if theRange < 15.0:
      ranges.append(theRange)
      bearings.append(( float(float(i[0])/2+ (float(len(i[1]))*0.5)/2) )*math.pi/180.0)
  return [ranges,bearings]

def FindClumps(laser,intensity):
  if len(laser) == len(intensity):
    result = []
    first = -1
    numBlanks = 0
    laserExcerpt = []
    intensityExcerpt = []
    for i in range(len(laser)):
      if intensity[i] > 0:
        if first == -1:
          first = i
          numBlanks = 0
          laserExcerpt = [laser[i]]
          intensityExcerpt = [intensity[i]]
        else:
          laserExcerpt.append(laser[i])
          intensityExcerpt.append(intensity[i])
      else:
        if first != -1:
          laserExcerpt.append(laser[i])
          intensityExcerpt.append(intensity[i])
          numBlanks += 1
          if numBlanks >= 3:
            result.append([first,laserExcerpt[:-3],intensityExcerpt[:-3]])
            first = -1
    if first != -1:
      result.append([first,laserExcerpt,intensityExcerpt])
    return result
  return False

