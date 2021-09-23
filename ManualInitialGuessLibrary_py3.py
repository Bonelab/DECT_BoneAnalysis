#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:27:32 2018

@author: chantaldebakker
"""

import vtk
import numpy as np

def GetPoints(filePath, fixedfnm, movingfnm, fixed_type, moving_type):
  while True:
    print("Navigate 2D slices and pick 4 points in each.")

    ReadMha(filePath,fixedfnm, movingfnm)
    Navigate2D(fixed_type, moving_type)

    print("When done, close out of window and type q")
    optionUser = input("?: ")

    if (optionUser == "q"):
      fixed_pts = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
      moving_pts = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
      fixed_pts[0] = points_picked[0]
      fixed_pts[0][1] = fixed.GetDimensions()[1] - fixed_pts[0][1]
      moving_pts[0] = points_picked[1]
      moving_pts[0][0] = moving_pts[0][0] - fixed.GetDimensions()[0]
      moving_pts[0][1] = moving.GetDimensions()[1] - moving_pts[0][1]
      fixed_pts[1] = points_picked[2]
      fixed_pts[1][1] = fixed.GetDimensions()[1] - fixed_pts[1][1]
      moving_pts[1] = points_picked[3]
      moving_pts[1][0] = moving_pts[1][0] - fixed.GetDimensions()[0]
      moving_pts[1][1] = moving.GetDimensions()[1] - moving_pts[1][1]
      fixed_pts[2] = points_picked[4]
      fixed_pts[2][1] = fixed.GetDimensions()[1] - fixed_pts[2][1]
      moving_pts[2] = points_picked[5]
      moving_pts[2][0] = moving_pts[2][0] - fixed.GetDimensions()[0]
      moving_pts[2][1] = moving.GetDimensions()[1] - moving_pts[2][1]
      fixed_pts[3] = points_picked[6]
      fixed_pts[3][1] = fixed.GetDimensions()[1] - fixed_pts[3][1]
      moving_pts[3] = points_picked[7]
      moving_pts[3][0] = moving_pts[3][0] - fixed.GetDimensions()[0]
      moving_pts[3][1] = moving.GetDimensions()[1] - moving_pts[3][1]

      #Convert to absolute values by adding in offset and converting from voxels to absolute distance:
      fixed_pts[0][0] = fixed_pts[0][0]*fixed.GetSpacing()[0]
      fixed_pts[1][0] = fixed_pts[1][0]*fixed.GetSpacing()[0]
      fixed_pts[2][0] = fixed_pts[2][0]*fixed.GetSpacing()[0]
      fixed_pts[3][0] = fixed_pts[3][0]*fixed.GetSpacing()[0]
      fixed_pts[0][1] = fixed_pts[0][1]*fixed.GetSpacing()[1]
      fixed_pts[1][1] = fixed_pts[1][1]*fixed.GetSpacing()[1]
      fixed_pts[2][1] = fixed_pts[2][1]*fixed.GetSpacing()[1]
      fixed_pts[3][1] = fixed_pts[3][1]*fixed.GetSpacing()[1]
      fixed_pts[0][2] = fixed_pts[0][2]*fixed.GetSpacing()[2]
      fixed_pts[1][2] = fixed_pts[1][2]*fixed.GetSpacing()[2]
      fixed_pts[2][2] = fixed_pts[2][2]*fixed.GetSpacing()[2]
      fixed_pts[3][2] = fixed_pts[3][2]*fixed.GetSpacing()[2]

      moving_pts[0][0] = moving_pts[0][0]*moving.GetSpacing()[0]
      moving_pts[1][0] = moving_pts[1][0]*moving.GetSpacing()[0]
      moving_pts[2][0] = moving_pts[2][0]*moving.GetSpacing()[0]
      moving_pts[3][0] = moving_pts[3][0]*moving.GetSpacing()[0]
      moving_pts[0][1] = moving_pts[0][1]*moving.GetSpacing()[1]
      moving_pts[1][1] = moving_pts[1][1]*moving.GetSpacing()[1]
      moving_pts[2][1] = moving_pts[2][1]*moving.GetSpacing()[1]
      moving_pts[3][1] = moving_pts[3][1]*moving.GetSpacing()[1]
      moving_pts[0][2] = moving_pts[0][2]*moving.GetSpacing()[2]
      moving_pts[1][2] = moving_pts[1][2]*moving.GetSpacing()[2]
      moving_pts[2][2] = moving_pts[2][2]*moving.GetSpacing()[2]
      moving_pts[3][2] = moving_pts[3][2]*moving.GetSpacing()[2]

      return fixed_pts.tolist(), moving_pts.tolist()
      break





def ReadMha(filePath, fixed_fnm, moving_fnm):
  print("---------------------------Reading image---------------------------")

  # Define a mha reader from vtk.
  MhaReader = vtk.vtkMetaImageReader()

  #Read the path that the user will input.
  fnm1 = filePath+'/'+fixed_fnm+'.mha'
  MhaReader.SetFileName(fnm1)
  print(fnm1)
  MhaReader.Update()

  #Store the image in a global variable, so that it can be accesed later in any part of the code.
  global fixed
  fixed = MhaReader.GetOutput()
  print(fixed.GetDimensions())


  # Define a mha reader from vtk.
  MhaReader2 = vtk.vtkMetaImageReader()

 #Read the path that the user will input.
  fnm2 = filePath+'/'+moving_fnm+'.mha'
  print(fnm2)
  MhaReader2.SetFileName(fnm2)
  MhaReader2.Update()

  #Store the image in a global variable, so that it can be accesed later in any part of the code.
  global moving
  moving = MhaReader2.GetOutput()
  print(moving.GetDimensions())

# This function is used to define the image orientation as axial, coronal, or sagittal.
# Note how it receives two parameters: imageToOrient, and orientation
def orientImage (imageToOrient, orientation):
  (xMin, xMax, yMin, yMax, zMin, zMax) = imageToOrient.GetExtent()
  (xSpacing, ySpacing, zSpacing) = imageToOrient.GetSpacing()
  (x0, y0, z0) = imageToOrient.GetOrigin()

  # Calculate the coordinates of the center of the image.
  center = [x0 + xSpacing * (xMin + xMax),
            y0 + ySpacing * (yMin + yMax),
            z0 + zSpacing * (zMin + zMax)]

  #This matrices are used to transform the coordinates of the image, for each orientation.
  axial = vtk.vtkMatrix4x4()
  axial.DeepCopy((1, 0, 0,center[0] ,
                  0, -1, 0,center[1] ,
                  0, 0, 1,center[2],
                  0, 0, 0, 1))

  coronal = vtk.vtkMatrix4x4()
  coronal.DeepCopy((1, 0, 0, center[0],
                    0, 0, 1, center[1],
                    0, -1, 0, center[2],
                    0, 0, 0, 1))

  sagittal = vtk.vtkMatrix4x4()
  sagittal.DeepCopy((0, 0,-1, center[0],
                     1, 0, 0, center[1],
                     0, -1, 0,center[2] ,
                    0, 0, 0, 1))

  flipimage = vtk.vtkMatrix4x4()
  flipimage.DeepCopy((-1, 0, 0, center[0],
                      0, -1, 0, center[1],
                      0, 0, -1, center[2],
                      0, 0, 0, 1))

  # vtkImageReslice orients an image, given a transformation matrix.
  reslice = vtk.vtkImageReslice()
  reslice.SetInputData(imageToOrient)
  reslice.SetOutputDimensionality(3)
  reslice.SetResliceAxesOrigin(0,0,0)

  if orientation=='axial':
    reslice.SetResliceAxes(axial)
  elif orientation=='coronal':
    reslice.SetResliceAxes(coronal)
  elif orientation=='sagittal':
    reslice.SetResliceAxes(sagittal)
  elif orientation=='flipimage':
    reslice.SetResliceAxes(flipimage)

  # interpolation is required to re-slice the image in the required orientation
  reslice.SetInterpolationModeToCubic()
  reslice.Update()

  return reslice.GetOutput()

# This is the main navigation function
def Navigate2D(fixed_type, moving_type):
  print("---------------------------Navigating 2D---------------------------")

  global orientation_fixed
  orientation_fixed = 'axial'
  global orientation_moving
  orientation_moving = 'axial'

#  if (fixed_type == 'CT'):
#      orientation_fixed = 'axial'
#  elif (fixed_type == 'MR'):
#      orientation_fixed = 'coronal'
#  global orientation_moving
#  if (moving_type == 'CT'):
#      orientation_moving = 'axial'
#  elif (moving_type == 'MR'):
#      orientation_moving = 'coronal'
  global zSlice_fixed
  global zSlice_moving
  global clickCount
  clickCount = 0
  zSlice_fixed = 150
  zSlice_moving = 150

#  if (optionUser == "1"):
#    orientation = "axial"
#  if (optionUser == "2"):
#    orientation = "coronal"
#  if (optionUser == "3"):
#    orientation = "sagittal"

#  orientation = "axial"

  #display fixed image:
  fixed_imageNavigate2D = orientImage(fixed, orientation_fixed)

  global mapperNavigate2D
  mapperNavigate2D = vtk.vtkImageMapper()
  mapperNavigate2D.SetInputData(fixed_imageNavigate2D)
  mapperNavigate2D.SetZSlice(zSlice_fixed)
  # window and level have to be adjusted for better visualization
  if (fixed_type == 'MR'):
      # mapperNavigate2D.SetColorWindow(623)
      mapperNavigate2D.SetColorWindow(200)
      mapperNavigate2D.SetColorLevel(100)

  elif (fixed_type == 'CT'):
      # mapperNavigate2D.SetColorWindow(12900)
      mapperNavigate2D.SetColorWindow(6000)
      mapperNavigate2D.SetColorLevel(0)

  actorNavigate2D = vtk.vtkActor2D()
  actorNavigate2D.SetMapper(mapperNavigate2D)

  global renderWindowNavigate2D
  rendererNavigate2D = vtk.vtkRenderer()
  rendererNavigate2D.AddActor(actorNavigate2D)

  renderWindowNavigate2D = vtk.vtkRenderWindow()
  renderWindowNavigate2D.AddRenderer(rendererNavigate2D)
  rendererNavigate2D.SetViewport(0,0,0.5,1)


#  del renderWindowNavigate2D, windowInteractorN2D

  #display moving image:
  moving_imageNavigate2D = orientImage(moving, orientation_moving)
#  if (fixed_type == 'MR'):
  #    if (moving_type == 'CT'):
  #        moving_imageNavigate2D = orientImage(moving, 'flipimage')
#      else:
#          moving_imageNavigate2D = orientImage(moving, orientation_moving)
#  else:
#      moving_imageNavigate2D = orientImage(moving, orientation_moving)

  global moving_mapperNavigate2D
  moving_mapperNavigate2D = vtk.vtkImageMapper()
  moving_mapperNavigate2D.SetInputData(moving_imageNavigate2D)
  moving_mapperNavigate2D.SetZSlice(zSlice_moving)

  # window and level have to be adjusted for better visualization
  # moving_mapperNavigate2D.SetColorWindow(200)
  # moving_mapperNavigate2D.SetColorLevel(100)
  if (moving_type == 'MR'):
      # mapperNavigate2D.SetColorWindow(623)
      moving_mapperNavigate2D.SetColorWindow(600)
      moving_mapperNavigate2D.SetColorLevel(300)
  elif(moving_type == 'HR'):
      moving_mapperNavigate2D.SetColorWindow(9000)
      moving_mapperNavigate2D.SetColorLevel(3000)


  moving_actorNavigate2D = vtk.vtkActor2D()
  moving_actorNavigate2D.SetMapper(moving_mapperNavigate2D)

  moving_rendererNavigate2D = vtk.vtkRenderer()
  moving_rendererNavigate2D.AddActor(moving_actorNavigate2D)

  renderWindowNavigate2D.AddRenderer(moving_rendererNavigate2D)
  moving_rendererNavigate2D.SetViewport(0.5,0,1,1)

  renderWindowNavigate2D.SetSize(fixed.GetDimensions()[0]+moving.GetDimensions()[0], max(fixed.GetDimensions()[1], moving.GetDimensions()[1]))


  windowInteractorN2D = vtk.vtkRenderWindowInteractor()
  windowInteractorN2D.SetRenderWindow(renderWindowNavigate2D)

  #Note here how the event is added to the observer. The name of the event is fixed, the name
  #of the function that handles the event is user defined.
  windowInteractorN2D.AddObserver("KeyPressEvent", KeyPressNavigate2D, 1.0)
  # windowInteractorN2D.AddObserver("CharEvent", KeyPressNavigate2D, 1.0)


  #initialize points list:
  global points_picked
  points_picked = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

  #look for mouse clicks to select points to initialize registration
  windowInteractorN2D.AddObserver("RightButtonPressEvent", PickPoints, 1.0)


  windowInteractorN2D.Initialize()
  renderWindowNavigate2D.Render()
  windowInteractorN2D.Start()

  del renderWindowNavigate2D, windowInteractorN2D

def KeyPressNavigate2D(obj, Event):
  key = obj.GetKeySym()
  limitUp_fixed = 0
  limitDown_fixed = 0
  limitUp_moving = 0
  limitDown_moving = 0

  limitUp_fixed = fixed.GetDimensions()[2] - 1
  limitUp_moving = moving.GetDimensions()[2] - 1

#  if (orientation_fixed == "axial"):
#    limitUp_fixed = fixed.GetDimensions()[2] - 1
#  if (orientation_moving == "axial"):
#    limitUp_moving = moving.GetDimensions()[2] - 1
#  if (orientation_fixed == "coronal"):
#    limitUp_fixed = fixed.GetDimensions()[1] - 1
#  if (orientation_moving == "coronal"):
#    limitUp_moving = moving.GetDimensions()[1] - 1
#  if (orientation_fixed == "sagittal"):
#    limitUp_fixed = fixed.GetDimensions()[0] - 1
#  if (orientation_moving == "sagittal"):
#    limitUp_moving = moving.GetDimensions()[0] - 1

  global zSlice_fixed
  global zSlice_moving

  if (key == "Up"):
    zSlice_fixed = zSlice_fixed + 1
    if (zSlice_fixed > limitUp_fixed):
      zSlice_fixed = limitUp_fixed

  if (key == "Down"):
    zSlice_fixed = zSlice_fixed - 1
    if (zSlice_fixed < 0):
      zSlice_fixed = 0

  if (key == "Right"):
    zSlice_moving = zSlice_moving + 1
    if (zSlice_moving > limitUp_moving):
      zSlice_moving = limitUp_moving

  if (key == "Left"):
    zSlice_moving = zSlice_moving - 1
    if (zSlice_moving < 0):
      zSlice_moving = 0

  mapperNavigate2D.SetZSlice(zSlice_fixed)
  moving_mapperNavigate2D.SetZSlice(zSlice_moving)
  renderWindowNavigate2D.Render()

def PickPoints(obj, Event):
  global clickCount
  [xpos,ypos] = obj.GetEventPosition()
  if clickCount == 0 or clickCount == 2 or clickCount == 4 or clickCount == 6:
      zpos = zSlice_fixed
  elif clickCount == 1 or clickCount == 3 or clickCount == 5 or clickCount == 7:
      zpos = zSlice_moving
  print("mouse clicked: ", (xpos, ypos, zpos), clickCount)
  global points_picked
  points_picked[clickCount] = [xpos, ypos, zpos]
  clickCount = clickCount + 1
