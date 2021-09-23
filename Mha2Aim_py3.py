"""Converts MHA to AIM files.

Description
  converts Mhd/Mha file to Aim

History:
  2016.07.06  Bryce Besler    Created
  2016.08.18  Andres Kroker    Adjust skalar type to be SCANCO compatible

Notes:
  - None
"""

# Imports
import vtk
import vtkbone
import argparse

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("inputImage",
                    type=str,
                    help="The input mha image")
parser.add_argument("outputImage",
                    type=str,
                    help="The output AIM image")
args = parser.parse_args()

inputImage = args.inputImage
outputImage = args.outputImage

# Read in the image
print("Reading {0}".format(inputImage))
reader = vtk.vtkMetaImageReader()
reader.SetFileName(inputImage)
reader.Update()


inputScalarType = reader.GetOutput().GetScalarType()
if (inputScalarType == vtk.VTK_BIT or
    inputScalarType == vtk.VTK_CHAR or
    inputScalarType == vtk.VTK_SIGNED_CHAR or
    inputScalarType == vtk.VTK_UNSIGNED_CHAR):


    scalarRange = reader.GetOutput().GetScalarRange()
    if scalarRange[0] >= vtk.VTK_SHORT_MIN and scalarRange[1] <= vtk.VTK_SHORT_MAX:
        outputScalarType = vtk.VTK_CHAR
    else:
        outputScalarType = vtk.VTK_CHAR
else:
    outputScalarType = vtk.VTK_SHORT

# Cast
print("Converting to {0}".format(vtk.vtkImageScalarTypeNameMacro(outputScalarType)))
caster = vtk.vtkImageCast()
caster.SetOutputScalarType(outputScalarType)
caster.SetInputConnection(reader.GetOutputPort())
caster.ReleaseDataFlagOff()
caster.Update()

# Write the image out
print("Writing to {0}".format(outputImage))
writer = vtkbone.vtkboneAIMWriter()
writer.SetFileName(outputImage)
writer.SetInputData(caster.GetOutput())
writer.NewProcessingLogOn()
writer.Write()
