"""Converts AIMs to MHA files.

Description
  converts Aim file to Mhd/Mha

History:
  2016.08.18  Andres Kroker    Created

Notes:
  - None
"""

# Imports
import vtk
import vtkbone
import argparse
import os

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("inputImage",
                    type=str,
                    help="The input AIM image")
parser.add_argument("outputImage",
                    type=str,
                    help="The output Mhd image")
args = parser.parse_args()

inputImage = args.inputImage
outputImage = args.outputImage

# extract directory, filename, basename, and extensions
directory, filename = os.path.split(outputImage)
basename, extension = os.path.splitext(filename)

# check if raw file needed
if extension.lower() == ".mha":
    print("no .raw file will be created as info integrated in image file.")
elif extension.lower() == ".mhd":
    outputImageRaw = os.path.join(directory, basename + ".raw")
else:
    print("output file extension must be .mhd or .mha. Setting it to .mha")
    outputImage = os.path.join(directory, basename + ".mha")
    extension = ".mha"

# Read in the image
print("Reading {0}".format(inputImage))
reader = vtkbone.vtkboneAIMReader()
reader.SetFileName(inputImage)
reader.DataOnCellsOff()
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
        outputScalarType = vtk.VTK_SHORT
else:
    outputScalarType = vtk.VTK_SHORT
#outputScalarType = vtk.VTK_SHORT

# Cast
print("Converting to {0}".format(vtk.vtkImageScalarTypeNameMacro(outputScalarType)))
caster = vtk.vtkImageCast()
caster.SetOutputScalarType(outputScalarType)
caster.SetInputConnection(reader.GetOutputPort())
caster.ReleaseDataFlagOff()
caster.Update()

# Write the image out
if extension.lower() == '.mha':
    print("Writing to {0}".format(outputImage))
    writer = vtk.vtkMetaImageWriter()
    writer.SetInputConnection(caster.GetOutputPort())
    writer.SetFileName(outputImage)
    writer.Write()
else:
    print("Writing to {0}".format(outputImage))
    writer = vtk.vtkMetaImageWriter()
    writer.SetInputConnection(caster.GetOutputPort())
    writer.SetFileName(outputImage)
    writer.SetRAWFileName(outputImageRaw)
    writer.Write()
