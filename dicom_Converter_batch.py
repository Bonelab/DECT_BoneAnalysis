import vtk
import vtkbone
import os
import argparse


#Convert DICOM to MHA and AIM
# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("filePath",
                    type=str,
                    help="The filepath")
parser.add_argument("outputImage",
                    type=str,
                    help="The output Mha image")
args = parser.parse_args()

filePath = args.filePath
outputImage = args.outputImage

dicomReader = vtk.vtkDICOMImageReader()
dicomReader.SetDirectoryName(filePath)
dicomReader.Update()
dicomImage = dicomReader.GetOutput()

mhaWriter = vtk.vtkMetaImageWriter()
# mhaWriter.SetDirectoryName(filePath)
mhaWriter.SetFileName(filePath+'/'+outputImage+'.mha')
mhaWriter.SetInputData(dicomImage)
mhaWriter.Write()

#aimWriter = vtkbone.vtkboneAIMWriter()
#aimWriter.SetInputData(dicomImage)
#aimWriter.SetFileName("summed_image.aim")
#aimWriter.Update()
