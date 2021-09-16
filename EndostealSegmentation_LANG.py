
# Imports
import SimpleITK as sitk
import argparse
import os

# Argument Parsing
parser = argparse.ArgumentParser(description='''
Endosteal segmentation of a QCT image using the method of Lang et al.

The input image is assumed to be in mg/cc HA equivalent densities.

If there are multiple objects in the segmentation, all are evaluated.

Lang, T. F., et al. "Volumetric quantitative computed tomography of the proximal femur: precision and relation to bone strength." Bone 21.1 (1997): 101-108.
''')
parser.add_argument("model_dir", type=str, help="The filepath")
parser.add_argument('--input_name', default='Calibrated_StandardSEQCT', help='Input name without extension (QCT_SEG_EX_0001)')
parser.add_argument('--erosion_distance', default=3.00, help='Trabecular erosion distance')
parser.add_argument('--trab_threshold', default=600.0, help='Trabecular upper threshold in mg/cc HA')
parser.add_argument('--cort_threshold', default=400.0, help='Cortical lower threshold in mg/cc HA')
parser.add_argument('--periosteal_ending', default='_SEG.mha', help='Postfix for periosteal image')
parser.add_argument('--endosteal_ending', default='_LANG.mha', help='Postfix for endosteal image')

args = parser.parse_args()


# Check inputs
print('Arguments:')
for arg in vars(args):
  print('  {}: {}'.format(arg, getattr(args, arg)))
print('')

filePath = args.model_dir
a = os.path.split(filePath)
b = len(a)
participant_id = a[b-1]

density_filename = os.path.join(args.model_dir,args.input_name+'.mha')
periosteal_filename = os.path.join(args.model_dir, args.input_name + args.periosteal_ending)
endosteal_filename = os.path.join(args.model_dir, args.input_name+args.endosteal_ending)

print('Reading density image ' + density_filename)
den = sitk.ReadImage(density_filename)

print('Reading periosteal segmentation ' + periosteal_filename)
peri = sitk.ReadImage(periosteal_filename)

print('Finding unique labels')
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(peri)
unique_labels = [int(x) for x in stats.GetLabels()]
print('  {}'.format(unique_labels))

if len(unique_labels) <= 0:
  os.sys.exit('Input image has no labels')

print('Checking input arguments:')
radius = [int(round(float(args.erosion_distance)/x)) for x in peri.GetSpacing()]
trab_thresh = float(args.trab_threshold)
cort_thresh = float(args.cort_threshold)
print('  Radius:               {}'.format(radius))
print('  Trabecular Threshold: {}'.format(trab_thresh))
print('  Cortical Threshold:   {}'.format(cort_thresh))
print('')

endo = None
new_label = 1
for label in unique_labels:
  print('Processing label {}'.format(label))
  this = peri==label

  #Close any holes in periosteal segmentation:
  filler = sitk.BinaryFillholeImageFilter()
  this_close = filler.Execute(this)

  # Trab
  erod = sitk.BinaryErode(
    this_close,
    radius,
    sitk.sitkBall,
    0.0,
    1.0,
    True
  )
  trab = sitk.Mask(den<trab_thresh, erod)

  # Cort
  cort = sitk.Mask(den>=cort_thresh, this)


  #for edema imaging, we only need the bone marrow region:
  result = new_label*trab

  # Add together in a single image
  if endo is None:
    endo = result
  else:
    endo = endo + result

  # Update
  # new_label += 2
  new_label += 1

print('Writing result to ' + endosteal_filename)
sitk.WriteImage(endo, endosteal_filename,True)

print('Finished!')
