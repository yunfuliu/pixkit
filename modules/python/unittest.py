import sys
import os

# ===============================================================================
# add programs
programs = ['test_image_halftoning']
for i in range(len(programs)):
  programs[i] = '.\\' + programs[i]
if sys.platform == 'win32':
  for i in range(len(programs)):
    programs[i] = programs[i] + '.exe'

# ===============================================================================
# Helper functions
def test_func_01(imagePath):
  command = " ".join([programs[0], ' -i ', imagePath])
  print command
  a = os.system(command)
  print a

# ===============================================================================
# Example datasets
imagePath1 = '.\datasets\iguazu\img1.pgm'

# Go!
test_func_01(imagePath1)







