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
def image_halftoning(delay):
  command = " ".join([programs[0], delay])
  print command
  a = os.system(command)
  print a

# ===============================================================================
# Example datasets
delay = '0' #ms

# Go!
image_halftoning(delay)







