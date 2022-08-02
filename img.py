# -*- coding = utf-8 -*-
# @Time : 20/06/2022 11:01
# @Author : Yan Zhu
# @File : img.py
# @Software : PyCharm

import sys
from PIL import Image

# list_im = ['image/EnSC1.png', 'image/EnSC2.png', 'image/EnSC3.png']
#
# images = [Image.open(x) for x in ['image/angle/EnSC1.png', 'image/angle/EnSC2.png', 'image/angle/EnSC3.png']]
# widths, heights = zip(*(i.size for i in images))
#
# total_width = sum(widths)
# max_height = max(heights)
#
# new_im = Image.new('RGB', (total_width, 5*max_height))
#
# x_offset = 0
# for im in images:
#   new_im.paste(im, (x_offset,0))
#   x_offset += im.size[0]
#
# x_offset = 0
# images = [Image.open(x) for x in ['image/angle/Kmeans1.png', 'image/angle/Kmeans2.png', 'image/angle/Kmeans3.png']]
# for im in images:
#   new_im.paste(im, (x_offset, 480))
#   x_offset += im.size[0]
#
# x_offset = 0
# images = [Image.open(x) for x in ['image/angle/LRR1.png', 'image/angle/LRR2.png', 'image/angle/LRR3.png']]
# for im in images:
#   new_im.paste(im, (x_offset, 960))
#   x_offset += im.size[0]
#
# x_offset = 0
# images = [Image.open(x) for x in ['image/angle/SpectralClustering1.png', 'image/angle/SpectralClustering2.png',
#                                   'image/angle/SpectralClustering3.png']]
# for im in images:
#   new_im.paste(im, (x_offset, 1440))
#   x_offset += im.size[0]
#
# x_offset = 0
# images = [Image.open(x) for x in ['image/angle/SSC-OMP1.png', 'image/angle/SSC-OMP2.png',
#                                   'image/angle/SSC-OMP3.png']]
# for im in images:
#   new_im.paste(im, (x_offset, 1920))
#   x_offset += im.size[0]
# new_im.save('Angle.jpg')


images = [Image.open(x) for x in ['image/yale/yale1.png', 'image/yale/yale2.png',
                                  'image/yale/yale3.png']]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset, 0))
  x_offset += im.size[0]

new_im.save('Yale.jpg')