#!/bin/bash

ffmpeg -r 10 -i frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p "$1.mp4"
