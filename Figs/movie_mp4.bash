ffmpeg -framerate 10 -i vortex_detection_DeepVortex_%03d.png -vf scale=2270:2060 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p vortex_detection_DeepVortex.mp4

