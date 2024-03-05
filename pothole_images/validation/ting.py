import os, csv

f=open("C:/Users/tobia/Documents/GitHub/Bachlor/pothole_images/training/training.csv",'r+')
w=csv.writer(f)
for path, dirs, files in os.walk("C:/Users/tobia/Documents/GitHub/Bachlor/pothole_images/training/images"):
    for filename in files:
        w.writerow([filename])