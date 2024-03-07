import os, csv

f=open("C:/Users/tobia/Documents/Github/Bachelor/pothole_images/validation/validation.csv",'r+')
w=csv.writer(f)
for path, dirs, files in os.walk("C:/Users/tobia/Documents/Github/Bachelor/pothole_images/validation/images"):
    for filename in files:
        w.writerow([filename])

