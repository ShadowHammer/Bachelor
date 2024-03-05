import os, csv

f=open("C:\Users\tobia\OneDrive\Skrivebord\SDU\4. semester\EMP\workspace\Assignment2\Bachlor\pothole_images\validationvalidation.csv",'r+')
w=csv.writer(f)
for path, dirs, files in os.walk("C:\Users\tobia\OneDrive\Skrivebord\SDU\4. semester\EMP\workspace\Assignment2\Bachlor\pothole_images\validation/images"):
    for filename in files:
        w.writerow([filename])