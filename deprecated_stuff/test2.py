import os

gt = os.listdir("C:\\Users\\GPSERS-DEXTER-04\\Downloads\\cataovo-annotations\\images")
dt = os.listdir("C:\\Users\\GPSERS-DEXTER-04\\Downloads\\cataovo-annotations\\Ground-Truth-Evaluation\\evaluation")

diff_files = set(dt) - set(gt)

print(diff_files)