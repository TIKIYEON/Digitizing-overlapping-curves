#Lasiotester
import matplotlib.pyplot as plt
import numpy as np

#lasfn = "../T14502Las/T14502_02-Feb-07_JewelryLog.las"
lasfn = "C:/Users/willi/OneDrive/Skrivebord/Bachelor/Github/Digitizing-overlapping-curves/T14502Las/T14502_02-Feb-07_JewelryLog.las"
import lasio
las = lasio.read(str(lasfn),ignore_header_errors=True)
#las = lasio.read(str(lasfn),encoding="cp866")
#las = lasio.read(str(lasfn),encoding="windows-1251")

headers=las.keys()
units = {}
for j in range(0, len(headers)):
     uval = las.curves[j].unit
     units[headers[j]] = uval

dataarr = las.data
metaheaders=las.well.keys()
metadata=[]
metadata.append({})
metadata.append({})
metadata.append({})


for j in range(0, len(metaheaders)):
     uval = las.well[j].unit
     metadata[0][metaheaders[j]] = uval

     tval = las.well[j].value
     metadata[1][metaheaders[j]] = str(tval)

     dval = las.well[j].descr
     metadata[2][metaheaders[j]] = str(dval)

print(metadata)
print(units)
depth = las['DEPT']
#y = np.flip(depth,0)
tempGAMM = las['GAMM']
x = 0
""" for i, k in enumerate(units):
     print(i,k)
     tempdic = las[k]
     x = np.array(tempdic)
     fig, ax = plt.subplots(figsize=(10,20))
     ax.plot(x, depth, color='red', linewidth=1)
     ax.invert_yaxis
     plt.show() """



x = np.array(depth[100:350])
y = np.array(tempGAMM[100:350]) 

for i in range(len(y)):
     if y[i] > 100:
          y[i] -= 100

interpolating = 50
y1 = y[:105]
x1 = x[:105]
print(x1[-1])
print(y1[-1])
x1last = x1[-1]
y1last = y1[-1]
y1end = 100.0
x1end = x[106]
xarray1 = np.linspace(x1last,x1end, interpolating)
yarray1 = np.linspace(y1last, y1end, interpolating)
y1 = np.append(y1, yarray1)
x1 = np.append(x1,xarray1)
y2 = y[106:112]
x2 = x[106:112]

x2start = x1[-1]
x2end = x2[0]
y2start = 0.0
y2end = y2[0]

xarray2 = np.linspace(x2start,x2end, 25)
yarray2 = np.linspace(y2start, y2end, 25)
y2 = np.append(yarray2, y2)
x2 = np.append(xarray2, x2)

""" y3 = y[113:120]
y4 = y[121:]
x3 = x[113:120]
x4 = x[121:] """
""" xreverse = x
yreverse = np.flip(y,0)
threshold = 100 """


#circularx = x % threshold
""" print(xreverse[0])
print(xreverse[:-1]) """
fig, ax = plt.subplots(figsize=(5,3))
#ax.plot(xreverse, yreverse, color='red', linewidth=1)
ax.set_xlim(10629.75, 10655.75)
ax.set_ylim(0, 100)
ax.plot(x1,y1,color='blue', linewidth=1)
#ax.plot(x2,y2,color='blue', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#ax.grid(True)
#ax.legend(['Curve reversed', 'Curve normal'])

# Display the plot
plt.savefig('Constructors/simplecurve.tif', dpi=200, format='tiff', bbox_inches='tight', pad_inches=0)

# Create the subplot


# Set axis limits


# Add labels, title, and grid
""" ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Large Curve') """

#iio.imwrite(ax)
plt.show()
# Define the threshold for y-values
#threshold = -26.0

# Filter the data based on the threshold
""" filtered_x = x[x < threshold]
filtered_y = y[x < threshold]
 """
""" # Generate more data points for a smooth curve using interpolation
y_smooth = np.linspace(min(filtered_y), max(filtered_y), 1000)
x_smooth = np.interp(y_smooth, filtered_x, filtered_x)

# Plot the results
plt.plot(x_smooth, y_smooth, label='Smooth Curve')
plt.scatter(filtered_x, filtered_y, color='red', s=10, label='Filtered Points', alpha=0.5)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Smooth Curve Plot with Thresholded Values') """

#plt.figure(figsize=(10,900))
#plt.plot(y,x)
#plt.plot(yreverse,xreverse)
#plt.gca().invert_yaxis()
#plt.show()