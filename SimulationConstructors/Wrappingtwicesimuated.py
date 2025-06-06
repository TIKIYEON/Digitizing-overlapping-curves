#Lasiotester
import matplotlib.pyplot as plt
import numpy as np

#lasfn = "../T14502Las/T14502_02-Feb-07_JewelryLog.las"
lasfn = "T14502Las/T14502_02-Feb-07_JewelryLog.las"
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

y1 = y[:105]
x1 = x[:105]
y2 = y[106:112]
x2 = x[106:112]

x1diffx2 = (x2[0].copy() - x1[-1].copy())/3
x1 = np.append(x1,((x1[-1]+x1diffx2)))
y1 = np.append(y1,99.5)

x2 = np.append((x2[0]-x1diffx2),x2)
y2 = np.append(0.05,y2)


y3 = y[113:120]
x3 = x[113:120]
x2diffx3 = (x3[0].copy() - x2[-1].copy())/3
x2 = np.append(x2, ((x2[-1]+x2diffx3)))
y2 = np.append(y2, 0.05)

""" x3 = np.append((x3[0]-x2diffx3),x3)
y3 = np.append(0.05,y3) """

y4 = y[121:]
x4 = x[121:]

x3diffx4 = (x4[0].copy() - x3[-1].copy())/3

x3 = np.append(x3,((x3[-1]+x3diffx4)))
y3 = np.append(y3,99.5)

x4 = np.append((x4[0]-x3diffx4),x4)
y4 = np.append(0.05,y4)

interpolating = 50
""" y1 = y[:105]
x1 = x[:105]
x1last = x1[-1]
y1last = y1[-1]
y1end = 100.0

y2 = y[106:112]
x2 = x[106:112]
difference = x2[0].copy() - x1[-1].copy()
x1end = x1[-1] + difference/2 
xarray1 = np.linspace(x1last,x1end, interpolating)
yarray1 = np.linspace(y1last, y1end, interpolating)
y1 = np.append(y1, yarray1)
x1 = np.append(x1,xarray1)

x2start = x2[0]
x2end = x2[0]
y2start = 0.0
y2end = y2[0]

xarray2 = np.linspace(x2start,x2end, 25)
yarray2 = np.linspace(y2start, y2end, 25)
y2 = np.append(yarray2, y2)
x2 = np.append(xarray2, x2)


y3 = y[113:120]
x3 = x[113:120]
x23start = x2[-1]
x23end = x3[0]
y23start = y2[-1]
y23end = 0.0
difference = x3[0].copy() - x2[-1].copy()
x23end = x2[-1] + difference/2 

xarray23 = np.linspace(x23start,x23end, 25)
yarray23 = np.linspace(y23start, y23end, 25)
y2 = np.append(y2, yarray23)
x2 = np.append(x2, xarray23)

x32start = x[-1]
x32end = x3[0]
y32start = 100.0
y32end = y3[0]
xarray32 = np.linspace(x32start, x32end, 25)
yarray32 = np.linspace(y32start, y32end, 25)

y3 = np.append(yarray32, y3)
x3 = np.append(xarray32, x3)

y4 = y[121:]
x4 = x[121:]
x34start = x3[-1]
x34end = x4[0]
y34start = y3[-1]
y34end = 100.0

xarray34 = np.linspace(x34start,x34end, 25)
yarray34 = np.linspace(y34start, y34end, 25)
y3 = np.append(y3, yarray34)
x3 = np.append(x3, xarray34)

x43start = x3[-1]
x43end = x4[0]
y43start = 0.0
y43end = y4[0]
xarray43 = np.linspace(x43start, x43end, 25)
yarray43 = np.linspace(y43start, y43end, 25)

y4 = np.append(yarray43, y4)
x4 = np.append(xarray43, x4) """
""" x1 = x1[:-5] """
#x2 = x2[12:]
""" x2 = x2[:-3]
x3 = x3[3:]
x3 = x3[:-3]
x4 = x4[3:]
y1 = y1[:-5] """
#y2 = y2[12:]
""" y2 = y2[:-3]
y3 = y3[3:]
y3 = y3[:-3]
y4 = y4[3:] """
y22emd = 100.0

xreverse = x
yreverse = np.flip(y,0)
threshold = 100


#circularx = x % threshold
print(xreverse[0])
print(xreverse[:-1])
fig, ax = plt.subplots(figsize=(5,3))
#ax.plot(xreverse, yreverse, color='red', linewidth=1)
ax.set_xlim(10629.75, x4[-1])
ax.set_ylim(0, 100)
ax.plot(x1,y1,color='blue', linewidth=1)
ax.plot(x2,y2,color='blue', linewidth=1)
ax.plot(x3,y3,color='blue', linewidth=1)
ax.plot(x4,y4,color='blue', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.grid(True)
#ax.legend(['Curve reversed', 'Curve normal'])

# Display the plot
plt.savefig('wrapping/twowrap2.tif', dpi=200, format='tiff', bbox_inches='tight', pad_inches=0)
print(x4[-1])
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
filtered_x = x[x < threshold]
filtered_y = y[x < threshold]

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