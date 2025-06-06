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



x = np.array(depth[:125])
y = np.array(tempGAMM[:125]) 
x2 = np.array (depth[135:])
y2 = np.array(tempGAMM[135:])

xreverse = x
yreverse = np.flip(y,0)
threshold = 100


#circularx = x % threshold
print(xreverse[0])
print(xreverse[:-1])
fig, ax = plt.subplots(figsize=(20,10))
#ax.plot(xreverse, yreverse, color='red', linewidth=1)
ax.set_xlim(10604.75, 10666.75)
ax.set_ylim(0, 200)
ax.plot(x,y,color='blue', linewidth=1)
ax.plot(x2,y2,color='blue', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.grid(True)
#ax.legend(['Curve reversed', 'Curve normal'])

# Display the plot
plt.savefig('Unconnectedcurve/test2.tif', dpi=200, format='tiff', bbox_inches='tight', pad_inches=0)

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
filtered_y = y[x < threshold] """

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