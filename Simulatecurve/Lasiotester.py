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

print(units)
depth = las['DEPT']
tempGAMM = las['GAMM']

# Generate sample data with 2700 points
x = np.array(depth[:250])
y = np.array(tempGAMM[:250]) # Example data (noisy sine wave)

xreverse = x
yreverse = np.flip(y,0)
threshold = 100


#circularx = x % threshold
print(xreverse[0])
print(xreverse[:-1])
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(xreverse, yreverse, color='red', linewidth=1)
ax.plot(x,y,color='red', linewidth=1)

# Create the subplot


# Set axis limits
ax.set_xlim(10604.75, 10666.75)
ax.set_ylim(0, 200)

# Add labels, title, and grid
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Large Curve')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(['Curve reversed', 'Curve normal'])

# Display the plot
#ax.savefig('../testfolder/testplot.tif', format='tif', dpi=300)
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