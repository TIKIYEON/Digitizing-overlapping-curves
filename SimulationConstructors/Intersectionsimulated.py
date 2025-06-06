import numpy as np
import matplotlib.pyplot as plt

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
x = np.array(depth[:250])
y = np.array(tempGAMM[:250]) 

def curve_function1(x):
    return -0.001*x**3 + 0.0042*x**2 + 0.11*x

def curve_function2(x):
    return 0.001*x**3 - 0.0042*x**2 - 0.11*x + 1

""" def curve_function1(x):
    return 0.1*x

def curve_function2(x):
    return -0.3*x + 1
 """
# Generate 500 points for x between 0 and 10
x_values1 = np.linspace(0, 10, 800)
y_values1 = curve_function1(x_values1)

x_values2 = np.linspace(0, 10, 800)
y_values2 = curve_function2(x_values2)
#plt.figure(figsize=(15,7))
# Plot the curve
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlim(0, 10.0)
ax.set_ylim(0.0, 1.0)
ax.plot(x_values1, y_values1, color='blue')
ax.plot(x_values2, y_values2, color='blue')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.grid(True)
#plt.savefig('NoscreenshotCurve2.tif', dpi=300, format='tiff', bbox_inches='tight', pad_inches=0)
plt.show()