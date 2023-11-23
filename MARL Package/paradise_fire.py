import vtk
import numpy as np
import matplotlib.pyplot as plt

vtk_file_path = '../../paradise_data/change_64x64x128_0000000100.vtk'

# Create a generic data object reader
reader = vtk.vtkGenericDataObjectReader()
reader.SetFileName(vtk_file_path)

# Try updating the reader
try:
    reader.Update()
except RuntimeError as e:
    print(f"Error reading VTK file: {str(e)}")
    exit()

# Get the output data object
data_object = reader.GetOutput()

# Check if the data object is loaded correctly
if data_object is None:
    print("Error: Data object is None. Check if the file contains valid VTK data.")
    exit()

# Get information about the data object
class_name = data_object.GetClassName()
print("Class Name:", class_name)
    

# # Read the .vtk file
# reader = vtk.vtkArrayReader()
# reader.SetFileName(vtk_file_path)
# reader.Update()
# # print('reader updated', reader)
# # Get the data set from the reader
# data_set = reader.GetOutput()

# # Get the number of points in the data set
# num_points = data_set.GetNumberOfPoints()

# # Get the point data from the data set
# point_data = data_set.GetPointData()

# print(point_data)

# if point_data:
#     # Get arrays for wind velocity (u, v, w) and temperature (t)
#     u_array = point_data.GetArray("u")
#     v_array = point_data.GetArray("v")
#     w_array = point_data.GetArray("w")
#     t_array = point_data.GetArray("t")

#     if u_array and v_array and w_array and t_array:
#         for point_id in range(num_points):
#             # Get coordinate point (i, j, k) for each point
#             point = data_set.GetPoint(point_id)
#             i, j, k = point

#             # Extract data at the point
#             u = u_array.GetTuple1(point_id)
#             v = v_array.GetTuple1(point_id)
#             w = w_array.GetTuple1(point_id)
#             t = t_array.GetTuple1(point_id)

#             print(f"Point ({i}, {j}, {k}) - Wind (u, v, w): ({u}, {v}, {w}), Temperature (t): {t}")
#     else:
#         print("Missing required arrays (u, v, w, t) in the VTK file.")

# else:
#     print("No point data found in the VTK file.")

# print("\n")

