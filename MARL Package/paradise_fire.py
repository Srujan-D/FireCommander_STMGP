import vtk

# Create a reader for your .vtk file
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("../paradise_data/change_64x64x128_0000000100.vtk")
reader.Update()

ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

arrow = vtk.vtkArrowSource()
arrow.SetTipResolution(16)
arrow.SetTipLength(0.3)
arrow.SetTipRadius(0.1)

glyph = vtk.vtkGlyph3D()
glyph.SetSourceConnection(arrow.GetOutputPort())
glyph.SetInputConnection(reader.GetOutputPort())
glyph.SetVectorModeToUseVector()
glyph.SetScaleFactor(1.0)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(glyph.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

ren.AddActor(actor)

ren.SetBackground(0.1, 0.1, 0.1)  # Set background color
renWin.SetSize(800, 600)  # Set window size
ren.GetActiveCamera().Azimuth(30)  # Set camera position
ren.GetActiveCamera().Elevation(30)
ren.ResetCamera()

iren.Initialize()
renWin.Render()
iren.Start()
