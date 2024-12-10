import vtk
import time

def visualize_objs_with_vtk(file1, file2, output_screenshot=None):
    """
    Visualize two OBJ files using VTK with distinct colors and black background.
    Apply a 90-degree rotation on the Y-axis to the red object.
    
    Parameters:
    - file1 (str): Path to the first OBJ file.
    - file2 (str): Path to the second OBJ file.
    - output_screenshot (str): Path to save the screenshot. If None, no screenshot is saved.
    """
    # Load first OBJ file
    reader1 = vtk.vtkOBJReader()
    reader1.SetFileName(file1)
    reader1.Update()

    # Load second OBJ file
    reader2 = vtk.vtkOBJReader()
    reader2.SetFileName(file2)
    reader2.Update()

    # Create mappers for the meshes
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputConnection(reader1.GetOutputPort())

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputConnection(reader2.GetOutputPort())

    # Create actors for the meshes
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetColor(1, 0, 0)  # Red
    actor1.GetProperty().SetOpacity(0.8)

    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetColor(0, 0, 1)  # Blue
    actor2.GetProperty().SetOpacity(1)

    legend_text_red = vtk.vtkTextActor()
    legend_text_red.SetInput("Oryginał  -  ")
    legend_text_red.GetTextProperty().SetFontSize(25)
    legend_text_red.SetPosition(10, 70)

    legend_text_blue = vtk.vtkTextActor()
    legend_text_blue.SetInput("Predykcja  -  ")
    legend_text_blue.GetTextProperty().SetFontSize(25)
    legend_text_blue.SetPosition(10, 35)

    legend_background = vtk.vtkTexturedActor2D()
    background_image = vtk.vtkImageCanvasSource2D()
    background_image.SetExtent(0, 200, 0, 100, 0, 0)
    background_image.SetDrawColor(0.5, 0.5, 0.5, 1)  # Grey color
    background_image.FillBox(0, 200, 0, 100)
    background_image.Update()

    background_mapper = vtk.vtkImageMapper()
    background_mapper.SetInputConnection(background_image.GetOutputPort())

    legend_background.SetMapper(background_mapper)
    legend_background.GetPositionCoordinate().SetValue(10, 10)  # Bottom-left corner of the legend
    legend_background.SetPosition2(0.2, 0.1)  # Adjust size of the rectangle

    # Create a renderer and set the background to black
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1024, 1024)

    # Create an interactor for user interaction
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Start the visualization
    render_window.Render()
    if output_screenshot:
        # Capture a screenshot
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(output_screenshot)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()
        print(f"Screenshot saved to {output_screenshot}")

    interactor.Start()

# File paths
file1 = r"C:\Users\Bartek\Desktop\mgr\stl\nower\alligned\untitled.obj"
file2 = r"C:\Users\Bartek\Desktop\mgr\stl\nower\alligned\untitled1.obj"

# Visualize the OBJ files with a 10-second rotation animation
visualize_objs_with_vtk(file1, file2)
