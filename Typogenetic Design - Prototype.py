
# AI4AD - TYPOGENETIC DESIGN OF ARCHITECTURAL SHAPES
#  ________                                     ________         ________
# |        |        |        |        |        |        |                |
# |        |        |        |        |        |        |                |
# |        |        |        |________|        |        |                |
# |        |        |                 |        |        |                |
# |        |________|                 |        |        |________________|
#
# DIRECTED SEARCH USING INTERACTIVE AND MULTI-CRITERIA FITNESS EVALUATION

# Developed by Manuel Muehlbauer
# Under supervision of Marcelo Stamm, Jane Burry and Andy Song

# adopted code from pygp.py and gp.py

#-------------------------------------------- IMPORT STATEMENTS --------------------------------------------------------------#

import random
import sys
import math
import System.Drawing as SD
import Rhino
import rhinoscriptsyntax as rs
from copy import deepcopy
import clr
import os
import imp
clr.AddReference("Eto")
clr.AddReference("Rhino.UI")
clr.AddReference("MIConvexHull")
import MIConvexHull
import time
from Rhino.UI import *
from Eto.Forms import Form, CheckBox, TableRow, TableCell, Drawable, GroupBox, BorderType, Panel, DynamicLayout, VerticalAlignment, TableLayout, ColorPicker,Dialog, Label, TextBox, StackLayout, StackLayoutItem, Orientation, Button, HorizontalAlignment, MessageBox, ProgressBar, ImageView, TextAlignment, Window
from Eto.Drawing import *
import scriptcontext as sc
import System
from System import Array, Double
import csv
from subprocess import Popen, PIPE
from subprocess import call
import ast
import Rhino.Geometry as rg
from shutil import copyfile

#-------------------------------------------- PARAMETER DECLARATION  --------------------------------------------------------------#

# ---------------- INITIALIZE VARIABLES -----------------

pop = []
newpop = []

xval = []
yval = []

convx = []
convy = []

average = []
reportelite = []

global allShapesArray
global qmax
global index

# ---------------- INITIALIZE SHAPE PARAMETERS -----------------

# Specifies shape constraints for generation of initial shapes
global intMinX
intMinX = 0
global intMaxX
intMaxX = 10
global intMinY
intMinY = 0
global intMaxY
intMaxY = 10
global intMinZ
intMinZ = 0
global intMaxZ
intMaxZ = 10

# Defines the trigger to get into interactive mode after five initial genertions 
global firstEvaluation
firstEvaluation = 0

#-------------------------------------------- TYPOGENETIC DESIGN SETUP INTERFACE  --------------------------------------------------------------#

# Sets the current directory so that we can use relative paths   
dir = os.path.dirname(__file__)
# Setup interface initializing content
initialForm = Dialog[bool]()
initialForm.Title = "Typogenetic Design Setup Interface"
initialForm.Resizable = False
layout2 = TableLayout()
layout2.Spacing = Size(5,5)
layout2.Padding = Padding(10,10,10,10)
textBoxPop = TextBox(PlaceholderText = "20")
textBoxCross = TextBox(PlaceholderText = "0.6 > Crossover and Mutation Rate can add to maxial 1")
textBoxMutation = TextBox(PlaceholderText = "0.2 > Difference between Crossover and Mutation Rate to 1 defines Elite Rate")
textBoxGeneration = TextBox(PlaceholderText = "50")
addReferenceButton = Button(Text = "Add reference image")
generateButton = Button(Text = "Generate")
boxAddSelected = ImageView()
boxAddSelected.Image = Bitmap(os.path.join(dir, 'box_init.png'))

def Lb(text):
    """ Text alignment """
    return Label(Text = text, VerticalAlignment = VerticalAlignment.Center, TextAlignment = TextAlignment.Right)

def firstForm():
    """ Setup interface for optimisation parameters """
    layout2.Rows.Add(TableRow(TableCell(Lb("PopSize: ")), textBoxPop))
    layout2.Rows.Add(TableRow(TableCell(Lb("Crossover: ")), textBoxCross))
    layout2.Rows.Add(TableRow(TableCell(Lb("Mutation: ")), textBoxMutation))
    layout2.Rows.Add(TableRow(TableCell(Lb("Generation: ")), textBoxGeneration))
    layout2.Rows.Add(TableRow(TableCell(), TableCell(boxAddSelected)))
    layout2.Rows.Add(TableRow(TableCell(), TableCell(addReferenceButton)))
    layout2.Rows.Add(TableRow(TableCell(), TableCell(generateButton)))
    initialForm.Content = layout2
    firstForm = initialForm.ShowModal(RhinoEtoApp.MainWindow)
    return firstForm


#-------------------------------------------- TYPOGENETIC DESIGN MAIN FUNCTION --------------------------------------------------------------#


def main(image = [], popcount = 20, crossoverrate = 0.6, mutationrate = 0.2, genmax = 50):
    """ Main function integrating genetic programming, online classification and shape comparison """
    # ---------------- SETUP PARAMETERS -----------------

    global gen
    gen = 0
    index = 0
    global allGeometry
    global genMax
    global addSelectedBoolean
    if crossoverrate + mutationrate < 1:
        eliterate = 1 - (crossoverrate + mutationrate)
    else:
        eliterate= 0
    genMax = genmax
    trackelite = []
    matingpool_size = max(3, int(popcount / 10))
    tournament_size = max(3, int(popcount / 50))

     # ---------------- DELETE TEMPORARY FILE CONTENT -----------------

    if os.path.exists("trainer.csv"):
        os.remove("trainer.csv")
    if os.path.exists("shapeInit.csv"):
        os.remove("shapeInit.csv")

    # ---------------- INITIALIZE MAIN LOOP -----------------

    # Initial generation of population
    print("Generating initial population")
    pop = new_population(popcount)
    initialForm.Close()

    # ---------------- RUN MAIN LOOP -----------------

    while (gen < genmax + 1) or (genmax == -1):  # Main Loop
        allGeometry = []
        allShapesArray = []
        print("\nCurrent generation: " + str(gen))

        # ---------------- FITNESS EVALUATION ---------------- 
        enum = 0
        print("after:" + str(len(pop)))
        print("\nFinding Fitness:")
        for p in pop:
            if image:
                # Gets fitness with classifier data if classifier has been trained
                if 'output' in globals(): 
                    for item in output:
                        p.ClassifierFitness = get_fitness_classifier(p, gen, output[enum])
                    if enum < (len(output)-1):
                        enum += 1
                else: 
                    p.ClassifierFitness = get_fitness_classifier(p, gen)
                with open('shapeInit.csv', 'ab') as f:
                    writer = csv.writer(f, delimiter='*')
                    shapeTempList = []
                    appendP = str(p.evaluate())
                    shapeTempList.append(appendP)
                    writer.writerow(shapeTempList)
                # Get both spatial fitness and distance measure
                if not p.SpatialFitness and not p.Distance:
                    get_combined_fitness(p)
                if gen > 5: 
                    p.PreferenceFitness = p.Distance * p.ClassifierFitness
                    p.Fitness = p.SpatialFitness + p.PreferenceFitness
                else:
                    p.Fitness = get_spatial_fitness(p)
            else:
                if not p.Fitness:
                    p.Fitness = get_spatial_fitness(p)
            average.append(p.Fitness)
            
        # ---------------- REMOVE DUPLICATES -----------------

        uniq = []
        seen = set()
        print("before:" + str(len(pop)))
        for ind in pop:
            if ind.Distance not in seen:
                uniq.append(ind)
                seen.add(ind.Distance)
        print("after:" + str(len(pop)))
        pop = uniq

        # ---------------- REPORTING ELITE -----------------

        print("\nGetting Best of Generation:")
        pop.sort(key=lambda individuals: individuals.Fitness)
        upperlimit = 6
        elite = pop[:upperlimit]
        pdot(len(elite))

        # ----------------  REPORTING BEST OF GENERATION -----------------

        qmax = elite[0]
        s = "\nIndividual " + str(qmax.Params) + ":" + str(qmax.Fitness)
        trackelite.append(qmax.Fitness)
        convx.append(gen)
        convy.append(qmax.Fitness)

        # ---------------- FINALISING SEARCH -----------------

        if gen == genmax:
            sc.doc.Views.RedrawEnabled = False
            choice_geometry(elite)
            sc.doc.Views.RedrawEnabled = True 
            break
            break
            break

        # ---------------- SELECTION PROCEDURE -----------------
        
        preSelection = elite

        # ---------------- USER SELECTION -----------------
        if image:
            # Display six shapes
            if (gen - 5) % 10 == 0:
                print("\nGetting User Choice of Desired Solutions:")
                addSelectedBoolean = 0
                sc.doc.Views.RedrawEnabled = False
                choice_geometry(preSelection)
                sc.doc.Views.RedrawEnabled = True    
                drawForm()
                # Generating data for decision tree
                for x in preSelection:
                    allShapesArray.append(str(x.evaluate()))
                # Writing variables to .csv file for training classifier
                with open('trainer.csv', 'ab') as f:
                    writer = csv.writer(f, delimiter='*')
                    for x in range(len(allShapesArray)):
                        tempList = []
                        tempList.append(allShapesArray[x])
                        tempList.append(target[x])
                        writer.writerow(tempList)
            # Init DecisionTree Learning
            command = ["python", "decisionTree.py"]
            p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            global output
            output = p.communicate()
            output = output[0].strip()
            output = output.replace("'", "")
            output = output.replace(" ", "")
            output = output.replace(",", "")
            output = output.replace("\r\n", "")
            output = list(output)
            # Results from decision tree learning
            output = output[1:-1]
            # Delete all geometry
            preview = rs.ObjectsByLayer("Default")
            rs.DeleteObjects(preview)

        # ---------------- MATING POOL TOURNAMENT SELECTION ---------------- 

        print("\nTournament Selection " + str(gen) + ":")
        matecount = 0
        matingpool = []
        while matecount < matingpool_size:
            matingpool.append(tournament_select(pop, tournament_size))
            matecount += 1
        pdot(len(matingpool))

        # ---------------- CROSSOVER ---------------- 

        print("\nMating for generation " + str(gen) + ":")
        mated = mate_all(matingpool, popcount, crossoverrate)
        pdot(len(mated))

        # ---------------- MUTATION ---------------- 

        mutated = []
        print("\nMutating:")
        for i in range(0, int(popcount * mutationrate)):
            mutated.append(mutate(random.choice(pop)))
        pdot(len(mutated))
           
        # ---------------- CREATE NEW GENERATION ---------------- 

        adjustPop = popcount - ( len(mutated) + len(elite) )
        mated = mated[:adjustPop]
        pop = elite + mated  + mutated
        elite = []
        mated = []
        mutated = []

        # ---------------- WRITE LOG FILE ---------------- 

        if gen == 0:
            with open('output.csv', 'ab') as f:
                writer = csv.writer(f)
                a = [os.path.basename(__file__), time.strftime("%d/%m/%Y")]
                writer.writerow(a)
                writer.writerow(["genetic parameters"])
                writer.writerow(["popcount", "generation", "crossoverrate", "mutation rate", "elite rate", "matingpool size", "tournament size"])
                writer.writerow([popcount, genmax, crossoverrate, mutationrate, eliterate, matingpool_size,tournament_size])
                writer.writerow(["Generation", "Best individual", "Best Fitness", "Average Fitness"])
        if gen > 0:
            qmax.evaluate()
            with open('output.csv', 'a') as f:
                writer = csv.writer(f)
                a = [gen, str(qmax.Params), str(qmax.Fitness), sum(average) / float(len(average))]
                writer.writerow(a)
        
        # ---------------- ADVANCE GENERATION ---------------- 

        if image:
            os.remove("shapeInit.csv")
        gen = gen + 1

        # ---------------- END OF MAIN FUNCTION ---------------- 


#-------------------------------------------- FUNCTION DECLARATION --------------------------------------------------------------#


def choice_geometry(elite):
    """ Positions shapes for user selection """
    geo = []
    # Generate shape for selection
    for i in range(6):
        elite[i].evaluate()
        print("Adding geometry: " + str(i))
        offset = 100
        offset_x = i % 3 * offset
        offset_y = i % 2 * offset
        geo.append(add_geometry(elite[i], offset_x, offset_y))
        # Capture the geometry
        view = rs.CurrentView()
        targetView = [offset_x, offset_y, 0]
        location = [offset_x+30, offset_y+30, 30]     
        if location and targetView:
            rs.ViewCameraTarget( view, location, targetView )
        view = sc.doc.Views.ActiveView
        size = System.Drawing.Size(view.Bounds.Width*0.5,view.Bounds.Height*0.5)
        bitmap = view.CaptureToBitmap(size)
        # Creates the shapes folder if it does not exist
        if not os.path.exists("shapes"):
            os.mkdir("shapes")
        # Exports the chosen shapes for review in user selection
        bitmap.Save(os.path.join(dir + '/shapes','generation' + str(gen) + 'shape' + str(i) + '.jpg'))
        memoryStream = System.IO.MemoryStream()
        format = System.Drawing.Imaging.ImageFormat.Png
        System.Drawing.Bitmap.Save(bitmap, memoryStream, format)
        if memoryStream.Length != 0:
           boxArray[i].Image = Bitmap(memoryStream)
           memoryStream.Dispose()
        # Form is redrawn so we need to make sure all the checkbox's are reset
        checkbox1.Checked = False
        checkbox2.Checked = False
        checkbox3.Checked = False
        checkbox4.Checked = False
        checkbox5.Checked = False
        checkbox6.Checked = False
    return geo

def flatten(lis):
    """ Flatten input list """
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis

def progBar():
    """ Update progress bar """
    return gen + 10

def getProgMax():
    """ Set maximum for progress bar """
    return genMax

def get_combined_fitness(ind):
    """ Fitness evaluation using classifier and criteria for trade-off """
    rs.EnableRedraw(False)
    evalGeometry = add_geometry(ind, 0, 0, 1)
    if rs.IsMesh(evalGeometry):
        volume = rs.MeshVolume(evalGeometry)
        area = rs.MeshArea(evalGeometry)
        if area and volume is not None:
            ind.SpatialFitness = area[1] / volume[1]
            ind.Distance = measure_distance(ind)
    else:
        ind.SpatialFitness = 1000
        ind.Distance = 1000
     # Clean Rhinoceros canvas
    arr1 = rs.AllObjects()
    if arr1: 
        rs.DeleteObjects(arr1)
    rs.EnableRedraw(True)

def get_fitness_classifier(ind, generation, output=2):
    """ Translate classifier output to fitness measure """
    output = int(output)
    if output == 0:
        return 1.2
    elif output == 1:
        return  0.4
    elif output == 2:
        return random.random()

def get_spatial_fitness(ind):
    """ Fitness evaluation using area and volume """
    rs.EnableRedraw(False)
    evalGeometry = add_geometry(ind, 0, 0)
    if rs.IsMesh(evalGeometry):
        volume = rs.MeshVolume(evalGeometry)
        area = rs.MeshArea(evalGeometry)
        if area and volume is not None:
            spatialFitness = area[1] / volume[1]
    else:
        spatialFitness = 1000
     # Clean canvas
    arr1 = rs.AllObjects()
    if arr1: 
        rs.DeleteObjects(arr1)
    rs.EnableRedraw(True)
    return spatialFitness

def load_image():
    """ Loading reference image """
    print("Loading preference image")
    # Filter = "PNG (*.png)|*.png"
    filter = "JPG (*.jpg)|*.jpg|BMP (*.bmp)|*.bmp|PNG (*.png)|*.png|All (*.*)|*.*||"
    # Read image    
    filename = rs.OpenFileName("Select image file", filter)
    if not filename: return
    if not os.path.exists("comparison"):
        os.mkdir("comparison")
    copyfile(filename, 'comparison\imageInput.png')
    return filename

def measure_distance(ind):
    """ Measures shape distance between reference image and screen-captured images """
    global returnDistance
    command = ["python", "distanceMeasure.py", "-d", "comparison"]
    q = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    returnDistance = q.communicate()
    returnDistance = returnDistance[0].strip()
    print("ShapeDistance: " + str(returnDistance))
    return float(returnDistance)

def new_population(popcount):
    """ Generates initial population """
    global pop
    while len(pop) < popcount:
        I = Individual()
        I.initialize()
        pop = pop + [I]
    pdot(len(pop))
    return pop

def pdot(length):
    """ Progress indicator """
    for i in range(0, length):
        sys.stdout.write(".")

def tournament_select(fitness_lst, tournament_size):
    """ Selects an individual according to tournament selection """
    tournament_best = None
    for n in range(0, tournament_size):
        ind = random.choice(fitness_lst)
        prog_fit = ind.Fitness
        if tournament_best is None:
            tournament_best = ind
        else:
            tour_fit = tournament_best.Fitness
            if prog_fit > tour_fit:
                tournament_best = ind
    return tournament_best

def mutate(Ind):
    """ Uses mutation operartor to introduce local changes to individual's node tree """
    newInd = deepcopy(Ind)
    newInd.Events[0].evaluate()
    random.choice(newInd.Events[0].subtree()).replace()
    newInd.Fitness = None
    newInd.Distance = None
    newInd.SpatialFitness = None
    newInd.ClassifierFitness = None
    newInd.PreferenceFitness = None
    return newInd

def mate_all(pool, popcount, crossoverrate):
    """ Perform mating of the entire population """
    npop = []
    while len(npop) < int(popcount * crossoverrate):
        # Produce new population by crossover
        npop = npop + mate(random.choice(pool),
                           random.choice(pool))
    pdot(len(npop))
    return npop

def mate(parentA, parentB):
    """ Mate two individuals by subtree replacement """
    ChildA = Individual()
    ChildB = Individual()
    switch = random.randint(0,1)
    if switch == 0:
        ChildA.Events[0].Children[0] = parentA.Events[0].Children[0]
        ChildB.Events[0].Children[0] = parentB.Events[0].Children[0]
        ChildA.Events[0].Children[1] = parentB.Events[0].Children[1]
        ChildB.Events[0].Children[1] = parentA.Events[0].Children[1]
    if switch == 1:
        ChildA.Events[0].Children[0] = parentA.Events[0].Children[0]
        ChildB.Events[0].Children[0] = parentB.Events[0].Children[0]
        ChildA.Events[0].Children[1] = parentA.Events[0].Children[1]
        ChildB.Events[0].Children[1] = parentB.Events[0].Children[1]
        ChildA.Events[0].Children[2] = parentB.Events[0].Children[2]
        ChildB.Events[0].Children[2] = parentA.Events[0].Children[2]
    return [ChildA, ChildB]


#-------------------------------------------- SHAPE GENERATION --------------------------------------------------------------#


def qHull(P):
    """ Returns parameter list for Convex Hull """
    Ilist = []
    for i in range(len(P)):
        arr = Array.CreateInstance(Double, 3)
        x = Double(P[i][0])
        y = Double(P[i][1])
        z = Double(P[i][2])
        listDoubles = [x,y,z]
        for j in range(len(listDoubles)):
            arr[j] = listDoubles[j]
        Ilist.append(arr)
    ListVertices = []
    CHullMesh = rg.Mesh()
    hull = MIConvexHull.ConvexHull.Create(Ilist)
    count = 0
    for face in hull.Faces:
        for i in range(3):
            CHullMesh.Vertices.Add(rg.Point3d(face.Vertices[i].Position[0], face.Vertices[i].Position[1], face.Vertices[i].Position[2]))
        CHullMesh.Faces.AddFace(count, count+1, count+2)
        count += 3
    CHullMesh.Normals.ComputeNormals()
    return CHullMesh

def add_geometry(ind, offset_x, offset_y, visibility = 0):
    """ Grammar translation of individual parameters to geometry calling qHull """
    rs.EnableRedraw(False)
    if ind is not None and ind.Events[0] is not None:
        ind.Params = ind.evaluate()
        completeList = [var for var in ind.Params if var]
        fitmeshlist = []
        # List for mesh generation
        list = [completeList[0]] + [completeList[1]]
        if len(list) > 1:
            for listitem in list:
                listitem=qHull(listitem)
                if listitem is not None:
                    if rs.IsMesh(listitem) and rs.IsMeshManifold(listitem) and rs.IsMeshClosed(listitem):
                        fitmeshlist.append(sc.doc.Objects.AddMesh(listitem))
            if len(fitmeshlist) > 1:
                fitmeshUnion = rs.MeshBooleanUnion(fitmeshlist, True)
                if fitmeshUnion is not None:
                    fitmeshUnion.sort(key=lambda solutions: rs.MeshVolume(solutions)[1])
                    returnGeometry = fitmeshUnion[-1:]
                    deleteArray = fitmeshUnion[:-1]
                    for deleteItem in deleteArray:
                        sc.doc.Objects.Delete(deleteItem, True)
                    if returnGeometry is not None:
                        volume = rs.MeshVolume(returnGeometry)
                        area = rs.MeshArea(returnGeometry)
                        if area and volume is not None:
                            rulesList = completeList[2]
                            # Prints shape grammar rules
                            # print(rulesList)
                            for rule in rulesList:
                                if rule is not None:
                                    bBox = rs.BoundingBox(returnGeometry)
                                    # Applying shape grammar rules
                                    if bBox:
                                        if rule[0] == 401: # Code for move rule
                                            returnGeometry.append(rs.MoveObject(returnGeometry, rs.PointDivide(rule[1], 2)))
                                            returnGeometry = flatten(returnGeometry)
                                            returnGeometry = rs.MeshBooleanUnion(returnGeometry, delete_input=True)
                                        if rule[0] == 402: # Code for rotate rule
                                            refPoint = bBox[0]
                                            returnGeometry.append(rs.RotateObject(returnGeometry, refPoint, rule[1]))
                                            returnGeometry = flatten(returnGeometry)
                                            returnGeometry = rs.MeshBooleanUnion(returnGeometry, delete_input=True)
                                        if rule[0] == 403: # Code for mirror rule
                                            axisStart = rs.PointScale(rs.PointSubtract(bBox[0], bBox[2]), rule[1])
                                            axisEnd = rs.PointAdd(axisStart, (0, 2000, 0))
                                            returnGeometry.append(rs.MirrorObjects(returnGeometry, axisStart, axisEnd, copy = True)) # Copy might be True as well, but needs debugging
                                            returnGeometry = flatten(returnGeometry)
                                            returnGeometry = rs.MeshBooleanUnion(returnGeometry, delete_input=True)
                                        if rule[0] == 404: # Code for scale rule
                                            refPoint = bBox[0]
                                            returnGeometry.append(rs.ScaleObject(returnGeometry, refPoint, rule[1]))
                                            returnGeometry = flatten(returnGeometry)
                                            returnGeometry = rs.MeshBooleanUnion(returnGeometry, delete_input=True)
                                    if returnGeometry is not None:
                                        returnGeometry = flatten(returnGeometry)
                                        returnGeometry.sort(key=lambda solutions: rs.MeshVolume(solutions)[1])
                                        deleteArray = returnGeometry[:-1]
                                        for deleteItem in deleteArray:
                                            sc.doc.Objects.Delete(deleteItem, True)
                                        returnGeometry = returnGeometry[-1:]
                                    else:
                                        return None
                            # Move shapes into position for screen capture
                            if rs.IsMesh(returnGeometry) and rs.IsMeshManifold(returnGeometry) and rs.IsMeshClosed(returnGeometry):
                                centrePoint = rs.AddPoint( rs.MeshAreaCentroid(returnGeometry) )
                                offsetVector = rs.VectorSubtract((offset_x, offset_y, 0), rs.PointCoordinates(centrePoint))
                                sc.doc.Objects.Delete(centrePoint, True)
                                finalGeometry = rs.CopyObjects(returnGeometry, offsetVector)
                                sc.doc.Objects.Delete(returnGeometry, True)
                                sc.doc.Objects.Delete(fitmeshUnion, True)
                                if image:
                                    # Capture images of the shapes
                                    if not ind.Distance:
                                        rs.EnableRedraw(True)
                                        view = rs.CurrentView()
                                        targetView = [offset_x, offset_y, 0]
                                        location = [offset_x+30, offset_y+30, 30]     
                                        if location and targetView:
                                            rs.ViewCameraTarget( view, location, targetView )
                                        view = sc.doc.Views.ActiveView
                                        size = System.Drawing.Size(view.Bounds.Width*0.8,view.Bounds.Height*0.8)
                                        bitmap = view.CaptureToBitmap(size)
                                        # Exports the chosen shapes
                                        bitmap.Save(os.path.join(dir + '/comparison/shape' + '.png'))
                                        memoryStream = System.IO.MemoryStream()
                                        format = System.Drawing.Imaging.ImageFormat.Png
                                        System.Drawing.Bitmap.Save(bitmap, memoryStream, format)
                                # Return correct mesh geometry
                                return finalGeometry
                            else:
                                return None
                        else:
                            return None
                    else:
                        return None
            else:
                return None
        else:
            return None
        # Delete construction geometry for shapes
        sc.doc.Objects.Delete(fitmeshUnion, True)
        for fitmeshitem in returnGeometry:
            sc.doc.Objects.Delete(fitmeshitem, True)
        rs.EnableRedraw(True)


#-------------------------------------------- INTERFACE DECLARATION --------------------------------------------------------------#


def SetToRendered():
    """ Sets Rhinoceros viewports for user evaluation """
    views = rs.ViewNames()
    modes=rs.ViewDisplayModes()
    viewtype="Rendered"
    if viewtype in modes:
        for view in views:
            rs.ViewDisplayMode(view, viewtype)
            if rs.ShowGrid(view)==True:
                rs.ShowGrid(view, False)
            if rs.ShowGridAxes(view)==True:
                rs.ShowGridAxes(view, False)
            if rs.ShowWorldAxes(view)==True:
                rs.ShowWorldAxes(view, False)
            rs.AppearanceColor(0, color=(255,255,255))

def L(text):
    """ Text alignment """
    return Label(Text = text, VerticalAlignment = VerticalAlignment.Center, TextAlignment = TextAlignment.Right)

# Defining textboxes, progress bars and buttons
progressBar = ProgressBar(Value = 0, MaxValue = 50)
addSelectedButton = Button(Text = "Add Selected to Favourites and Continue...")
continueButton = Button(Text = "Continue...")

# Initialise array to store selected solutions
target = []

# ----------------  ADDING BUTTON FUNCTIONALITY -----------------

def generateButton_click(sender, e): 
    """ Starts Typogenetic Design """
    # Sets Rhinoceros viewports
    SetToRendered()
    # Semi-automated vs. automated scenario
    if 'image' in globals():
        # System runs genetic programming in Typogenetic Design mode
        print(image)
        main(image[0],
            int(textBoxPop.Text) if textBoxPop.Text != "" else 20,
            float(textBoxCross.Text) if textBoxCross.Text != "" else 0.7,
            float(textBoxMutation.Text) if textBoxMutation.Text != "" else 0.3,
            int(textBoxGeneration.Text) if textBoxGeneration.Text != "" else 25)
    # System runs genetic programming in automated mode as shape optimizer
    else:
        global image
        image = []
        main(image,
            int(textBoxPop.Text) if textBoxPop.Text != "" else 20,
            float(textBoxCross.Text) if textBoxCross.Text != "" else 0.7,
            float(textBoxMutation.Text) if textBoxMutation.Text != "" else 0.3,
            int(textBoxGeneration.Text) if textBoxGeneration.Text != "" else 25)
generateButton.Click += generateButton_click

def addReferenceButton_click(sender, e):
    """ Loads reference image """
    global image
    image = load_image()
    boxAddSelected.Image = Bitmap(image)
addReferenceButton.Click += addReferenceButton_click

# ----------------  USER SELECTION -----------------

def addSelectedButton_click(sender, e): 
    """ Add shapes to array if checkboxes are checked """
    addSelectedBoolean = 1
    if checkbox1.Checked == True:
        target.append(int(1))
    else:
        target.append(int(0))
    if checkbox2.Checked == True:
        target.append(int(1))
    else:
        target.append(int(0))
    if checkbox3.Checked == True:
        target.append(int(1))
    else:
        target.append(int(0))
    if checkbox4.Checked == True:
        target.append(int(1))
    else:
        target.append(int(0))
    if checkbox5.Checked == True:
        target.append(int(1))
    else:
        target.append(int(0))
    if checkbox6.Checked == True:
        target.append(int(1))
    else:
        target.append(int(0))
    selectionForm.Close()
    return(target)
addSelectedButton.Click += addSelectedButton_click

def continueButton_click(sender, e):
    """ Check if shapes were selected """
    if addSelectedBoolean == 0:
        MessageBox.Show("Please add selected before continuing")
    else:
        selectionForm.Close()
continueButton.Click += continueButton_click    

# Defining form objects
boxArray = []
box1 = ImageView()
box1.Image = Bitmap(os.path.join(dir, 'box.png'))
box2 = ImageView()
box2.Image = Bitmap(os.path.join(dir, 'box.png'))
box3 = ImageView()
box3.Image = Bitmap(os.path.join(dir, 'box.png'))
box4 = ImageView()
box4.Image = Bitmap(os.path.join(dir, 'box.png'))
box5 = ImageView()
box5.Image = Bitmap(os.path.join(dir, 'box.png'))
box6 = ImageView()
box6.Image = Bitmap(os.path.join(dir, 'box.png'))
# Creating array from form objects
boxArray.append(box1)
boxArray.append(box2)
boxArray.append(box3)
boxArray.append(box4)
boxArray.append(box5)
boxArray.append(box6)
# Defining checkboxes for images
checkbox1 = CheckBox()
checkbox2 = CheckBox()
checkbox3 = CheckBox()
checkbox4 = CheckBox()
checkbox5 = CheckBox()
checkbox6 = CheckBox()

def drawForm():
    """ Initialising form """
    global selectionForm
    selectionForm = Dialog[bool]()
    selectionForm.Title = "Artificial Evaluation Form"
    selectionForm.Resizable = False
    layout = TableLayout()
    layout.Spacing = Size(5,5)
    layout.Padding = Padding(10,10,10,10)
    # Drawing objects to form
    layout.Rows.Add(TableRow(TableCell(L("Please choose which shapes you like"))))
    layout.Rows.Add(TableRow(TableCell(box1), TableCell(box2), TableCell(box3)))
    layout.Rows.Add(TableRow(TableCell(checkbox1), TableCell(checkbox2), TableCell(checkbox3)))
    layout.Rows.Add(TableRow(TableCell(box4), TableCell(box5), TableCell(box6)))
    layout.Rows.Add(TableRow(TableCell(checkbox4), TableCell(checkbox5), TableCell(checkbox6)))
    layout.Rows.Add(TableRow(TableCell(addSelectedButton)))
    layout.Rows.Add(TableRow(TableCell(progressBar)))
    layout.Rows.Add(None)
    progressBar.Value = progBar()
    progressBar.MaxValue = getProgMax()
    selectionForm.Content = layout
    selectionForm.DefaultButton = addSelectedButton
    selectionForm.ShowModal(RhinoEtoApp.MainWindow)


#-------------------------------------------- GENETIC PROGRAMMING DECLARATION --------------------------------------------------------------#


pass  # ----------------  INDIVIDUAL DECLARATION -----------------

class Individual:
    """ Defines the individual as part of the population """
    def __init__(self):
        self.Events = [NodeRoot()]
        self.Params = []
        self.Fitness = None
        self.SpatialFitness = None
        self.Distance = None
        self.ClassifierFitness = None
        self.PreferenceFitness = None

    def initialize(self):
        """ Initializes node tree """
        self.Events[0].add()

    def evaluate(self):
        """ Evaluates node tree for grammar translation """
        self.Params = self.Events[0].evaluate()
        return self.Params

pass  # ----------------  NODE CLASS DECLARATION -----------------

class Node:
    def __init__(self):
        self.Children = []
        self.Parameters = []
        self.OutType = None
        self.AcceptsTypes = None

    def evaluate(self):
        """ Receiving tree items in preorder sequence """
        tempParams = []
        if self.Children:
            for i in range(0, len(self.Children)):
                if self.Children[i] is not None:
                    self.Children[i].evaluate()
                    tempParams.append(self.Parameters)
            self.Parameters = tempParams
        return tempParams

    def add(self):
        """ Builds node tree """
        if self.Children:
            for i in range(0, len(self.Children)):
                if self.Children[i] is None:
                    self.Children[i] = self.select_node()
                    self.Children[i].add()
                if self.Children[i] is not None:
                    self.Children[i].add()

    def replace(self):
        """ Replaces sub tree """
        if self.Children:
            for i in range(0, len(self.Children)):
                self.Children[i] = self.select_node()
                self.Children[i].replace()

    def select_node(self):
        """ Select nodes using grammar syntax """
        for t in range(0, len(self.AcceptsTypes)):
            item = self.AcceptsTypes[t]
            if item == 'fInteger':
                n = random.choice([NodeIntegerX(), NodeIntegerY(), NodeIntegerZ()])
                return n
            if item == 'fPoint':
                n = NodePoint()
                return n
            if item == 'fShape':
                n = NodeShape()
                return n
            if item == 'fGrammarRules':
                n = NodeGrammarRules()
            if item == 'fRule':
                n = random.choice([NodeRuleMove(), NodeRuleRotate(), NodeRuleMirror(), NodeRuleScale()])
                return n
            else:
                return None

    def subtree(self):
        """ Constructs sub-tree """
        subTree = []
        if self.Children:
            for i in range(0, len(self.Children)):
                    if self.Children[i] is not None:
                        subTree.append(self.Children[i])
        return subTree

pass  # ----------------  ROOT NODE DECLARATION -----------------

class NodeRoot(Node):
    """ Defines root node for constructing node tree """
    def __init__(self):
        Node.__init__(self)
        self.OutType = [None]
        self.AcceptsTypes = ['fShape','fGrammarRules']
        self.Children = [NodeShape(), NodeShape(),NodeGrammarRules()]

    def evaluate(self):
        """ Receiving sub tree items in preorder sequence """
        tempParams = []
        for c in self.Children:
            tempParams.append(c.evaluate())
        self.Parameters = tempParams
        return self.Parameters

pass  # ----------------  ELEMENT NODE DECLARATION -----------------

class NodeShape(Node):
    """ Defines shape node for constructing node tree """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fShape']
        self.AcceptsTypes = ['fPoint']
        # Number of points defined by number of items in self.Children
        self.Children = [None, None, None, None, None,
                         None, None, None, None, None]

    def evaluate(self):
        """ Receiving sub tree items in preorder sequence """
        tempParams = []
        for c in self.Children:
                tempParams.append(c.evaluate())
        tempParams = [var for var in tempParams if var]
        self.Parameters = tempParams
        return self.Parameters

class NodePoint(Node):
    """ Defines point node for constructing node tree """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fPoint']
        self.AcceptsTypes = ['fInteger']
        self.Children = [NodeIntegerX(), NodeIntegerY(), NodeIntegerZ()]

    def evaluate(self):
        """ Receiving sub tree items in preorder sequence """
        tempParams = []
        for c in self.Children:
            tempParams.append(c.evaluate())
        self.Parameters = tempParams
        return self.Parameters

class NodePointScale(Node):
    """ Defines scale node for constructing node tree """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fPoint']
        self.AcceptsTypes = ['fFloat']
        self.Children = [NodeFloatScale(), NodeFloatScale(), NodeFloatScale()]

    def evaluate(self):
        """ Receiving sub tree items in preorder sequence """
        tempParams = []
        for c in self.Children:
            tempParams.append(c.evaluate())
        self.Parameters = tempParams
        return self.Parameters

pass  # ----------------  NODE CLASS BASICS -----------------

class NodeEmpty(Node):
    """ Defines empty node for constructing node tree """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fPoint']
        self.AcceptsTypes = None
        self.Children = []
        self.Parameters = []

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters


class NodeIntegerX(Node):
    """ Defines integer node in x direction for constructing node tree """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fInteger']
        self.AcceptsTypes = None
        self.Parameters = random.randint(intMinX, intMaxX)

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters

class NodeIntegerY(Node):
    """ Defines integer node in x direction for constructing node tree """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fInteger']
        self.AcceptsTypes = None
        self.Parameters = random.randint(intMinY, intMaxY)

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters

class NodeIntegerZ(Node):
    """ Defines integer node in x direction for constructing node tree """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fInteger']
        self.AcceptsTypes = None
        self.Parameters = random.randint(intMinZ, intMaxZ)

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters

pass  # ----------------  SHAPE GRAMMAR DECLARATION -----------------

class NodeGrammarRules(Node):
    """ Shape grammar rules for shape construction """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fGrammarRules']
        self.AcceptsTypes = ['fRule']
        # Number of rules defined by number of items in self.Children
        self.Children = [None, None, None, None]

    def evaluate(self):
        """ Receiving sub tree items in preorder sequence """
        tempParams = []
        for c in self.Children:
            if c is not None:
                tempParams.append(c.evaluate())
            else:
                # Add rule nodes if node tree is ill-defined
                c = random.choice([NodeRuleMove(), NodeRuleRotate(), NodeRuleMirror(), NodeRuleScale()])
                tempParams.append(c.evaluate())
        self.Parameters = tempParams
        return self.Parameters

class NodeRuleRotate(Node):
    """ Shape grammar rule for rotation """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fRule']
        self.AcceptsTypes = ['fInteger']
        # Number of rules defined by number of items in self.Children
        self.Children = [NodeMarkerRotate(), NodeIntegerAngle()]

    def evaluate(self):
        """ Receiving sub tree items in preorder sequence """
        tempParams = []
        for c in self.Children:
            tempParams.append(c.evaluate())
        self.Parameters = tempParams
        return self.Parameters

class NodeRuleMove(Node):
    """ Shape grammar rule for translation """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fRule']
        self.AcceptsTypes = ['fInteger', 'fPoint']
        # Number of rules defined by number of items in self.Children
        self.Children = [NodeMarkerMove(), NodePoint()]

    def evaluate(self):
        """ Receiving sub tree items in preorder sequence """
        tempParams = []
        for c in self.Children:
            tempParams.append(c.evaluate())
        self.Parameters = tempParams
        return self.Parameters

class NodeRuleMirror(Node):
    """Shape grammar rule for mirroring"""
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fRule']
        self.AcceptsTypes = ['fInteger', 'fFloat']
        # Number of rules defined by number of items in self.Children
        self.Children = [NodeMarkerMirror(), NodeFloatMirror()]

    def evaluate(self):
        """ Receiving sub tree items in preorder sequence """
        tempParams = []
        for c in self.Children:
            tempParams.append(c.evaluate())
        self.Parameters = tempParams
        return self.Parameters

class NodeRuleScale(Node):
    """ Shape grammar rule for scaling """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fRule']
        self.AcceptsTypes = ['fInteger', 'fFloat']
        # Number of rules defined by number of items in self.Children
        self.Children = [NodeMarkerScale(), NodePointScale()]

    def evaluate(self):
        """ Receiving sub tree items in preorder sequence """
        tempParams = []
        for c in self.Children:
            tempParams.append(c.evaluate())
        self.Parameters = tempParams
        return self.Parameters

pass  # ----------------  PARAMETER NODES FOR SHAPE GRAMMAR -----------------

class NodeIntegerAngle(Node):
    """ Parameter node for rotation rule """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fInteger']
        self.AcceptsTypes = None
        self.Parameters = random.randint(0, 45)

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters

class NodeFloatScale(Node):
    """ Parameter node for scale rule """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fFloat']
        self.AcceptsTypes = None
        self.Parameters = random.uniform(0.8, 1.2)

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters
        
class NodeFloatMirror(Node):
    """ Parameter node for mirror rule """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fFloat']
        self.AcceptsTypes = None
        self.Parameters = random.random()

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters

pass  # ----------------  MARKER NODES FOR SHAPE GRAMMAR -----------------

class NodeMarkerMove(Node):
    """ Marker node to trigger move rule in grammar translation """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fInteger']
        self.AcceptsTypes = None
        self.Parameters = 401

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters

class NodeMarkerRotate(Node):
    """ Marker node to trigger rotate rule in grammar translation """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fInteger']
        self.AcceptsTypes = None
        self.Parameters = 402

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters
        
class NodeMarkerMirror(Node):
    """ Marker node to trigger mirror rule in grammar translation """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fInteger']
        self.AcceptsTypes = None
        self.Parameters = 403

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters
        
class NodeMarkerScale(Node):
    """ Marker node to trigger scale rule in grammar translation """
    def __init__(self):
        Node.__init__(self)
        self.OutType = ['fInteger']
        self.AcceptsTypes = None
        self.Parameters = 404

    def evaluate(self):
        """ Return paramter value """
        return self.Parameters


#-------------------------------------------- RUN TYPOGENETIC DESIGN --------------------------------------------------------------#


firstForm()