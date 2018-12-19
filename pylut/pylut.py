# MIT License
# Authored by Greg Cotten


import os
import math
import numpy as np
import resources.kdtree as kdtree
from resources.progress.bar import Bar
import struct


def empty_lattice_of_size(cube_size):
    return np.zeros((cube_size, cube_size, cube_size), object)


def indices_01(cube_size):
    _indices = []
    ratio = 1.0 / float(cube_size - 1)
    for i in range(cube_size):
        _indices.append(float(i) * ratio)
    return _indices


def indices(cube_size, max_val):
    _indices = []
    for i in indices_01(cube_size):
        _indices.append(int(i * max_val))
    return _indices


def RemapIntTo01(val, max_val):
    return float(val) / float(max_val)


def Remap01ToInt(val, max_val):
    return int(iround(float(val) * float(max_val)))


def iround(num):
    if num > 0:
        return int(num + .5)
    else:
        return int(num - .5)


def LerpColor(beginning, end, value01):
    if value01 < 0 or value01 > 1:
        raise NameError("Improper Lerp")
    return Color(Lerp1D(beginning.r, end.r, value01), Lerp1D(beginning.g, end.g, value01),
                 Lerp1D(beginning.b, end.b, value01))


def Lerp3D(beginning, end, value01):
    if value01 < 0 or value01 > 1:
        raise NameError("Improper Lerp")
    return [Lerp1D(beginning[0], end[0], value01), Lerp1D(beginning[1], end[1], value01),
            Lerp1D(beginning[2], end[2], value01)]


def Lerp1D(beginning, end, value01):
    if value01 < 0 or value01 > 1:
        raise NameError("Improper Lerp")

    Lrange = float(end) - float(beginning)
    return float(beginning) + float(Lrange) * float(value01)


def Distance3D(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def Clamp(value, min_boundary, max_boundary):
    if min_boundary > max_boundary:
        raise NameError("Invalid Clamp Values")
    if value < min_boundary:
        return float(min_boundary)
    if value > max_boundary:
        return float(max_boundary)
    return value


def Checksum(data):
    total = 0
    for x in data:
        total += sum(struct.unpack("<B", x))
    return total


def ToIntArray(string):
    array = []
    for x in string:
        array.append(ord(x))
    return array


class Color:
    """
    RGB floating point representation of a color. 0 is absolute black, 1 is absolute white.
    Access channel data by color.r, color.g, or color.b.
    """

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def Clamped01(self):
        return Color(Clamp(float(self.r), 0, 1), Clamp(float(self.g), 0, 1), Clamp(float(self.b), 0, 1))

    @staticmethod
    def FromRGBInteger(r, g, b, bitdepth):
        """
        Instantiates a floating point color from RGB integers at a bitdepth.
        """
        maxBits = 2 ** bitdepth - 1
        return Color(RemapIntTo01(r, maxBits), RemapIntTo01(g, maxBits), RemapIntTo01(b, maxBits))

    @staticmethod
    def FromFloatArray(array):
        """
        Creates Color from a list or tuple of 3 floats.
        """
        return Color(array[0], array[1], array[2])

    @staticmethod
    def FromRGBIntegerArray(array, bitdepth):
        """
        Creates Color from a list or tuple of 3 RGB integers at a specified bitdepth.
        """
        maxBits = 2 ** bitdepth - 1
        return Color(RemapIntTo01(array[0], maxBits), RemapIntTo01(array[1], maxBits), RemapIntTo01(array[2], maxBits))

    def ToFloatArray(self):
        """
        Creates a tuple of 3 floating point RGB values from the floating point color.
        """
        return tuple([self.r, self.g, self.b])

    def ToRGBIntegerArray(self, bitdepth):
        """
        Creates a list of 3 RGB integer values at specified bitdepth from the floating point color.
        """
        max_val = (2 ** bitdepth - 1)
        return Remap01ToInt(self.r, max_val), Remap01ToInt(self.g, max_val), Remap01ToInt(self.b, max_val)

    def ClampColor(self, minC, maxC):
        """
        Returns a clamped color.
        """
        return Color(Clamp(self.r, minC.r, maxC.r), Clamp(self.g, minC.g, maxC.g), Clamp(self.b, minC.b, maxC.b))

    def DistanceToColor(self, color):
        if isinstance(color, Color):
            return Distance3D(self.ToFloatArray(), color.ToFloatArray())
        return NotImplemented

    def __add__(self, color):
        return Color(self.r + color.r, self.g + color.g, self.b + color.b)

    def __sub__(self, color):
        return Color(self.r - color.r, self.g - color.g, self.b - color.b)

    def __mul__(self, color):
        if not isinstance(color, Color):
            mult = float(color)
            return Color(self.r * mult, self.g * mult, self.b * mult)
        return Color(self.r * color.r, self.g * color.g, self.b * color.b)

    def __eq__(self, color):
        if isinstance(color, Color):
            return self.r == color.r and self.g == color.g and self.b == color.b
        return NotImplemented

    def __ne__(self, color):
        result = self.__eq__(color)
        if result is NotImplemented:
            return result
        return not result

    def __str__(self):
        return "(" + str(self.r) + ", " + str(self.g) + ", " + str(self.b) + ")"

    def FormattedAsFloat(self, formatstr='{:1.6f}'):
        return formatstr.format(self.r) + " " + formatstr.format(self.g) + " " + formatstr.format(self.b)

    def FormattedAsInteger(self, max_val):
        rjustValue = len(str(max_val)) + 1
        return str(Remap01ToInt(self.r, max_val)).rjust(rjustValue) + " " + str(Remap01ToInt(self.g, max_val)).rjust(
            rjustValue) + " " + str(Remap01ToInt(self.b, max_val)).rjust(rjustValue)


class LUT:
    """
    A class that represents a 3D LUT with a 3D numpy array. The idea is that the modifications are non-volatile, meaning that every modification method returns a new LUT object.
    """

    def __init__(self, lattice, name="Untitled LUT"):
        self.lattice = lattice
        """
        Numpy 3D array representing the 3D LUT.
        """

        self.cube_size = self.lattice.shape[0]
        """
        LUT is of size (cube_size, cube_size, cube_size) and index positions are from 0 to cube_size-1
        """

        self.name = str(name)
        """
        Every LUT has a name!
        """

    def Resize(self, newcube_size):
        """
        Scales the lattice to a new cube size.
        """
        if newcube_size == self.cube_size:
            return self

        newLattice = empty_lattice_of_size(newcube_size)
        ratio = float(self.cube_size - 1.0) / float(newcube_size - 1.0)
        for x in range(newcube_size):
            for y in range(newcube_size):
                for z in range(newcube_size):
                    newLattice[x, y, z] = self.ColorAtInterpolatedLatticePoint(x * ratio, y * ratio, z * ratio)
        return LUT(newLattice, name=self.name + "_Resized" + str(newcube_size))

    def _ResizeAndAddToData(self, newcube_size, data, progress=False):
        """
        Scales the lattice to a new cube size.
        """
        ratio = float(self.cube_size - 1.0) / float(newcube_size - 1.0)
        max_val = newcube_size - 1

        bar = Bar("Building search tree", max=max_val, suffix='%(percent)d%% - %(eta)ds remain')
        try:
            for x in range(newcube_size):
                if progress:
                    bar.next()
                for y in range(newcube_size):
                    for z in range(newcube_size):
                        data.add(self.ColorAtInterpolatedLatticePoint(x * ratio, y * ratio, z * ratio).ToFloatArray(),
                                 (RemapIntTo01(x, max_val), RemapIntTo01(y, max_val), RemapIntTo01(z, max_val)))
        except KeyboardInterrupt:
            bar.finish()
            raise KeyboardInterrupt
        bar.finish()
        return data

    def Reverse(self, progress=False):
        """
        Reverses a LUT. Warning: This can take a long time depending on if the input/output is a bijection.
        """
        tree = self.KDTree(progress)
        newLattice = empty_lattice_of_size(self.cube_size)
        max_val = self.cube_size - 1
        bar = Bar("Searching for matches", max=max_val, suffix='%(percent)d%% - %(eta)ds remain')
        try:
            for x in range(self.cube_size):
                if progress:
                    bar.next()
                for y in range(self.cube_size):
                    for z in range(self.cube_size):
                        newLattice[x, y, z] = Color.FromFloatArray(tree.search_nn(
                            (RemapIntTo01(x, max_val), RemapIntTo01(y, max_val), RemapIntTo01(z, max_val))).aux)
        except KeyboardInterrupt:
            bar.finish()
            raise KeyboardInterrupt
        bar.finish()
        return LUT(newLattice, name=self.name + "_Reverse")

    def KDTree(self, progress=False):
        tree = kdtree.create(dimensions=3)

        tree = self._ResizeAndAddToData(self.cube_size * 3, tree, progress)

        return tree

    def CombineWithLUT(self, otherLUT):
        """
        Combines LUT with another LUT.
        """
        if self.cube_size is not otherLUT.cube_size:
            raise NameError("Lattice Sizes not equivalent")

        cube_size = self.cube_size
        newLattice = empty_lattice_of_size(cube_size)

        for x in range(cube_size):
            for y in range(cube_size):
                for z in range(cube_size):
                    selfColor = self.lattice[x, y, z].Clamped01()
                    newLattice[x, y, z] = otherLUT.ColorFromColor(selfColor)
        return LUT(newLattice, name=self.name + "+" + otherLUT.name)

    def ClampColor(self, minC, maxC):
        """
        Returns a new RGB clamped LUT.
        """
        cube_size = self.cube_size
        newLattice = empty_lattice_of_size(cube_size)
        for x in range(cube_size):
            for y in range(cube_size):
                for z in range(cube_size):
                    newLattice[x, y, z] = self.ColorAtLatticePoint(x, y, z).ClampColor(minC, maxC)
        return LUT(newLattice)

    def _LatticeTo3DLString(self, bitdepth):
        """
        Used for internal creating of 3DL files.
        """
        string = ""
        cube_size = self.cube_size
        for currentCubeIndex in range(0, cube_size ** 3):
            redIndex = currentCubeIndex / (cube_size * cube_size)
            greenIndex = ((currentCubeIndex % (cube_size * cube_size)) / cube_size)
            blueIndex = currentCubeIndex % cube_size

            latticePointColor = self.lattice[redIndex, greenIndex, blueIndex].Clamped01()

            string += latticePointColor.FormattedAsInteger(2 ** bitdepth - 1) + "\n"

        return string

    def ToLustre3DLFile(self, fileOutPath, bitdepth=12):
        # do not use 3dl
        cube_size = self.cube_size
        inputDepth = math.log(cube_size - 1, 2)

        if int(inputDepth) != inputDepth:
            raise NameError("Invalid cube size for 3DL. Cube size must be 2^x + 1")

        lutFile = open(fileOutPath, 'w')

        lutFile.write("3DMESH\n")
        lutFile.write("Mesh " + str(int(inputDepth)) + " " + str(bitdepth) + "\n")
        lutFile.write(' '.join([str(int(x)) for x in indices(cube_size, 2 ** 10 - 1)]) + "\n")

        lutFile.write(self._LatticeTo3DLString(bitdepth))

        lutFile.write("\n#Tokens required by applications - do not edit\nLUT8\ngamma 1.0")

        lutFile.close()

    def ToNuke3DLFile(self, fileOutPath, bitdepth=16):
        # do not use 3dl
        cube_size = self.cube_size

        lutFile = open(fileOutPath, 'w')

        lutFile.write(' '.join([str(int(x)) for x in indices(cube_size, 2 ** bitdepth - 1)]) + "\n")

        lutFile.write(self._LatticeTo3DLString(bitdepth))

        lutFile.close()

    def ToCubeFile(self, cubeFileOutPath):
        # should only be using this method
        cube_size = self.cube_size
        cubeFile = open(cubeFileOutPath, 'w')
        fileName = os.path.basename(cubeFileOutPath)
        cubeFile.write('TITLE "' + fileName + '"' + '\n\n')
        cubeFile.write("LUT_3D_SIZE " + str(cube_size) + "\n\n")

        for currentCubeIndex in range(0, cube_size ** 3):
            redIndex = currentCubeIndex % cube_size
            greenIndex = int((currentCubeIndex % (cube_size * cube_size)) / cube_size)
            blueIndex = int(currentCubeIndex / (cube_size * cube_size))

            latticePointColor = self.lattice[redIndex, greenIndex, blueIndex].Clamped01()

            cubeFile.write(latticePointColor.FormattedAsFloat())

            if currentCubeIndex != cube_size ** 3 - 1:
                cubeFile.write("\n")
        cubeFile.write("\n")
        cubeFile.close()

    def ToFSIDatFile(self, datFileOutPath):
        cube_size = 64
        datFile = open(datFileOutPath, 'w+b')
        if self.cube_size is not 64:
            lut = self.Resize(64)
        else:
            lut = self
        lut_checksum = 0
        lut_bytes = []
        for currentCubeIndex in range(0, cube_size ** 3):
            redIndex = currentCubeIndex % cube_size
            greenIndex = int((currentCubeIndex % (cube_size * cube_size)) / cube_size)
            blueIndex = int(currentCubeIndex / (cube_size * cube_size))

            latticePointColor = lut.lattice[redIndex, greenIndex, blueIndex].Clamped01()

            rgb_packed = (Remap01ToInt(latticePointColor.r, 1008) | Remap01ToInt(latticePointColor.g,
                                                                                 1008) << 10 | Remap01ToInt(
                latticePointColor.g, 1008) << 20)
            rgb_packed_binary = struct.pack("<L", rgb_packed)
            lut_checksum = (lut_checksum + rgb_packed) % 4294967296
            lut_bytes.append(rgb_packed_binary)

        header_bytes = list()
        header_bytes.append(struct.pack("<L", 0x42340299))  # magic number
        header_bytes.append(struct.pack("<L", 0x01000002))  # spec version number?
        header_bytes.append(bytearray("None".ljust(16)))  # monitor ID (real ID not required if dit.dat file)
        header_bytes.append(bytearray("V1.0".ljust(16)))  # lut version number
        header_bytes.append(struct.pack("<L", lut_checksum))
        header_bytes.append(struct.pack("<L", 1048576))  # number of bytes in LUT (always the same)
        header_bytes.append(bytearray("pylut generated".ljust(16)))  # author
        header_bytes.append(bytearray(" ".ljust(63)))  # reserved

        header_checksum = 0

        for item in header_bytes:
            if isinstance(item, str):
                itemSum = sum(map(ord, item))
            else:
                itemSum = sum(item)
            header_checksum = (header_checksum + itemSum) % 256

        header_bytes.append(struct.pack("<B", header_checksum))

        [datFile.write(x) for x in header_bytes]
        [datFile.write(x) for x in lut_bytes]

        datFile.close()

    def ColorFromColor(self, color):
        """
        Returns what a color value should be transformed to when piped through the LUT.
        """
        color = color.Clamped01()
        cube_size = self.cube_size
        return self.ColorAtInterpolatedLatticePoint(color.r * (cube_size - 1), color.g * (cube_size - 1),
                                                    color.b * (cube_size - 1))

    # integer input from 0 to cube_size-1
    def ColorAtLatticePoint(self, redPoint, greenPoint, bluePoint):
        """
        Returns a color at a specified lattice point - this value is pulled from the actual LUT file and is not interpolated.
        """
        cube_size = self.cube_size
        if redPoint > cube_size - 1 or greenPoint > cube_size - 1 or bluePoint > cube_size - 1:
            raise NameError(
                "Point Out of Bounds: (" + str(redPoint) + ", " + str(greenPoint) + ", " + str(bluePoint) + ")")

        return self.lattice[int(redPoint), int(greenPoint), int(bluePoint)]

    # float input from 0 to cube_size-1
    def ColorAtInterpolatedLatticePoint(self, redPoint, greenPoint, bluePoint):
        """
        Gets the interpolated color at an interpolated lattice point.
        """
        cube_size = self.cube_size

        if 0 < redPoint > cube_size - 1 or 0 < greenPoint > cube_size - 1 or 0 < bluePoint > cube_size - 1:
            raise NameError("Point Out of Bounds")

        lowerRedPoint = Clamp(int(math.floor(redPoint)), 0, cube_size - 1)
        upperRedPoint = Clamp(lowerRedPoint + 1, 0, cube_size - 1)

        lowerGreenPoint = Clamp(int(math.floor(greenPoint)), 0, cube_size - 1)
        upperGreenPoint = Clamp(lowerGreenPoint + 1, 0, cube_size - 1)

        lowerBluePoint = Clamp(int(math.floor(bluePoint)), 0, cube_size - 1)
        upperBluePoint = Clamp(lowerBluePoint + 1, 0, cube_size - 1)

        C000 = self.ColorAtLatticePoint(lowerRedPoint, lowerGreenPoint, lowerBluePoint)
        C010 = self.ColorAtLatticePoint(lowerRedPoint, lowerGreenPoint, upperBluePoint)
        C100 = self.ColorAtLatticePoint(upperRedPoint, lowerGreenPoint, lowerBluePoint)
        C001 = self.ColorAtLatticePoint(lowerRedPoint, upperGreenPoint, lowerBluePoint)
        C110 = self.ColorAtLatticePoint(upperRedPoint, lowerGreenPoint, upperBluePoint)
        C111 = self.ColorAtLatticePoint(upperRedPoint, upperGreenPoint, upperBluePoint)
        C101 = self.ColorAtLatticePoint(upperRedPoint, upperGreenPoint, lowerBluePoint)
        C011 = self.ColorAtLatticePoint(lowerRedPoint, upperGreenPoint, upperBluePoint)

        C00 = LerpColor(C000, C100, 1.0 - (upperRedPoint - redPoint))
        C10 = LerpColor(C010, C110, 1.0 - (upperRedPoint - redPoint))
        C01 = LerpColor(C001, C101, 1.0 - (upperRedPoint - redPoint))
        C11 = LerpColor(C011, C111, 1.0 - (upperRedPoint - redPoint))

        C1 = LerpColor(C01, C11, 1.0 - (upperBluePoint - bluePoint))
        C0 = LerpColor(C00, C10, 1.0 - (upperBluePoint - bluePoint))

        return LerpColor(C0, C1, 1.0 - (upperGreenPoint - greenPoint))

    @staticmethod
    def FromIdentity(cube_size):
        """
        Creates an indentity LUT of specified size.
        """
        identityLattice = empty_lattice_of_size(cube_size)
        ind_01 = indices_01(cube_size)
        for r in range(cube_size):
            for g in range(cube_size):
                for b in range(cube_size):
                    identityLattice[r, g, b] = Color(ind_01[r], ind_01[g], ind_01[b])
        return LUT(identityLattice, name="Identity" + str(cube_size))

    @staticmethod
    def FromLustre3DLFile(lutFilePath):
        lutFile = open(lutFilePath, 'rU')
        lutFileLines = lutFile.readlines()
        lutFile.close()
        outputDepth = -1
        meshLineIndex = 0
        cube_size = -1

        for line in lutFileLines:
            if "Mesh" in line:
                inputDepth = int(line.split()[1])
                outputDepth = int(line.split()[2])
                cube_size = 2 ** inputDepth + 1
                break
            meshLineIndex += 1

        if cube_size == -1 or outputDepth == -1:
            raise NameError("Invalid .3dl file.")

        lattice = empty_lattice_of_size(cube_size)
        currentCubeIndex = 0

        for line in lutFileLines[meshLineIndex + 1:]:
            if len(line) > 0 and len(line.split()) == 3 and "#" not in line:
                # valid cube line
                redValue = line.split()[0]
                greenValue = line.split()[1]
                blueValue = line.split()[2]

                redIndex = int(currentCubeIndex / (cube_size * cube_size))
                greenIndex = int((currentCubeIndex % (cube_size * cube_size)) / cube_size)
                blueIndex = currentCubeIndex % cube_size

                lattice[redIndex, greenIndex, blueIndex] = Color.FromRGBInteger(redValue, greenValue, blueValue,
                                                                                bitdepth=outputDepth)
                currentCubeIndex += 1

        return LUT(lattice, name=os.path.splitext(os.path.basename(lutFilePath))[0])

    @staticmethod
    def FromNuke3DLFile(lutFilePath):
        lutFile = open(lutFilePath, 'rU')
        lutFileLines = lutFile.readlines()
        lutFile.close()

        meshLineIndex = 0
        # lineSkip = 0
        lastOutput = -1
        outputDepth = -1

        for line in lutFileLines:
            if "#" in line or line == "\n":
                meshLineIndex += 1

        cube_size = len(lutFileLines[meshLineIndex].split())  # the input depth

        if cube_size <= 0:
            raise NameError("Invalid .3dl file, unable to resolve cube size.")

        # outputDepth = int(math.log(int(lutFileLines[meshLineIndex].split()[-1])+1,2))
        # check meshline color depth

        for backwardindex in range(1, 99):
            if len(lutFileLines[-backwardindex]) > 0 and \
                    len(lutFileLines[-backwardindex].split()) == 3 and \
                    "#" not in lutFileLines[-backwardindex]:
                lastOutput = int(max(lutFileLines[-backwardindex].split()))
                break
        # the last output should be near the true color depth limit

        for ColorDepth in range(0, 16):
            if lastOutput - (2 ** ColorDepth) < 0:
                outputDepth = ColorDepth
                break
        # double until exceeds last output to test for real color depth

        if outputDepth == -1 or outputDepth == 0:
            raise NameError("Invalid .3dl file, unable to determine output color depth.")

        lattice = empty_lattice_of_size(cube_size)
        currentCubeIndex = 0

        # for line in lutFileLines[meshLineIndex+1:]:
        for line in lutFileLines[meshLineIndex + 1:]:
            # print line
            if len(line) > 0 and len(line.split()) == 3 and "#" not in line:
                # valid cube line
                redValue = line.split()[0]
                greenValue = line.split()[1]
                blueValue = line.split()[2]

                redIndex = int(currentCubeIndex / (cube_size * cube_size))
                greenIndex = int((currentCubeIndex % (cube_size * cube_size)) / cube_size)
                blueIndex = currentCubeIndex % cube_size

                lattice[redIndex, greenIndex, blueIndex] = Color.FromRGBInteger(redValue, greenValue, blueValue,
                                                                                bitdepth=outputDepth)
                currentCubeIndex += 1
        return LUT(lattice, name=os.path.splitext(os.path.basename(lutFilePath))[0])

    @staticmethod
    def FromCubeFile(cubeFilePath):
        cubeFile = open(cubeFilePath, 'rU')
        cubeFileLines = cubeFile.readlines()
        cubeFile.close()

        cube_sizeLineIndex = 0
        cube_size = -1

        for line in cubeFileLines:
            if "LUT_3D_SIZE" in line:
                cube_size = int(line.split()[1])
                break
            cube_sizeLineIndex += 1
        if cube_size == -1:
            raise NameError("Invalid .cube file.")

        lattice = empty_lattice_of_size(cube_size)
        currentCubeIndex = 0
        for line in cubeFileLines[cube_sizeLineIndex + 1:]:
            if len(line) > 0 and len(line.split()) == 3 and "#" not in line:
                # valid cube line
                redValue = float(line.split()[0])
                greenValue = float(line.split()[1])
                blueValue = float(line.split()[2])

                redIndex = currentCubeIndex % cube_size
                greenIndex = int((currentCubeIndex % (cube_size * cube_size)) / cube_size)
                blueIndex = int(currentCubeIndex / (cube_size * cube_size))

                lattice[redIndex, greenIndex, blueIndex] = Color(redValue, greenValue, blueValue)
                currentCubeIndex += 1

        return LUT(lattice, name=os.path.splitext(os.path.basename(cubeFilePath))[0])

    @staticmethod
    def FromFSIDatFile(datFilePath):
        datBytes = bytearray(open(datFilePath, 'r').read())
        cube_size = 64
        lattice = empty_lattice_of_size(cube_size)
        lutBytes = datBytes[128:]
        for currentCubeIndex in range(int(len(lutBytes) / 4)):
            rgb_packed = np.uint32(struct.unpack("<L", lutBytes[currentCubeIndex * 4:(currentCubeIndex * 4) + 4])[0])

            redValue = RemapIntTo01(rgb_packed & 1023, 1008)
            greenValue = RemapIntTo01(rgb_packed >> 10 & 1023, 1008)
            blueValue = RemapIntTo01(rgb_packed >> 20 & 1023, 1008)

            redIndex = currentCubeIndex % cube_size
            greenIndex = int((currentCubeIndex % (cube_size * cube_size)) / cube_size)
            blueIndex = int(currentCubeIndex / (cube_size * cube_size))

            lattice[redIndex, greenIndex, blueIndex] = Color(redValue, greenValue, blueValue)

        return LUT(lattice, name=os.path.splitext(os.path.basename(datFilePath))[0])

    def AddColorToEachPoint(self, color):
        """
        Add a Color value to every lattice point on the cube.
        """
        cube_size = self.cube_size
        newLattice = empty_lattice_of_size(cube_size)
        for r in range(cube_size):
            for g in range(cube_size):
                for b in range(cube_size):
                    newLattice[r, g, b] = self.lattice[r, g, b] + color
        return LUT(newLattice)

    def SubtractColorFromEachPoint(self, color):
        """
        Subtract a Color value to every lattice point on the cube.
        """
        cube_size = self.cube_size
        newLattice = empty_lattice_of_size(cube_size)
        for r in range(cube_size):
            for g in range(cube_size):
                for b in range(cube_size):
                    newLattice[r, g, b] = self.lattice[r, g, b] - color
        return LUT(newLattice)

    def MultiplyEachPoint(self, color):
        """
        Multiply by a Color value or float for every lattice point on the cube.
        """
        cube_size = self.cube_size
        newLattice = empty_lattice_of_size(cube_size)
        for r in range(cube_size):
            for g in range(cube_size):
                for b in range(cube_size):
                    newLattice[r, g, b] = self.lattice[r, g, b] * color
        return LUT(newLattice)

    def __add__(self, other):
        if self.cube_size is not other.cube_size:
            raise NameError("Lattice Sizes not equivalent")

        return LUT(self.lattice + other.lattice)

    def __sub__(self, other):
        if self.cube_size is not other.cube_size:
            raise NameError("Lattice Sizes not equivalent")

        return LUT(self.lattice - other.lattice)

    def __mul__(self, other):
        className = other.__class__.__name__
        if "Color" in className or "float" in className or "int" in className:
            return self.MultiplyEachPoint(other)

        if self.cube_size is not other.cube_size:
            raise NameError("Lattice Sizes not equivalent")

        return LUT(self.lattice * other.lattice)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, lut):
        if isinstance(lut, LUT):
            return self.lattice == lut.lattice
        return NotImplemented

    def __ne__(self, lut):
        result = self.__eq__(lut)
        if result is NotImplemented:
            return result
        return not result

    def Plot(self):
        """
        Plot a LUT as a 3D RGB cube using matplotlib. Stolen from https://github.com/mikrosimage/ColorPipe-tools/tree/master/plotThatLut.
        """

        try:
            import matplotlib
            # matplotlib : general plot
            from matplotlib.pyplot import title, figure
            # matplotlib : for 3D plot
            # mplot3d has to be imported for 3d projection
            import mpl_toolkits.mplot3d
            from matplotlib.colors import rgb2hex
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return

        # for performance reasons lattice size must be 9 or less
        # lut = None
        if self.cube_size > 9:
            lut = self.Resize(9)
        else:
            lut = self

        # init vars
        cube_size = lut.cube_size
        input_range = range(0, cube_size)
        max_value = cube_size - 1.0
        red_values = []
        green_values = []
        blue_values = []
        colors = []
        # process color values
        for r in input_range:
            for g in input_range:
                for b in input_range:
                    # get a value between [0..1]
                    norm_r = r / max_value
                    norm_g = g / max_value
                    norm_b = b / max_value
                    # apply correction
                    res = lut.ColorFromColor(Color(norm_r, norm_g, norm_b))
                    # append values
                    red_values.append(res.r)
                    green_values.append(res.g)
                    blue_values.append(res.b)
                    # append corresponding color
                    colors.append(rgb2hex([norm_r, norm_g, norm_b]))
        # init plot
        fig = figure()
        fig.canvas.set_window_title('pylut Plotter')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_xlim(min(red_values), max(red_values))
        ax.set_ylim(min(green_values), max(green_values))
        ax.set_zlim(min(blue_values), max(blue_values))
        title(self.name)
        # plot 3D values
        ax.scatter(red_values, green_values, blue_values, c=colors, marker="o")
        matplotlib.pyplot.show()
