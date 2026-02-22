"""Provides the :class:`.Model` class, which can store a model of Minecraft blocks."""


from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING

import numpy as np
from nbt.nbt import NBTFile, TAG_Byte, TAG_Compound, TAG_List, TAG_String
from pyglm.glm import bvec3, ivec3

from .block import Block
from .vector_tools import Box, Vec3bLike, Vec3iLike, loop3D


if TYPE_CHECKING:
    from nbt.nbt import NBTFile
    from numpy.typing import NDArray

    from .editor import Editor
    from .transform import TransformLike


class Model:
    """A 3D model of Minecraft blocks.

    Can be used to store a structure in memory, allowing it to be built under different
    transformations.
    """

    def __init__(self, size: Vec3iLike, blocks: list[Block | None] | None = None) -> None:
        """Constructs a Model of size ``size``, optionally filled with ``blocks``."""
        # Original signature is maintained for parity with original Model
        self._palette: list[Block | None] = [None]  # The first palette entry is always None
        self._matrix: NDArray[np.uint8] = np.zeros(shape=(size[0], size[1], size[2]), dtype=np.uint8)

        if blocks is None:
            return

        for x, y, z in loop3D(size):
            block = blocks[(x * size[1] + y) * size[2] + z]
            if block not in self._palette:
                self._palette.append(block)
            block_index = self._palette.index(block)
            self._matrix[x, y, z] = block_index

    @classmethod
    def fromNBT(cls, nbt_file: NBTFile) -> Model:
        """Constructs a PaletteModel from an NBT file."""
        # NOTE: A lot of type checking gets ignored here, since the NBT file is expected to have a very specific structure, and the nbt library doesn't have great type annotations. If the NBT file doesn't have the expected structure, this function will likely raise an error.
        # extract relevant information from the NBT file

        size = ivec3(
            nbt_file["size"][0].value, nbt_file["size"][1].value, nbt_file["size"][2].value, # type: ignore
        )

        palette: list[Block | None] = [
            None,
            *[
                Block(
                    block["Name"].value, # type: ignore
                    (
                        {tag.name: tag.value for tag in block["Properties"].tags} # type: ignore
                        if "Properties" in block
                        else {}
                    ),
                )
                for block in nbt_file["palette"] # type: ignore
            ],
        ]

        palettemodel = cls(size)

        palettemodel._palette = palette
        matrix = np.zeros(shape=(size.x, size.y, size.z), dtype=np.uint8)

        for block in nbt_file["blocks"]: # type: ignore
            pos = block["pos"] # type: ignore
            x, y, z = pos[0].value, pos[1].value, pos[2].value # type: ignore
            state = block["state"].value # type: ignore
            matrix[x, y, z] = (
                state + 1
            )  # state is 0-indexed, but our palette is 1-indexed, since the first entry is None

        palettemodel._matrix = matrix
        return palettemodel

    def toNBT(self) -> NBTFile:
        """Dumps this Model to an NBT file."""
        nbt_file = NBTFile()
        nbt_file.name = "PaletteModel"

        # size
        size_list = TAG_List(name="size", type=TAG_Byte)
        size_list.tags.append(TAG_Byte(int(self.size.x)))
        size_list.tags.append(TAG_Byte(int(self.size.y)))
        size_list.tags.append(TAG_Byte(int(self.size.z)))
        nbt_file.tags.append(size_list)

        # palette
        palette_list = TAG_List(name="palette", type=TAG_Compound)
        for block in self._palette[1:]:  # Skip the first entry, which is always None
            block_tag = TAG_Compound()
            block_tag["Name"] = TAG_String(block.id if block else "")
            if block and block.states:
                properties_tag = TAG_Compound()
                for key, value in block.states.items():
                    properties_tag[key] = TAG_String(value)
                block_tag["Properties"] = properties_tag
            palette_list.tags.append(block_tag)
        nbt_file.tags.append(palette_list)

        # blocks
        blocks_list = TAG_List(name="blocks", type=TAG_Compound)
        for x, y, z in loop3D(self.size):
            block_index = int(self._matrix[x, y, z])
            if block_index == 0:
                continue  # Skip air blocks
            block_tag = TAG_Compound()
            pos_list = TAG_List(name="pos", type=TAG_Byte)
            pos_list.tags.append(TAG_Byte(int(x)))
            pos_list.tags.append(TAG_Byte(int(y)))
            pos_list.tags.append(TAG_Byte(int(z)))
            block_tag["pos"] = pos_list
            block_tag["state"] = TAG_Byte(block_index - 1)  # Convert back to 0-indexed
            blocks_list.tags.append(block_tag)
        nbt_file.tags.append(blocks_list)

        return nbt_file

    @property
    def size(self) -> ivec3:
        """Returns the size of this model."""
        return ivec3(*self._matrix.shape)

    @property
    def blocks(self) -> list[Block | None]:
        """Returns a list of the blocks in this model, in the same order as the constructor's ``blocks`` argument."""
        blocks_list: list[Block | None] = []
        for x, y, z in loop3D(self._matrix.shape):
            blockIndex = self._matrix[x, y, z]
            try:
                block: Block | None = self._palette[blockIndex]
                blocks_list.append(block)
            except IndexError:
                blocks_list.append(None)
        return blocks_list

    def transform(self, rotation: int = 0, flip: Vec3bLike | None = None) -> None:
        """Transforms this model.\n
        Flips first, rotates second."""

        if flip is None:
            flip = bvec3()

        # Transform the palette
        for block in self._palette:
            if block is not None:
                block.transform(rotation, flip)

        # Transform the matrix
        if flip[0]:
            self._matrix = np.flip(self._matrix, axis=0)
        if flip[1]:
            self._matrix = np.flip(self._matrix, axis=1)
        if flip[2]:
            self._matrix = np.flip(self._matrix, axis=2)
        self._matrix = np.rot90(self._matrix, k=-rotation, axes=(0, 2))  # Rotate clockwise around the y-axis


    def transformed(self, rotation: int = 0, flip: Vec3bLike | None = None) -> Model:
        """Returns a transformed copy of this model.\n
        Flips first, rotates second."""
        model = deepcopy(self)
        model.transform(rotation, flip)
        return model

    def getBlock(self, position: Vec3iLike) -> Block | None:
        """Returns the block at the given position in this model, or None if there is no block at that position."""
        block_index = self._matrix[tuple(position)]
        try:
            return self._palette[block_index]
        except IndexError:
            return None

    def setBlock(self, position: Vec3iLike, block: Block | None) -> None:
        """Sets the block at the given position in this model to the given block."""
        if block not in self._palette:
            self._palette.append(block)
        block_index = self._palette.index(block)
        self._matrix[tuple(position)] = block_index

    def build(
        self,
        editor:         Editor,
        transformLike:  TransformLike | None         = None,
        substitutions:  dict[str, str] | None = None,
        replace:        str | list[str] | None       = None,
    ) -> None:
        """Builds the model.

        Use ``substitutions`` to build the model with certain blocks replaced by others.
        Use ``replace`` to specify which blocks in the world should be replaced by the model's blocks.

        """
        if substitutions is None: substitutions = {}

        with editor.pushTransform(transformLike):
            for vec in Box(size=self.size):
                block = self.getBlock(vec)
                if block is not None:
                    blockToPlace = copy(block)
                    blockToPlace.id = substitutions.get(block.id, block.id)
                    editor.placeBlock(vec, blockToPlace, replace)

    def __repr__(self) -> str:
        """Returns a string representation that is guaranteed to `eval()` to this Model."""
        return f"Model(size={repr(self.size)}, blocks={repr(self.blocks)})"
