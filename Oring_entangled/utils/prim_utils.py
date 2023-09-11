# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# python
import typing

# omniverse
from pxr import UsdGeom, Usd
import omni.usd
from omni.usd.commands import  DeletePrimsCommand


# isaacsim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.semantics import add_update_semantics


def delete_prim(prim_path: str) -> None:
    """Remove the USD Prim and its decendants from the scene if able

    Args:
        prim_path (str): path of the prim in the stage
    """
    # DeletePrimsCommand(paths=[prim_path], destructive=False).do() # chan
    from omni.usd.commands import  DeletePrimsCommand

    DeletePrimsCommand(paths=[prim_path], destructive=True).do() # chan
