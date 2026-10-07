.. _base_components:

====================================
Base Classes
====================================

For extendability and modularity purposes, ``qlbm`` provides several base classes
with interfaces that aim to make the development of novel QLBM methods easier.
This page documents the base classes that have to do with the register
setup and quantum circuit generation.
The architecture of the inheritance hierarchy is two-fold:
vertically based on specificity (i.e., more specialized classes for specific QLBMs)
and horizontally based on methods (i.e., different implementations for different QLBMs).

.. _lattice_bases:

Lattice Base
----------------------------------

.. autoclass:: qlbm.lattice.lattices.base.Lattice
    :members:

.. autoclass:: qlbm.lattice.lattices.base.AmplitudeLattice
    :members:

.. autoclass:: qlbm.lattice.geometry.shapes.base.Shape
    :members:

.. _components_bases:

Components Base
----------------------------------

Every component is a qarp block: leaves extend :class:`qarp.blocks.SimpleBlock`
and trees extend :class:`qarp.blocks.CompositeBlockBase`, so components compose
with ``ControlledBlock``, ``~``, ``**`` and the qarp engines directly.

.. autoclass:: qlbm.components.base.LBMPrimitive
    :members:

.. autoclass:: qlbm.components.base.LatticePrimitive

.. autoclass:: qlbm.components.base.LBMComposite
    :members:

.. autoclass:: qlbm.components.base.ControllableComponent
    :members:

.. autoclass:: qlbm.components.base.SequenceBlock
    :members:

.. autoclass:: qlbm.components.base.LBMOperator

.. autoclass:: qlbm.components.base.LBMAlgorithm

.. autofunction:: qlbm.components.base.on

.. autofunction:: qlbm.components.base.x_layer

.. autofunction:: qlbm.components.base.controlled

.. autofunction:: qlbm.components.base.flip_if
