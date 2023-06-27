"""
Steady-state solvers of the Optical Bloch Equations.
"""
import numpy as np
from importlib.metadata import version

from .sensor import Sensor
from .sensor_utils import *
from .doppler_utils import *
from .slicing.slicing import *
from .sensor_solution import Solution

from typing import Optional, Iterable, Union
try:
    import cupy as cp
except ImportError:
    cupy_imported = False
else:
    cupy_imported = True


def solve_steady_state(
        sensor: Sensor, doppler: bool = False, doppler_mesh_method: Optional[MeshMethod] = None,
        sum_doppler: bool = True, weight_doppler: bool = True, 
        n_slices: Union[int, None] = None,
        ) -> Solution:
    if cupy_imported:
        return solve_steady_state_cupy(
            sensor = sensor, doppler = doppler, doppler_mesh_method = doppler_mesh_method,
            sum_doppler = sum_doppler, weight_doppler = weight_doppler)
    """
    Finds the steady state solution for a system characterized by a sensor.

    If insuffucent system memory is available to solve the system in a single call, 
    system is broken into "slices" of manageable memory footprint which are solved indivudually.
    This slicing behavior does not affect the result.
    Can be performed with or without doppler averging.

    Parameters
    ----------
    sensor : :class:`~.Sensor`
        The sensor for which the solution will be calculated.
    doppler : bool, optional
        Whether to calculate the solution for a doppler-broadened
        gas. If `True`, only uses dopper brodening defined by `kvec` parameters
        for couplings in the `sensoe`, so setting this `True` without `kvec` definitions 
        will have no effect. Default is `False`.
    doppler_mesh_method (dict,optional): 
        If not `None`, should be a dictionary of meshing parameters to be passed 
        to :func:`~.doppler_classes`. See :func:`~.doppler_classes` for more 
        information on supported methods and arguments. If `None, uses the 
        default doppler meshing. Default is `None`.
    sum_doppler : bool
        Whether to average over doppler classes after the solve
        is complete. Setting to `False` will not perform the sum, allowing viewing
        of the weighted results of the solve for each doppler class. In this case,
        an axis will be prepended to the solution for each axis along which doppler
        broadening is computed. Ignored if `doppler=False`. Default is `True`.
    weight_doppler : bool
        Whether to apply weights to doppler solution to perform
        averaging. If `False`, will **not** apply weights or perform a doppler_average,
        regardless of the value of `sum_doppler`. Changing from default intended
        only for internal use. Ignored if `doppler=False` or `sum_doppler=False`. 
        Default is `True`.
    n_slices : int or None, optional
        How many sets of equations to break the full equations into.
        The actual number of slices will be the largest between this value and the minumum
        number of slices to solve the system without a memory error. If `None`, uses the minimum
        number of slices to solve the system without a memory error. Detailed information about 
        slicing behavior can be found in :func:`~.slicing.slicing.matrix_slice`. Default is `None`.

    Notes
    -----
    .. note::
        If decoherence values are not sufficiently populated in the sensor, the resulting
        equations may be singular, resulting in an error in `numpy.linalg`. This error is not
        caught for flexibility, but is likely the culprit for `numpy.linalg` errors encountered
        in steady-state solves.
    
    .. note::
        The solution produced by this function will be expressed using rydiqule's convention
        of converting a density matrix into the real basis and removing the ground state to
        improve numerical stability.

    Returns
    -------
    :class:`~.Solution`
        A bunch-type object contining information about the
        solution. Presently, only attribute "rho" is added to the solution, corresponding
        to the density matrix of the steady state solution. Will include solutions
        to all parameter value combinations if array-like parameters are specified.
        
    Examples
    --------
    A basic solve for a 3-level system would have a "density matrix" solution of size 8 (3^2-1)
    
    >>> s = rq.Sensor(3)
    >>> s.add_coupling((0,1), detuning = 1, rabi_freqency=1)
    >>> s.add_coupling((1,2), detuning = 2, rabi_freqency=2)
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_steady_state(s)
    >>> print(type(sol))
    >>> print(type(sol.rho))
    >>> print(sol.rho.shape)
    <class 'rydiqule.sensor_solution.Solution'>
    <class 'numpy.ndarray'>
    (8,)
    
    Defining an array-like parameter will automatically calculate the density matrix solution
    for every value. Here we use 11 values, resulting in 11 density matrices. The `axis_labels`
    attribute of the solution can clarify which axes are which.
    
    >>> s = rq.Sensor(3)
    >>> det = np.linspace(-1,1,11)
    >>> s.add_coupling((0,1), detuning = det, rabi_freqency=1)
    >>> s.add_coupling((1,2), detuning = 2, rabi_freqency=2)
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_steady_state(s)
    >>> print(sol.rho.shape)
    >>> print(sol.axis_labels)
    (11, 8)
    ['(0,1)_detuning']
    
    >>> s = rq.Sensor(3)
    >>> det = np.linspace(-1,1,11)
    >>> s.add_coupling((0,1), detuning = det, rabi_freqency=1)
    >>> s.add_coupling((1,2), detuning = det, rabi_freqency=2)
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_steady_state(s)
    >>> print(sol.rho.shape)
    >>> print(sol.axis_labels)
    (11, 11, 8)
    ['(0,1)_detuning', '(1,2)_detuning']
    
    If the solve uses doppler broadening, but not averaging for doppler is specified,
    there will be a solution axis corresponding to doppler classes.
    
    >>> s = rq.Sensor(3)
    >>> det = np.linspace(-1,1,11)
    >>> s.add_coupling((0,1), detuning = det, rabi_freqency=1)
    >>> s.add_coupling((1,2), detuning = 2, rabi_freqency=2, kvec=(1,0,0))
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_steady_state(s, doppler=True, sum_doppler=False)
    >>> print(sol.rho.shape)
    >>> print(sol.axis_labels)
    (561, 11, 8)
    ['doppler_0', '(0,1)_detuning']
    
    """
    solution = Solution()

    # relevant sensor-related quantities
    stack_shape = sensor._stack_shape()
    basis_size = sensor.basis_size
    spatial_dim = sensor.spatial_dim()

    # initialize doppler-related quantities
    doppler_axis_shape: Iterable[int] = ()
    dop_classes = None
    doppler_shifts = None
    doppler_axes: Iterable[slice] = ()

    # update doppler-related values
    if doppler:
        dop_classes = doppler_classes(method=doppler_mesh_method)
        doppler_shifts = sensor.get_doppler_shifts()
        doppler_axis_shape = tuple(len(dop_classes) for _ in range(spatial_dim))

        if not sum_doppler:
            doppler_axes = tuple(slice(None) for _ in range(spatial_dim))

    n_slices, out_sol_shape = get_slice_num(basis_size, stack_shape, doppler_axis_shape,
                                                sum_doppler, weight_doppler, n_slices)

    if n_slices > 1:
        print(f"Breaking equations of motion into {n_slices} sets of equations...")
        
    # allocate arrays
    hamiltonians = sensor.get_hamiltonian()
    hamiltonians_time, _ = sensor.get_time_couplings()
    hamiltonians_total = hamiltonians + np.sum(hamiltonians_time, axis=0)
    gamma = sensor.decoherence_matrix()
    sols = np.zeros(out_sol_shape)
    
    # loop over individual slices of hamiltonian
    n_slices_true = sum(1 for _ in matrix_slice(gamma, n_slices=n_slices))

    for i, (idx, H, G) in enumerate(matrix_slice(hamiltonians_total, gamma, n_slices=n_slices)):
    
        if n_slices_true > 1:
            print(f"Solving equation slice {i+1}/{n_slices_true}", end='\r')
        
        full_idx = (*doppler_axes, *idx)
        sols[full_idx] = _solve_hamiltonian_stack(
            H, G, doppler=doppler, dop_classes=dop_classes, sum_doppler=sum_doppler,
            weight_doppler=weight_doppler, doppler_shifts=doppler_shifts,
            spatial_dim=spatial_dim
            )

    # save results to Solution object
    solution.rho = sols
    solution.eta = sensor.eta
    solution.kappa = sensor.kappa
    solution.couplings = sensor.get_couplings()
    solution.axis_labels = ([f'doppler_{i:d}' for i in range(spatial_dim) if not sum_doppler]
                            + sensor.axis_labels()
                            + ["density_matrix"])
    solution.axis_values = ([dop_classes for i in range(spatial_dim) if not sum_doppler]
                            + [val for _,_,val in sensor.variable_parameters()]
                            + [sensor.basis()])
    solution.basis = sensor.basis()
    solution.rq_version = version("rydiqule")
    solution.doppler_classes = dop_classes

    return solution


def _solve_hamiltonian_stack(
        hamiltonians: np.ndarray, gamma_matrix: np.ndarray,
        doppler: bool = False, dop_classes: Optional[np.ndarray] = None,
        sum_doppler: bool = True, weight_doppler: bool = True,
        doppler_shifts: Optional[np.ndarray] = None, spatial_dim: int = 0
        ) -> np.ndarray:
    """
    Solves a the equations of motion corresponding to the given set of hamiltonians.
    
    Typically used as an auxillary function for :meth:`~.solve_steady_state`. Hamiltonian and
    gamma matrices must be of broadcastable shapes. 
    """
    eom, const = generate_eom(hamiltonians, gamma_matrix)

    if doppler:
        assert dop_classes is not None and doppler_shifts is not None
        dop_velocities, dop_volumes = doppler_mesh(dop_classes, spatial_dim)

        eom = get_doppler_equations(eom, doppler_shifts, dop_velocities)

        # this is required for linalg.solve boadcasting to work
        const = np.expand_dims(const, tuple(np.arange(spatial_dim)))
        sols_full = steady_state_solve_stack(eom, const)

        if weight_doppler:
            sols_weighted = apply_doppler_weights(sols_full, dop_velocities, dop_volumes)
            if sum_doppler:
                sum_axes = tuple(np.arange(spatial_dim))
                sols = np.sum(sols_weighted, axis=sum_axes)
            else:
                sols = sols_weighted
        else:
            sols = sols_full

    else:
        sols = steady_state_solve_stack(eom, const)

    return sols


def steady_state_solve_stack(eom: np.ndarray, const: np.ndarray) -> np.ndarray:
    """
    Helper function which returns the solution to the given equations of motion

    Solves an equation of the form :math:`\dot{x} = Ax + b`, or a set of such equations
    arranged into stacks. 
    Essentially just wraps numpy.linalg.solve(), but included as its own 
    function for modularity if another solver is found to be worth invesitigating. 

    Parameters
    ----------
    eom : numpy.ndarray
        An square array of shape `(*l,n,n)` representing the differential
        equations to be solved. The matrix (or matrices) A in the above formula.
    const : numpy.ndarray
        An array or shape `(*l,n)` representing the constant in the matrix form
        of the differential equation. The constant b in the above formula. Stack shape
        `*l` must be consistent with that in the `eom` argument

    Returns
    -------
    numpy.ndarray
        A 1xn array representing the steady-state solution
        of the differential equation
    """
    
    sol = np.linalg.solve(eom, -const)
    return sol


def solve_eom_cupy(cpu_hamiltonian_array: np.ndarray, cpu_gamma_matrix_array: np.ndarray, MaxPieceSize: int = 200000) -> np.ndarray:
    """
    Create the optical bloch equations for a hamiltonian and decoherence matrix
    using the Lindblad master equation.

    """
    
    if not cpu_hamiltonian_array.shape[-2:] == cpu_gamma_matrix_array.shape[-2:]:
        raise ValueError("hamiltonian and gamma matrix must have matching shape")
    if not cpu_hamiltonian_array.shape[-1] == cpu_hamiltonian_array.shape[-2]:
        raise ValueError("hamiltonian and gamma matrix must be square")
    if not len(cpu_hamiltonian_array.shape)==3:
        raise ValueError("hamiltonian and gamma matrix must be serialized")
    
    basis_size = cpu_hamiltonian_array.shape[-1]
    cpu_full_stack_length = cpu_hamiltonian_array.shape[0]
    cpu_full_solutions = np.zeros(shape=(cpu_full_stack_length, basis_size**2-1),dtype=np.float64)

    ListOfStartIndex = [StartIndex for StartIndex in range(0,cpu_full_stack_length,MaxPieceSize)]
    ListOfStartEndIndex = [(StartIndex,min(StartIndex+MaxPieceSize,cpu_full_stack_length)) for StartIndex in ListOfStartIndex]
    for StartEndTupleIndex in ListOfStartEndIndex:
        hamiltonian = cp.asarray(cpu_hamiltonian_array[StartEndTupleIndex[0]:StartEndTupleIndex[1]][:][:])
        gamma_matrix = cp.asarray(cpu_gamma_matrix_array[StartEndTupleIndex[0]:StartEndTupleIndex[1]][:][:])
        
        stack_length = hamiltonian.shape[0]
        
        
        
        U, U_Inverse = get_basis_transform(basis_size)
        gpu_U = cp.asarray(U[1:,1:])
        gpu_U_Inverse = cp.asarray(U_Inverse[1:,1:])
        gpu_rho = cp.zeros((basis_size,basis_size,basis_size,basis_size), dtype=cp.complex128)
        for i in range(basis_size):
            for j in range(basis_size):
                gpu_rho[i,j,i,j] = 1.0 + 0.0j
        
        
        """
        Calculate the first term of the Lindblad master equation.
        
        """
        
        stackshape_11nn = (stack_length, 1, 1, basis_size, basis_size)
        stackshape_111n = (stack_length, 1, 1, 1, basis_size)
        stackshape_n2n2 = (stack_length, basis_size**2, basis_size**2)
        
        hamiltonian_exp = hamiltonian.reshape(stackshape_11nn)
        
        
        hamiltonian_term_super: cp.ndarray = -1j*(hamiltonian_exp @ gpu_rho - gpu_rho @ hamiltonian_exp)
        hamiltonian_term_super = hamiltonian_term_super.reshape(stackshape_n2n2)
        
        """
        Determine the second term of the Lindblad master equation.
    
        """
        
        gamma_exp = gamma_matrix.reshape(stackshape_11nn)
        g = cp.multiply((cp.sum(gamma_matrix, axis=-1)).reshape(stackshape_111n),gpu_rho)
        
        g_T = cp.swapaxes(cp.swapaxes(g,-2,-3),-1,-4)
        
        decoherence_term_super: cp.ndarray = cp.swapaxes((cp.matmul(gpu_rho,cp.matmul(gamma_exp,gpu_rho))), -2, -3) - (g_T + g)/2
        decoherence_term_super = decoherence_term_super.reshape(stackshape_n2n2)
        
        # create optical bloch equations
        ob_equations: cp.ndarray = hamiltonian_term_super + decoherence_term_super
        
        
        """
        Remove the ground state from the equations of motion using population conservation.
    
        """
    
        # basis_size = int(cp.sqrt(ob_equations.shape[-1]))
        eqn_size = ob_equations.shape[-1]
    
        # get the constant term
        eqns_column1 = ob_equations[:,:,0]
        constant_term = ob_equations[:,1:,0]
    
        # find the indices where populations need to be subtracted
        plocations = cp.asarray([(basis_size+1)*x for x in range(basis_size)])
        pvector = cp.asarray([int(i in plocations) for i in range(eqn_size)])
        
        # make a matrix to subtract populations
        pop_subtract = cp.einsum('ij,k', eqns_column1, pvector)
        
        # subtract populations
        equation_new = ob_equations - pop_subtract
    
        # remove the ground state
        equations_reduced = equation_new[:, 1:, 1:]
    
        """
        Converts equations of motion from complex basis to real basis.
    
        """
        # Define the basis for printout purposes for ground removed or not removed
    
    
        # transform to the real basis
        gpu_Eom = cp.real(cp.matmul(gpu_U,cp.matmul(equations_reduced,gpu_U_Inverse)))
        # print('gpu_U.shape=' + str(gpu_U.shape))
        # print('constant_term.shape=' + str(constant_term.shape))
        
        # gpu_Const = cp.real(cp.matmul(gpu_U, constant_term))
        gpu_Const = cp.real(cp.einsum('jk,ik',gpu_U, constant_term))
        
        gpu_Sol = cp.linalg.solve(gpu_Eom, -gpu_Const)
        cpu_Sol = cp.asnumpy(gpu_Sol)
        cpu_full_solutions[StartEndTupleIndex[0]:StartEndTupleIndex[1]][:] = cpu_Sol[:][:]
        
        
    return cpu_full_solutions



def solve_steady_state_cupy(
        sensor: Sensor, doppler: bool = False, doppler_mesh_method: Optional[MeshMethod] = None,
        sum_doppler: bool = True, weight_doppler: bool = True):
    solution = Solution()

    # relevant sensor-related quantities
    stack_shape = tuple(sensor._stack_shape())
    basis_size = sensor.basis_size
    spatial_dim = sensor.spatial_dim()
    
    # initialize doppler-related quantities
    doppler_axis_shape: Iterable[int] = ()
    dop_classes = None
    doppler_shifts = None
    doppler_axes: Iterable[slice] = ()
    
    # update doppler-related values
    if doppler:
        dop_classes = doppler_classes(method=doppler_mesh_method)
        doppler_shifts = sensor.get_doppler_shifts()
        doppler_axis_shape = tuple(len(dop_classes) for _ in range(spatial_dim))

        if not sum_doppler:
            doppler_axes = tuple(slice(None) for _ in range(spatial_dim))

    
    # allocate arrays
    hamiltonians_constant = sensor.get_hamiltonian()
    hamiltonians_time, _ = sensor.get_time_couplings()
    H =  hamiltonians_constant + np.sum(hamiltonians_time, axis=0)
    G =  sensor.decoherence_matrix()
    
    sol_basis_size = basis_size**2-1
    sol_basis_size_tuple = tuple([sol_basis_size])
    
    if doppler and weight_doppler and not sum_doppler:    
        out_sol_shape = doppler_axis_shape + stack_shape + sol_basis_size_tuple
    else:
        out_sol_shape = stack_shape + sol_basis_size_tuple
    
    if doppler:
        dop_classes = doppler_classes(method=doppler_mesh_method)
        doppler_shifts = sensor.get_doppler_shifts()
        dop_volumes = np.gradient(dop_classes)  # smoothly handles irregular arrays
        # generate the velocity meshgrids
        dets = [dop_classes for _ in range(spatial_dim)]
        diffs = [dop_volumes for _ in range(spatial_dim)]
        H_flattened = np.reshape(H,newshape=(int(round(np.prod(H.shape[:-2]))),basis_size,basis_size))
        G_flattened = np.reshape(G,newshape=(int(round(np.prod(G.shape[:-2]))),basis_size,basis_size))
        
        # Create array of all possible velocity vector combinations:
        #       shape = (len(doppler_classes)**spatial_dim,spatial_dim)
        Vs_serialized_array = []
        if len(dets)==1:
            for i in dop_classes:
                Vs_serialized_array.append([i])
        elif len(dets)==2:
            for i in dop_classes:
                for j in dop_classes:
                    Vs_serialized_array.append([i,j])
        elif len(dets)==3:
            for i in dop_classes:
                for j in dop_classes:
                    for k in dop_classes:
                        Vs_serialized_array.append([i,j,k])
        Vs_serialized = np.array(Vs_serialized_array)
        
        # Create array of summed hamiltonians times their 
        # respective velocity components using Einstein summation.
        # This takes a doppler list of velocity vectors:
        #       shape = (len(doppler_classes)**spatial_dim,spatial_dim)
        # And a hamiltonians for every spatial dimension:
        #       shape = (spatial_dim,basis_size,basis_size)
        # And creates a summed hamiltonian for all velocity vectors:
        #       shape = (len(doppler_classes)**spatial_dim,basis_size,basis_size)
        # This maintains the ordering for reshaping to expected output.
        Vs_H_flattened = np.einsum(
            'ij,jkl->ikl', 
            Vs_serialized, 
            doppler_shifts)
        
        # Get Cartesian product and add all possible combinations of 
        # Hamiltonians based on Doppler shift and other parameters:
        hamiltonians_serialized= np.einsum('ijkl->ikl', \
            np.array([i for i in itertools.product(
                Vs_H_flattened, H_flattened)]))
        gamma_matrix_serialized_unflattened = np.array([G_flattened for _ in range(len(Vs_H_flattened))])
        gamma_matrix_serialized = np.reshape(gamma_matrix_serialized_unflattened,newshape=hamiltonians_serialized.shape)
        # doppler_axis_length = len(Vs)
        # base_doppler_shift_eoms = generate_doppler_shift_eom(doppler_hamiltonians)
        sols_serialized = solve_eom_cupy(hamiltonians_serialized, gamma_matrix_serialized)
        sols_doppler_shaped = sols_serialized.reshape(
            (len(Vs_H_flattened),len(H_flattened),basis_size**2-1))
        if weight_doppler:
            # Vs_meshgrid, H_meshgrid = np.meshgrid(Vs,H,indexing='ij'])
            # Get Cartesian product and add all possible combinations of weights:
            Vols_tuples_serialized = np.array([i for i in itertools.product(*diffs)]) # np.array(np.meshgrid(*diffs,indexing="ij"))
            
            
            # Get gaussian3d function in serialized output form
            spatial_dim = Vs_serialized.shape[-1]
            if spatial_dim > 3:
                raise ValueError(f"Too many axes supplied: {spatial_dim}")

            prefactor = np.power(1/(np.pi),spatial_dim*0.5)
            squared_mag_array = np.square(Vs_serialized).sum(axis=-1)
            weights_serialized = prefactor*np.exp(-squared_mag_array)
            
            # Get elementwise product of all the weight dimensions
            Vols_serialized = [np.prod(i) for i in Vols_tuples_serialized]
            weighted_vols = Vols_serialized*weights_serialized
            sols_weighted = np.einsum(
                'ijk,i->ijk', 
                sols_doppler_shaped, 
                weighted_vols)
            if sum_doppler:
                sols = np.sum(sols_weighted, axis=0).reshape(out_sol_shape)
            else:
                sols = sols_weighted.reshape(out_sol_shape)
        else:
            sols = sols_serialized.reshape(out_sol_shape)
    else:
        hamiltonians_serialized = np.reshape(H,newshape=(int(round(np.prod(H.shape[:-2]))),basis_size,basis_size))
        gamma_matrix_serialized = np.reshape(G,newshape=(int(round(np.prod(G.shape[:-2]))),basis_size,basis_size))
        sols_flattened = solve_eom_cupy(hamiltonians_serialized, gamma_matrix_serialized)
        sols = sols_flattened.reshape(out_sol_shape)
    solution.rho = sols
    solution.eta = sensor.eta
    solution.kappa = sensor.kappa
    solution.couplings = sensor.get_couplings()
    solution.axis_labels = ([f'doppler_{i:d}' for i in range(spatial_dim) if not sum_doppler]
                            + sensor.axis_labels()
                            + ["density_matrix"])
    solution.axis_values = ([dop_classes for i in range(spatial_dim) if not sum_doppler]
                            + [val for _,_,val in sensor.variable_parameters()]
                            + [sensor.basis()])
    solution.basis = sensor.basis()
    solution.rq_version = version("rydiqule")
    solution.doppler_classes = dop_classes
    
    return solution
