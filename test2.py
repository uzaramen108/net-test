"""
FSI: ALE Navier-Stokes + Poisson Membrane + Darcy Permeation + Grid Support (LVPP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기반: Document 12 (구 없는 FSI)
추가:
  ① 막 뒤 스테인리스 정사각형 격자 (Brinkman no-slip)
  ② 막 변위 상한 LVPP: w = φ_grid - e^ψ ≤ φ_grid 항상 만족
     (하부 장애물 부호 반전: +e^ψ·η 사용)
  ③ LVPP 초기화: ψ₀ = ln(φ_grid) → w₀ = 0 (초기 변위 0 보장)
"""

import numpy as np
import gmsh
import basix
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path
from scipy.spatial import cKDTree

from dolfinx import io
from dolfinx.mesh import create_submesh, exterior_facet_indices, GhostMode, create_cell_partitioner
from dolfinx.fem import (
    Constant, Function, functionspace, form,
    dirichletbc, locate_dofs_topological, locate_dofs_geometrical,
)
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector, apply_lifting,
    create_vector, set_bc, create_matrix, NonlinearProblem,
)
from dolfinx.io import VTXWriter
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from basix.ufl import element as belem
import ufl
from ufl import (
    Identity, TestFunction, TrialFunction, FacetNormal,
    div, dot, inner, lhs, nabla_grad, rhs, sym,
    avg, Measure, grad, SpatialCoordinate, exp, split, TestFunctions,
)

# ═══════════════════════════════════════════════════════════════
# § 0. Parameters
# ═══════════════════════════════════════════════════════════════
L, R, x_m  = 5.0, 0.5, 3.75
T_sim, N   = 1.5, 1500
dt         = T_sim / N

rho_v    = 1.0
mu_v     = 0.01
kappa_v  = 5e-3
T_mem    = 25.0
u_max    = 5.0
DK       = mu_v / kappa_v
mem_lc   = 0.04
EPS_BASE = 0.03

# ── 격자 파라미터 (SS 316L 용접 스크린, ×200 스케일) ──────────
# 실제: 와이어 Ø0.15 mm, 피치 1.0 mm → 개구율 ~64%
# 시뮬: 와이어 반지름 0.015 m, 피치 0.10 m → 개구율 ~49%
a_grid = 0.10    # 격자 피치 [m]
r_wire = 0.03   # 와이어 반지름 [m]
d_grid = 0.04    # 막 → 격자 중심 거리 [m]
x_grid = x_m + d_grid
w_bar  = 0.03   # 기둥의 y 또는 z 방향 반폭 [m] (기존 r_wire와 동일)
t_bar  = 0.03   # 기둥의 x 방향 반두께 [m] (기존 r_wire와 동일)

# Brinkman 페널티 (격자 내부 u → 0)
eta_wire = mu_v / 1e-6 # 1e-7 -> 1e-6 (격자 내부 유체 거의 고정)

# LVPP 파라미터 (상부 장애물)
LVPP_MAX  = 10       # 25 → 10
LVPP_TOL  = 1e-5     # 1e-6 → 1e-5 (막 변위 정밀도 충분)
LVPP_C    = 1.0
LVPP_R    = 1.5
LVPP_Q    = 1.5
LVPP_AMAX = 1e5

# ═══════════════════════════════════════════════════════════════
# § 1. Mesh  (격자 근처 세밀화 추가)
# ═══════════════════════════════════════════════════════════════
gmsh.initialize()
gmsh.model.add("fsi")
if MPI.COMM_WORLD.rank == 0:
    c1 = gmsh.model.occ.addCylinder(0,   0, 0, x_m,     0, 0, R)
    c2 = gmsh.model.occ.addCylinder(x_m, 0, 0, L - x_m, 0, 0, R)
    dk = gmsh.model.occ.addDisk(x_m, 0, 0, R, R)
    gmsh.model.occ.rotate([(2, dk)], x_m, 0, 0, 0, 1, 0, np.pi / 2)
    out_tags, out_map = gmsh.model.occ.fragment([(3, c1), (3, c2)], [(2, dk)])
    gmsh.model.occ.synchronize()

    mem_surfs = [t for d, t in out_map[2] if d == 2]
    vol_tags  = [t for d, t in out_tags   if d == 3]
    bnd = gmsh.model.getBoundary(
        [(3, t) for t in vol_tags], oriented=False, combined=True)
    In, Out, Wall = [], [], []
    for d, tag in bnd:
        if d != 2 or tag in mem_surfs: continue
        cx = gmsh.model.occ.getCenterOfMass(d, tag)[0]
        if   np.isclose(cx, 0., atol=.15): In.append(tag)
        elif np.isclose(cx, L,  atol=.15): Out.append(tag)
        else:                               Wall.append(tag)

    gmsh.model.addPhysicalGroup(3, vol_tags,  tag=1, name="fluid")
    gmsh.model.addPhysicalGroup(2, In,        tag=1, name="inlet")
    gmsh.model.addPhysicalGroup(2, Out,       tag=2, name="outlet")
    gmsh.model.addPhysicalGroup(2, Wall,      tag=3, name="wall")
    gmsh.model.addPhysicalGroup(2, mem_surfs, tag=4, name="membrane")

    # 격자 위치 근처 메시 세밀화 (막과 십자 기둥이 만나는 타깃 구역 촘촘하게!)
    # 격자 위치 근처 메시 세밀화 (막과 십자 기둥이 만나는 타깃 구역 유지!)
    # Field 1: 막 평면만 얇게 (두께 0.01m, h=0.008 유지)
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "XMin", x_m - 0.005)
    gmsh.model.mesh.field.setNumber(1, "XMax", x_m + 0.005)  # ← 두께 1cm만
    gmsh.model.mesh.field.setNumber(1, "YMin", -R)
    gmsh.model.mesh.field.setNumber(1, "YMax",  R)
    gmsh.model.mesh.field.setNumber(1, "ZMin", -R)
    gmsh.model.mesh.field.setNumber(1, "ZMax",  R)
    gmsh.model.mesh.field.setNumber(1, "VIn",  0.008)
    gmsh.model.mesh.field.setNumber(1, "VOut", 0.10)

    # Field 2: 기둥 Brinkman 영역 (h=0.03)
    gmsh.model.mesh.field.add("Box", 2)
    gmsh.model.mesh.field.setNumber(2, "XMin", x_grid - t_bar - 0.01)
    gmsh.model.mesh.field.setNumber(2, "XMax", x_grid + t_bar + 0.01)
    gmsh.model.mesh.field.setNumber(2, "YMin", -R)
    gmsh.model.mesh.field.setNumber(2, "YMax",  R)
    gmsh.model.mesh.field.setNumber(2, "ZMin", -R)
    gmsh.model.mesh.field.setNumber(2, "ZMax",  R)
    gmsh.model.mesh.field.setNumber(2, "VIn",  0.03)
    gmsh.model.mesh.field.setNumber(2, "VOut", 0.10)

    # 두 필드 Min 합성
    gmsh.model.mesh.field.add("Min", 3)
    gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])
    gmsh.model.mesh.field.setAsBackgroundMesh(3)

    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.12)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005)
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    
    print("메시 생성 시작...", flush=True)
    gmsh.model.mesh.generate(3)
    print("메시 최적화 시작...", flush=True)
    gmsh.model.mesh.optimize("Netgen")

# 1. 병렬 연산 시 면(facet)을 공유하게 만드는 파티셔너 생성
partitioner = create_cell_partitioner(GhostMode.shared_facet)

# 2. 메시 변환 (ghost_mode 대신 partitioner 사용)
gdata = io.gmsh.model_to_mesh(
    gmsh.model, 
    MPI.COMM_WORLD, 
    0, 
    gdim=3,
    partitioner=partitioner
)
msh, ct, ft = gdata.mesh, gdata.cell_tags, gdata.facet_tags
gmsh.finalize()
gdim = msh.geometry.dim

# ═══════════════════════════════════════════════════════════════
# § 2. Membrane 2D submesh
# ═══════════════════════════════════════════════════════════════
sub_m, emap_m, vmap_m, geom_map_raw = create_submesh(
    msh, msh.topology.dim - 1, ft.find(4))
geom_map = np.array(geom_map_raw, dtype=np.int32)

# ═══════════════════════════════════════════════════════════════
# § 3. Function spaces
# ═══════════════════════════════════════════════════════════════
V_f   = functionspace(msh, belem("Lagrange", msh.basix_cell(), 2, shape=(gdim,)))
Q_f   = functionspace(msh, belem("Lagrange", msh.basix_cell(), 1))
V_ale = functionspace(msh, belem("Lagrange", msh.basix_cell(), 1, shape=(gdim,)))

# DG0: 격자 Brinkman 지시자 (chi_grid=1 → 와이어 내부)
V_ind    = functionspace(msh, belem("DG", msh.basix_cell(), 0))
chi_grid = Function(V_ind, name="grid_indicator")

# LVPP 혼합 공간 (w, ψ): 상부 장애물 w = φ_grid - e^ψ ≤ φ_grid
P_m       = basix.ufl.element("Lagrange", sub_m.basix_cell(), 1)
V_m_mixed = functionspace(sub_m, basix.ufl.mixed_element([P_m, P_m]))
V_m0, sub0_to_parent = V_m_mixed.sub(0).collapse()

# φ_grid 상부 장애물 함수 공간
V_phi = functionspace(sub_m, belem("Lagrange", sub_m.basix_cell(), 1))

# ═══════════════════════════════════════════════════════════════
# § 4. Integration measures
# ═══════════════════════════════════════════════════════════════
dxf  = Measure("dx", domain=msh,   subdomain_data=ct)
dS_m = Measure("dS", domain=msh,   subdomain_data=ft, subdomain_id=4)
dxm  = Measure("dx", domain=sub_m)
n    = FacetNormal(msh)

# ═══════════════════════════════════════════════════════════════
# § 5. Boundary conditions
# ═══════════════════════════════════════════════════════════════
fdim = msh.topology.dim - 1

class InletVelocity:
    def __init__(self): self.t = 0.0
    def __call__(self, x):
        v  = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        r2 = x[1]**2 + x[2]**2
        v[0] = (u_max * np.sin(np.pi * self.t / (T_sim * 2))
                * np.maximum(1 - r2 / R**2, 0.))
        return v

iv   = InletVelocity()
u_in = Function(V_f)
u_in.interpolate(iv)

bcu = [
    dirichletbc(u_in, locate_dofs_topological(V_f, fdim, ft.find(1))),
    dirichletbc(np.zeros(gdim, dtype=PETSc.ScalarType),
        locate_dofs_topological(V_f, fdim, ft.find(3)), V_f),
]
bcp = [
    dirichletbc(Constant(msh, PETSc.ScalarType(0.)),
        locate_dofs_topological(Q_f, fdim, ft.find(2)), Q_f),
]

# 막 테두리 w=0 (혼합 공간 sub(0) 적용)
sub_m.topology.create_connectivity(1, 2)
bnd_facets_m = exterior_facet_indices(sub_m.topology)
edge_dofs_m  = locate_dofs_topological((V_m_mixed.sub(0), V_m0), 1, bnd_facets_m)
w_bc_fn = Function(V_m0); w_bc_fn.x.array[:] = 0.
bc_wm   = dirichletbc(w_bc_fn, edge_dofs_m, V_m_mixed.sub(0))

bc_ale_fixed = dirichletbc(
    np.zeros(gdim, dtype=PETSc.ScalarType),
    locate_dofs_geometrical(V_ale, lambda x: (
        np.isclose(x[0], 0., atol=.05) |
        np.isclose(x[0], L,  atol=.05) |
        np.isclose(np.sqrt(x[1]**2 + x[2]**2), R, atol=.05)
    )), V_ale)

# ═══════════════════════════════════════════════════════════════
# § 6. NS 약형 (IPCS + Brinkman 격자)
# ═══════════════════════════════════════════════════════════════
def eps_(u): return sym(nabla_grad(u))
def sig_(u, p): return 2. * mu_v * eps_(u) - p * Identity(gdim)

u_t, vf  = TrialFunction(V_f), TestFunction(V_f)
p_t, qf  = TrialFunction(Q_f), TestFunction(Q_f)
u_n, p_n = Function(V_f), Function(Q_f)
u_,  p_  = Function(V_f), Function(Q_f)
u_.name, p_.name = "Velocity", "Pressure"

w_msh    = Function(V_f)
U        = 0.5 * (u_n + u_t)
ETA_WIRE = Constant(msh, PETSc.ScalarType(eta_wire))

# Brinkman 항: chi_grid=1 셀에서 u → 0 (격자 no-slip, 고정 벽 동일)
F1 = (
    rho_v / dt * inner(u_t - u_n, vf) * dxf
  + rho_v * inner(dot(u_n - w_msh, nabla_grad(u_n)), vf) * dxf
  + inner(sig_(U, p_n), eps_(vf)) * dxf
  + DK * dot(avg(U), n("+")) * dot(avg(vf), n("+")) * dS_m
  + ETA_WIRE * chi_grid * dot(U, vf) * dxf   # 격자 Brinkman (타깃=0)
)
a1_ufl = form(lhs(F1))
L1_ufl = form(rhs(F1))

a2_ufl = form(dot(nabla_grad(p_t), nabla_grad(qf)) * dxf)
L2_ufl = form(dot(nabla_grad(p_n), nabla_grad(qf)) * dxf
            - rho_v / dt * div(u_) * qf * dxf)

a3_ufl = form(rho_v * dot(u_t, vf) * dxf)
L3_ufl = form(rho_v * dot(u_, vf) * dxf
            - dt * dot(nabla_grad(p_ - p_n), vf) * dxf)

# ═══════════════════════════════════════════════════════════════
# § 7. LVPP 막 약형 — 상부 장애물: w ≤ φ_grid
#
#  혼합 공간 (w, ψ):  w = φ_grid - e^ψ  →  항상 w ≤ φ_grid
#
#  w 방정식: α T ∫∇w·∇v + ∫ψ·v − α∫[[p]]·v − ∫ψ_k·v = 0
#  ψ 방정식: ∫w·η + ∫e^ψ·η − ∫φ_grid·η = 0
#            → w + e^ψ = φ_grid  ✓
#
#  하부 장애물(구)와 비교:  -∫e^ψ·η (하부)  vs  +∫e^ψ·η (상부)
# ═══════════════════════════════════════════════════════════════

def pressure_jump_dynamic(p_fn, w_arr, gtree_cur):
    """
    막 전후 압력 점프 [[p]] 계산 → LVPP 압력 하중
    V_phi 공간 기준 (DOF 불일치 방지)
    """
    w_loc = np.zeros(n_local_phi)
    ok_map = np.where(valid_phi_m0)[0]
    w_loc[ok_map] = w_arr[idx_phi_to_m0[ok_map]]

    y_c = phi_dof_local[:, 1]
    z_c = phi_dof_local[:, 2]
    cx  = phi_dof_local[:, 0] + w_loc

    if n_local_phi > 0:
        yz      = np.column_stack([y_c, z_c])
        tree_m  = cKDTree(yz)
        all_nbr = tree_m.query_ball_point(yz, r=3.0 * mem_lc)
    else:
        all_nbr = []

    dw_dy = np.zeros(n_local_phi); dw_dz = np.zeros(n_local_phi)
    for i, raw in enumerate(all_nbr):
        nbrs = np.array([j for j in raw if j != i])
        if not len(nbrs): continue
        dy = y_c[nbrs]-y_c[i]; dz = z_c[nbrs]-z_c[i]; dw = w_loc[nbrs]-w_loc[i]
        syy=dy@dy; szz=dz@dz; syz=dy@dz; syw=dy@dw; szw=dz@dw
        det = syy*szz - syz*syz
        if abs(det) > 1e-20:
            dw_dy[i] = (szz*syw - syz*szw) / det
            dw_dz[i] = (syy*szw - syz*syw) / det

    nx = np.ones(n_local_phi); ny = -dw_dy; nz = -dw_dz
    mag = np.maximum(np.sqrt(nx**2+ny**2+nz**2), 1e-14)
    nx /= mag; ny /= mag; nz /= mag

    pm_l = np.zeros(n_local_phi); pp_l = np.zeros(n_local_phi); fd_l = np.zeros(n_local_phi)
    for sign, arr in [(-1, pm_l), (+1, pp_l)]:
        if n_local_phi == 0: continue
        pts = np.column_stack([
            cx  + sign*EPS_BASE*nx,
            y_c + sign*EPS_BASE*ny,
            z_c + sign*EPS_BASE*nz,
        ])
        r   = np.sqrt(pts[:,1]**2 + pts[:,2]**2)
        out = r > R*0.92
        pts[out,1] *= R*0.92/r[out]; pts[out,2] *= R*0.92/r[out]
        pts[:,0] = np.clip(pts[:,0], 0.01, L-0.01)

        coll  = compute_collisions_points(gtree_cur, pts)
        col_c = compute_colliding_cells(msh, coll, pts)
        cells = np.array([
            col_c.links(j)[0] if len(col_c.links(j))>0 else -1
            for j in range(n_local_phi)], dtype=np.int32)
        ok = cells >= 0
        if ok.any():
            arr[ok] = p_fn.eval(pts[ok].reshape(-1,3), cells[ok]).reshape(-1)
        if sign == -1:
            fd_l[:] = (cells >= 0).astype(float)

    g0p, g1p = local_range_phi
    pm_g = np.zeros(n_global_phi); pm_g[g0p:g1p] = pm_l
    pp_g = np.zeros(n_global_phi); pp_g[g0p:g1p] = pp_l
    fd_g = np.zeros(n_global_phi); fd_g[g0p:g1p] = fd_l
    msh.comm.Allreduce(MPI.IN_PLACE, pm_g, op=MPI.SUM)
    msh.comm.Allreduce(MPI.IN_PLACE, pp_g, op=MPI.SUM)
    msh.comm.Allreduce(MPI.IN_PLACE, fd_g, op=MPI.SUM)

    pj_new = (pm_g - pp_g) / np.maximum(fd_g, 1.)
    pj_fn.x.array[:n_local_phi] = pj_new[g0p:g1p]
    pj_fn.x.scatter_forward()

def phi_grid_set(x):
    """
    사분원 4개로 나누는 정중앙 십자 기둥 장애물 (LVPP용)
    """
    y = x[1]
    z = x[2]
    
    # 1. 기본 상한선 (사분원 빈 공간: d_grid까지 팽창 허용)
    phi_vals = np.full_like(y, 0.2) # d_grid가 아닌 그냥 최대로 해놓음
    
    # 2. 정중앙 십자 기둥 위치 판별
    # y=0을 중심으로 한 수평 바 OR z=0을 중심으로 한 수직 바
    on_bar = (np.abs(y) < w_bar) | (np.abs(z) < w_bar)
    
    # 3. 십자 기둥이 있는 곳만 천장을 낮춤
    phi_vals[on_bar] = d_grid - t_bar
    
    return phi_vals

alpha_lvpp = Constant(sub_m, PETSc.ScalarType(1.0))

sol_m   = Function(V_m_mixed, name="membrane_sol")
sol_m_k = Function(V_m_mixed, name="membrane_sol_k")

phi_fn  = Function(V_phi, name="phi_grid")      # 격자 상부 장애물
phi_fn.interpolate(phi_grid_set)                # 초기 1회만 계산
pj_fn   = Function(V_phi, name="pressure_jump") # 압력 점프 [[p]]

w_s,  psi_s = split(sol_m)
_,    psi_k = split(sol_m_k)
vm0,  vm1   = TestFunctions(V_m_mixed)
'''
F_m = (
    T_mem * inner(grad(w_s), grad(vm0)) * dxm   # 막 탄성
  + psi_s      * vm0 * dxm                                    # ψ → w 결합
  + w_s        * vm1 * dxm                                    # w → ψ 결합
  + exp(-psi_s) * vm1 * dxm                                    # ★ 상부 LVPP: +e^ψ
  - phi_fn     * vm1 * dxm                                    # φ_grid 상한
  - alpha_lvpp * (pj_fn) * vm0 * dxm                  # 압력 하중
  - psi_k      * vm0 * dxm                                    # proximal 항
)
'''
F_m = (
    T_mem * inner(grad(w_s), grad(vm0)) * dxm     
    - pj_fn * vm0 * dxm           
    + psi_s * vm0 * dxm          
    + w_s * vm1 * dxm
    + exp(-psi_s) * vm1 * dxm
    - phi_fn * vm1 * dxm
    + (1.0 / alpha_lvpp) * (psi_s - psi_k) * vm0 * dxm 
)
J_m = ufl.derivative(F_m, sol_m)

lvpp_problem = NonlinearProblem(
    F_m, u=sol_m, bcs=[bc_wm], J=J_m,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type":  "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-6,
        "snes_atol": 1e-10,
        "snes_max_it": 100,
        "snes_linesearch_type": "bt",
        "snes_error_if_not_converged": False,
    },
    petsc_options_prefix="lvpp_grid_"
)

# ═══════════════════════════════════════════════════════════════
# § 8. ALE 약형
# ═══════════════════════════════════════════════════════════════
d_t3, e3 = TrialFunction(V_ale), TestFunction(V_ale)
d_ale    = Function(V_ale); d_ale.name = "ALE_disp"
d_mem_fn = Function(V_ale)
x_f      = SpatialCoordinate(msh)
mesh_stiffness = 1.0 / (abs(x_f[0] - x_m) + 0.05)**3

a_ale_ufl = form(mesh_stiffness * inner(nabla_grad(d_t3), nabla_grad(e3)) * dxf)
L_ale_ufl = form(inner(Constant(msh, PETSc.ScalarType((0., 0., 0.))), e3) * dxf)

# ═══════════════════════════════════════════════════════════════
# § 9. KSP 솔버
# ═══════════════════════════════════════════════════════════════
PETSc.Options()["ksp_error_if_not_converged"] = False

def make_ksp(A, ktype, pctype, comm):
    s = PETSc.KSP().create(comm)
    s.setOperators(A); s.setType(ktype)
    s.getPC().setType(pctype)
    s.setTolerances(rtol=1e-7, max_it=800)
    s.setFromOptions()
    return s

A1 = create_matrix(a1_ufl); A2 = create_matrix(a2_ufl)
A3 = create_matrix(a3_ufl); A_a = create_matrix(a_ale_ufl)
b1 = create_vector(V_f); b2 = create_vector(Q_f)
b3 = create_vector(V_f); b_a = create_vector(V_ale)

s1  = make_ksp(A1,  "bcgs", "bjacobi", msh.comm)
s2  = make_ksp(A2,  "cg",   "hypre",   msh.comm)
s3  = make_ksp(A3,  "cg",   "jacobi",  msh.comm)
s_a = make_ksp(A_a, "cg",   "hypre",   msh.comm)

def ksp_solve(solver, b_vec, x_fn, L_f, a_f=None, bcs=None):
    with b_vec.localForm() as loc: loc.set(0.)
    assemble_vector(b_vec, L_f)
    if a_f and bcs:
        apply_lifting(b_vec, [a_f], [bcs])
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    if bcs:
        set_bc(b_vec, bcs)
    solver.solve(b_vec, x_fn.x.petsc_vec)
    x_fn.x.scatter_forward()

# ═══════════════════════════════════════════════════════════════
# § 10. DOF 좌표 + MPI 인덱스
# ═══════════════════════════════════════════════════════════════
ale_coords = V_ale.tabulate_dof_coordinates()

imap_m0        = V_m0.dofmap.index_map
n_local_m0     = imap_m0.size_local
n_global_m0    = imap_m0.size_global
local_range_m0 = imap_m0.local_range
mem_dof_local  = V_m0.tabulate_dof_coordinates()[:n_local_m0]

imap_phi        = V_phi.dofmap.index_map
n_local_phi     = imap_phi.size_local
n_global_phi    = imap_phi.size_global
local_range_phi = imap_phi.local_range
phi_dof_local   = V_phi.tabulate_dof_coordinates()[:n_local_phi]

n_local_ind = V_ind.dofmap.index_map.size_local

geo_orig        = msh.geometry.x.copy()
tree_geo        = cKDTree(geo_orig)
dist_ag, idx_ag = tree_geo.query(ale_coords)
valid_ag        = dist_ag < 1e-8
ale_bs          = V_ale.dofmap.index_map_bs

tree_ale        = cKDTree(ale_coords)
dist_ma, idx_ma = tree_ale.query(mem_dof_local)
valid_ma        = dist_ma < 1e-8

mem_ale_dofs = locate_dofs_topological(V_ale, fdim, ft.find(4))
w_prev_arr   = np.zeros(n_local_m0)

# V_phi ↔ V_m0 매핑
tree_m0_local              = cKDTree(mem_dof_local)
dist_phi_m0, idx_phi_to_m0 = tree_m0_local.query(phi_dof_local)
valid_phi_m0               = dist_phi_m0 < 1e-8

# V_m_mixed.sub(1) ↔ V_phi 매핑 (LVPP 초기화용)
V_m1, m1_to_m = V_m_mixed.sub(1).collapse()
n_local_m1    = V_m1.dofmap.index_map.size_local
m1_coords     = V_m1.tabulate_dof_coordinates()[:n_local_m1]
if n_local_phi > 0 and n_local_m1 > 0:
    tree_phi_loc          = cKDTree(phi_dof_local)
    dist_m1p, idx_m1p     = tree_phi_loc.query(m1_coords)
    valid_m1p             = dist_m1p < 1e-8
else:
    valid_m1p = np.zeros(n_local_m1, dtype=bool)
    idx_m1p   = np.zeros(n_local_m1, dtype=np.int32)
sub0_dofs = sub0_to_parent   # shape: (n_dofs_m0,) 1D 전역 인덱스

# ═══════════════════════════════════════════════════════════════
# § 11. 헬퍼 함수
# ═══════════════════════════════════════════════════════════════
'''
def update_grid_indicator():
    """
    격자 와이어 내부 셀 → chi_grid = 1 (Brinkman no-slip)

    정사각형 격자 (yz 평면):
      ① z방향 와이어: 단면 (xc-x_grid)²+(yc-y_w)² < r_wire²
      ② y방향 와이어: 단면 (xc-x_grid)²+(zc-z_w)² < r_wire²

    현재 물리 좌표(ALE 이동 후) 기준으로 계산 → 매 5스텝 갱신
    """
    coords = V_ind.tabulate_dof_coordinates()[:n_local_ind]
    xc = coords[:, 0]; yc = coords[:, 1]; zc = coords[:, 2]
    in_wire = np.zeros(n_local_ind, dtype=bool)
    dx2     = (xc - x_grid) ** 2
    i_max   = int(np.floor(R / a_grid)) + 1

    for i in range(-i_max, i_max + 1):
        y_w = i * a_grid
        if abs(y_w) >= R: continue
        in_wire |= (dx2 + (yc - y_w) ** 2 < r_wire ** 2)  # ① z방향 와이어
        in_wire |= (dx2 + (zc - y_w) ** 2 < r_wire ** 2)  # ② y방향 와이어

    chi_grid.x.array[:n_local_ind] = in_wire.astype(PETSc.ScalarType)
    chi_grid.x.scatter_forward()
'''
def update_grid_indicator():
    """
    사분원 4개로 나누는 정중앙 십자 기둥 (3D 유체 Brinkman용)
    """
    coords = V_ind.tabulate_dof_coordinates()[:n_local_ind]
    xc = coords[:, 0]; yc = coords[:, 1]; zc = coords[:, 2]
    
    # x축 방향 기둥 두께 판별
    x_ok = np.abs(xc - x_grid) < t_bar
    
    # 정중앙 십자 영역 판별 (y=0 수평바 OR z=0 수직바)
    in_bar = x_ok & ((np.abs(yc) < w_bar) | (np.abs(zc) < w_bar))

    chi_grid.x.array[:n_local_ind] = in_bar.astype(PETSc.ScalarType)
    chi_grid.x.scatter_forward()

def update_phi_grid():
    """
    격자 상부 장애물 함수 φ_grid(y,z) [초기 1회만 계산]

    와이어 표면까지 최소 x 거리:
      z방향 와이어 (y=y_w): φ = d_grid − √(r_wire² − (y−y_w)²)  for |y−y_w| < r_wire
      y방향 와이어 (z=z_w): φ = d_grid − √(r_wire² − (z−z_w)²)  for |z−z_w| < r_wire
      구멍:                  φ = phi_inf  (사실상 무제약)

    상부 장애물: w ≤ φ_grid
    """
    current_coords = V_phi.tabulate_dof_coordinates()[:n_local_phi]
    y_q = current_coords[:, 1]
    z_q = current_coords[:, 2]
    phi_inf = d_grid + r_wire * 20   # 구멍 위치 (충분히 큰 값)
    phi_arr = np.full(n_local_phi, phi_inf)
    i_max   = int(np.floor(R / a_grid)) + 1

    for i in range(-i_max, i_max + 1):
        y_w = i * a_grid
        # ① z방향 와이어: y 방향 거리로 제약
        dy = y_q - y_w
        mk = np.abs(dy) < r_wire
        if mk.any():
            phi_arr[mk] = np.minimum(
                phi_arr[mk],
                d_grid - np.sqrt(np.maximum(r_wire**2 - dy[mk]**2, 0.))
            )
        # ② y방향 와이어: z 방향 거리로 제약
        z_w = i * a_grid
        dz  = z_q - z_w
        mk  = np.abs(dz) < r_wire
        if mk.any():
            phi_arr[mk] = np.minimum(
                phi_arr[mk],
                d_grid - np.sqrt(np.maximum(r_wire**2 - dz[mk]**2, 0.))
            )

    phi_fn.x.array[:n_local_phi] = phi_arr
    phi_fn.x.scatter_forward()


def init_lvpp_psi():
    """
    LVPP 초기화: ψ₀ = ln(φ_grid) → w₀ = φ_grid − e^ψ₀ = 0

    초기 막 변위 0 보장. V_m_mixed.sub(1) ↔ V_phi 좌표 매핑 사용.
    """
    psi_init = np.zeros(n_local_m1)
    if valid_m1p.any():
        phi_vals         = phi_fn.x.array[idx_m1p[valid_m1p]]
        psi_init[valid_m1p] = np.log(np.maximum(phi_vals, 1e-10))
    sol_m.x.array[m1_to_m[:n_local_m1]] = psi_init
    sol_m.x.scatter_forward()

# collapse 결과를 재사용할 캐시 함수 추가 (§10 DOF 블록 끝에)
_w_collapse_fn = Function(V_m0)   # 재사용 가능한 함수 객체

def get_w_arr():
    """collapse 대신 dofmap 직접 접근으로 임시 함수 생성 방지"""
    _w_collapse_fn.x.array[:] = sol_m.x.array[
        V_m_mixed.sub(0).dofmap.list.array
    ]
    return _w_collapse_fn.x.array[:n_local_m0]

"""
def phi_grid_set(x):
"""
"""
LVPP 상부 장애물 함수 (2D 막 메시에 덧씌울 스칼라 함수)
x[0], x[1], x[2] = 각각 x, y, z 좌표 (막 서브메시는 x가 일정함)
"""
"""
    y = x[1]
    z = x[2]
    
    # 1. 기본 상한선: 구멍(Hole) 위치에서는 막이 d_grid 까지는 부풀 수 있다고 설정
    phi_vals = np.full_like(y, 10) # d_grid가 아닌 그냥 최대로 해놓음
    
    # 2. 십자 기둥 위치 찾기 및 상한선 깎기
    i_max = int(np.floor(R / a_grid)) + 1
    for i in range(-i_max, i_max + 1):
        center = i * a_grid
        if abs(center) >= R: continue
        
        # 기둥이 지나가는 십자 영역 판별 (y축 평행 기둥 OR z축 평행 기둥)
        on_bar = (np.abs(y - center) < w_bar) | (np.abs(z - center) < w_bar)
        
        # 기둥이 있는 곳은 막이 기둥 전면(d_grid - t_bar)까지만 올 수 있도록 천장을 낮춤
        phi_vals[on_bar] = d_grid - t_bar
        
    return phi_vals
"""


def solve_lvpp_membrane():
    alpha_k = LVPP_C; total_newton = 0
    sol_m_k.x.array[:] = sol_m.x.array[:]

    for k in range(LVPP_MAX):
        try:
            alpha_new = max(LVPP_C * LVPP_R**(LVPP_Q**k) - alpha_k, LVPP_C)
        except OverflowError:
            alpha_new = LVPP_AMAX
        alpha_k = min(alpha_new, LVPP_AMAX)
        alpha_lvpp.value = PETSc.ScalarType(alpha_k)

        lvpp_problem.solve()
        total_newton += lvpp_problem.solver.getIterationNumber()

        w_new   = sol_m.x.array[sub0_dofs][:n_local_m0].copy()
        w_old = sol_m_k.x.array[sub0_dofs][:n_local_m0]

        inc = float(msh.comm.allreduce(
            float(np.dot(w_new - w_old, w_new - w_old)), op=MPI.SUM))**0.5

        # ★ 수정: 수렴 판정 전에 갱신하면 안 됨
        if inc < LVPP_TOL:
            break
        sol_m_k.x.array[:] = sol_m.x.array[:]  # ← break 이후로 이동

    return total_newton

# ═══════════════════════════════════════════════════════════════
# § 12. 초기화 & Output
# ═══════════════════════════════════════════════════════════════
Path("results").mkdir(exist_ok=True, parents=True)

vtx_u   = VTXWriter(msh.comm,   "results/u.bp",   u_,    engine="BP4")
vtx_p   = VTXWriter(msh.comm,   "results/p.bp",   p_,    engine="BP4")
vtx_d   = VTXWriter(msh.comm,   "results/d.bp",   d_ale, engine="BP4")
chi_vis = Function(V_ind, name="grid_chi")
vtx_chi = VTXWriter(msh.comm,   "results/chi.bp", chi_vis, engine="BP4")

w_vis   = Function(V_m0,  name="Deflection")
phi_vis = Function(V_phi, name="Phi_grid")
vtx_w   = VTXWriter(sub_m.comm, "results/w.bp",   w_vis,   engine="BP4")
vtx_phi = VTXWriter(sub_m.comm, "results/phi.bp", phi_vis, engine="BP4")

for fn in [u_, u_n, p_, p_n, d_ale, w_msh]:
    fn.x.array[:] = 0.
sol_m.x.array[:] = 0.; sol_m_k.x.array[:] = 0.
pj_fn.x.array[:] = 0.
phi_fn.interpolate(phi_grid_set) # 기존 객체에 네모 그물 공식 덮어씌우기

update_grid_indicator()   # chi_grid 설정
init_lvpp_psi()            # ψ₀ = ln(φ_grid) → w₀ = 0

# ═══════════════════════════════════════════════════════════════
# § 13. Time loop
# ═══════════════════════════════════════════════════════════════
gtree = bb_tree(msh, msh.topology.dim)
t = 0.0

for step in range(N):
    t += dt
    iv.t = t
    u_in.interpolate(iv)

    # ── NS IPCS (chi_grid 매 5스텝 갱신 → A1 재조립 필요) ────
    A1.zeroEntries(); assemble_matrix(A1, a1_ufl, bcs=bcu); A1.assemble()
    s1.setOperators(A1)
    A2.zeroEntries(); assemble_matrix(A2, a2_ufl, bcs=bcp); A2.assemble()
    s2.setOperators(A2)
    A3.zeroEntries(); assemble_matrix(A3, a3_ufl); A3.assemble()
    s3.setOperators(A3)

    ksp_solve(s1, b1, u_, L1_ufl, a1_ufl, bcu)
    ksp_solve(s2, b2, p_, L2_ufl, a2_ufl, bcp)
    ksp_solve(s3, b3, u_, L3_ufl)
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]

    # ── LVPP 막 (상부 장애물 w ≤ φ_grid) ─────────────────────
    pressure_jump_dynamic(p_, sol_m.x.array[sub0_dofs], gtree)
    total_n = solve_lvpp_membrane()
    w_new   = sol_m.x.array[sub0_dofs].copy()

    # 1. 막의 최대 변위 및 최대 x 좌표 (빈 구멍으로 부푼 정도)
    local_max_w = np.max(w_new) if len(w_new) > 0 else -1.0
    global_max_w = msh.comm.allreduce(local_max_w, op=MPI.MAX)
    max_x_coord = x_m + global_max_w  # 막이 도달한 절대 x 좌표

    # 2. 막과 십자 기둥(장애물) 사이의 최소 갭(Gap) 계산
    # w_new(V_m0 공간)를 phi_fn(V_phi 공간)과 비교하기 위해 매핑
    w_on_phi = np.zeros(n_local_phi)
    ok_map = np.where(valid_phi_m0)[0]
    if len(ok_map) > 0:
        w_on_phi[ok_map] = w_new[idx_phi_to_m0[ok_map]]
    
    # gap = phi - w (항상 0보다 크거나 같음. 0에 가까우면 충돌)
    local_min_gap = np.min(phi_fn.x.array[:n_local_phi] - w_on_phi) if n_local_phi > 0 else 1.0
    global_min_gap = msh.comm.allreduce(local_min_gap, op=MPI.MIN)

    # 3. 접촉 판별 (오차 고려해서 갭이 0.1mm 이하로 떨어지면 접촉으로 간주)
    is_contact = global_min_gap < 1e-4

    # ── ALE 조화 확장 ─────────────────────────────────────────
    dw = w_new[:n_local_m0] - w_prev_arr
    dw = np.clip(dw, -d_grid*0.1, d_grid*0.1)   # 증분 안정화

    d_mem_fn.x.array[:] = 0.
    ok = np.where(valid_ma)[0]
    d_mem_fn.x.array[idx_ma[ok] * ale_bs + 0] = dw[ok]
    d_mem_fn.x.scatter_forward()

    bc_ale_dyn = dirichletbc(d_mem_fn, mem_ale_dofs)
    bcs_ale    = [bc_ale_fixed, bc_ale_dyn]

    A_a.zeroEntries(); assemble_matrix(A_a, a_ale_ufl, bcs=bcs_ale); A_a.assemble()
    s_a.setOperators(A_a)
    ksp_solve(s_a, b_a, d_ale, L_ale_ufl, a_ale_ufl, bcs_ale)

    d_inc = d_ale.x.array.reshape(-1, ale_bs)
    if valid_ag.any():
        msh.geometry.x[idx_ag[valid_ag]] += d_inc[valid_ag]
    sub_m.geometry.x[:] = msh.geometry.x[geom_map]

    w_msh.interpolate(d_ale)
    w_msh.x.array[:] /= dt
    w_prev_arr[:] = w_new[:n_local_m0]

    # ── (추가) 십자 기둥(장애물) 영역 내 막의 최대 변위 계산 ──
    # mem_dof_local[:, 1]은 y좌표, mem_dof_local[:, 2]는 z좌표
    y_m = mem_dof_local[:, 1]
    z_m = mem_dof_local[:, 2]
    
    # 기둥 폭(w_bar) 안에 들어오는 노드만 필터링 (십자 모양)
    on_bar_mask = (np.abs(y_m) < w_bar) | (np.abs(z_m) < w_bar)
    w_on_bar = w_new[:n_local_m0][on_bar_mask]

    # 해당 영역 내의 로컬 최대 변위 구하기 (영역에 노드가 없으면 -1.0)
    local_max_w_bar = np.max(w_on_bar) if len(w_on_bar) > 0 else -1.0
    
    # 모든 프로세스의 결과를 모아서 글로벌 최대 변위 계산 (★ rank==0 밖에서 실행 필수!)
    global_max_w_bar = msh.comm.allreduce(local_max_w_bar, op=MPI.MAX)

    # 매 5스텝: bb_tree + 격자 지시자 갱신 (메시 이동 반영)
    if step % 5 == 0:
        gtree = bb_tree(msh, msh.topology.dim)
        update_grid_indicator()

    # ── Output ───────────────────────────────────────────────
    if step % 20 == 0:
        vtx_u.write(t); vtx_p.write(t); vtx_d.write(t)
        chi_vis.x.array[:] = chi_grid.x.array[:]
        vtx_chi.write(t)
        w_vis.x.array[:]   = w_new
        phi_vis.x.array[:] = phi_fn.x.array[:]
        vtx_w.write(t); vtx_phi.write(t)

    if step % 100 == 0 and msh.comm.rank == 0:
        # 접촉했으면 빨간색이나 눈에 띄게 표시, 아니면 남은 거리 표시
        if is_contact:
            contact_str = f"HIT! (기둥 위 변위: {global_max_w_bar*1000:.2f}mm)"
        else:
            contact_str = f"Gap: {global_min_gap*1000:.2f}mm"
        
        print(f"[{step:04d}] t={t:.3f}s | "
              f"최대 X좌표: {max_x_coord:.4f}m (변위: {global_max_w*1000:.2f}mm) | "
              f"장애물: {contact_str} | "
              f"LVPP_Iter={total_n}", flush=True)

for v in [vtx_u, vtx_p, vtx_d, vtx_chi, vtx_w, vtx_phi]:
    v.close()

if msh.comm.rank == 0:
    print("\nDone.")
    print("  u.bp   : 유체 속도  (와이어 주변 no-slip 확인)")
    print("  p.bp   : 유체 압력  (와이어 후방 저압)")
    print("  chi.bp : 격자 지시자 (Brinkman 영역)")
    print("  w.bp   : 막 변위    (w ≤ φ_grid 상부 LVPP)")
    print("  phi.bp : 장애물 함수 (와이어 표면)")
