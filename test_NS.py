"""
FSI: ALE Navier-Stokes + Poisson Membrane + Darcy Permeation
특이사항:
  1. NS 행렬 A1,A2,A3 매 스텝 재조립
  2. sub_m.geometry.x를 msh와 매 스텝 동기화(막 메쉬 매번 업뎃한다는 뜻)
  3. pressure_jump를 현재 막 위치 기반 동적 샘플링으로 수정
"""

import numpy as np
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path
from scipy.spatial import cKDTree

from dolfinx import io
from dolfinx.mesh import create_submesh
from dolfinx.fem import (
    Constant, Function, functionspace, form,
    dirichletbc, locate_dofs_topological, locate_dofs_geometrical,
)
from dolfinx.fem.petsc import (
    assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc, create_matrix
)
from dolfinx.io import VTXWriter
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from basix.ufl import element as belem
from ufl import (
    Identity, TestFunction, TrialFunction, FacetNormal,
    div, dot, dx, inner, lhs, nabla_grad, rhs, sym,
    avg, Measure, grad, SpatialCoordinate
)

# ═══════════════════════════════════════════════════════════════
# § 0. Parameters
# ═══════════════════════════════════════════════════════════════
L, R, x_m  = 5.0, 0.5, 3.75
T_sim, N   = 2.0, 2000 # 2초 시뮬레이션, 2000 타임스텝 -> dt=1ms
dt         = T_sim / N

rho_v      = 1.0 # 유체 밀도
mu_v       = 0.01
kappa_v    = 5e-3
T_mem      = 25.0
u_max      = 10.0
DK         = mu_v / kappa_v        # μ/κ: Darcy 저항 계수 (물리량)
mem_lc = 0.04   # 막 메시 크기 (법선 계산 이웃 탐색 반경용)


# 압력 샘플링 오프셋 (초기 메시 크기의 절반 수준 — 동적으로 재계산됨)
EPS_BASE   = 0.03

# ═══════════════════════════════════════════════════════════════
# § 1. Mesh
#  - 3D 원통 메시 2개 + 2D 막 서브메시 생성
# ═══════════════════════════════════════════════════════════════
gmsh.initialize()
gmsh.model.add("fsi")
if MPI.COMM_WORLD.rank == 0:
    c1 = gmsh.model.occ.addCylinder(0,   0, 0, x_m,     0, 0, R) # 막이 초기 위치인 x=3.75 기준으로 왼쪽에 원통 생성
    c2 = gmsh.model.occ.addCylinder(x_m, 0, 0, L - x_m, 0, 0, R) # 막이 초기 위치인 x=3.75 기준으로 오른쪽에 원통 생성
    dk = gmsh.model.occ.addDisk(x_m, 0, 0, R, R) # 막 위치에 원판 추가 (초기 막 위치에서의 경계면)
    gmsh.model.occ.rotate([(2, dk)], x_m, 0, 0, 0, 1, 0, np.pi / 2)
    out_tags, out_map = gmsh.model.occ.fragment(
        #-> fragment를 통해 원통 2개와 원판 1개를 겹쳐서 3D 영역과 2D 경계면으로 분할. 
        # out_tags: (dim, tag) 리스트, out_map: dim별로 tag 리스트
        [(3, c1), (3, c2)], [(2, dk)] )
    gmsh.model.occ.synchronize() # -> OCC 모델링 후 gmsh.model.occ.synchronize() 호출하여 내부 데이터 구조 업데이트

    mem_surfs = [t for d, t in out_map[2] if d == 2] # dim이 2인 태그 중에서 막 경계면 태그만 추출
    vol_tags  = [t for d, t in out_tags   if d == 3] # dim이 3인 태그 중에서 3D 영역 태그만 추출
    bnd       = gmsh.model.getBoundary(
        [(3, t) for t in vol_tags], oriented=False, combined=True
    ) # 3D 영역 태그를 입력으로 getBoundary 호출하여 경계면 태그와 유형(경계면이 속한 3D 영역과의 관계)을 반환. combined=True로 중복 제거
    In, Out, Wall = [], [], []
    for d, tag in bnd:
        if d != 2 or tag in mem_surfs: continue # dim이 2가 아니거나 막 경계면 태그인 경우 = 벽 경계면이 아닌 경우 for 루프에서 제외
        cx = gmsh.model.occ.getCenterOfMass(d, tag)[0] # 경계면의 중심 좌표 계산 (x 좌표)
        if   np.isclose(cx, 0., atol=.15): In.append(tag) # x=0에 가까운 경계면은 유입구로 분류하여 In 리스트에 추가
        elif np.isclose(cx, L,  atol=.15): Out.append(tag) # x=L에 가까운 경계면은 유출구로 분류하여 Out 리스트에 추가
        else:                               Wall.append(tag) # 나머지 경계면은 벽으로 분류하여 Wall 리스트에 추가

    gmsh.model.addPhysicalGroup(3, vol_tags,  tag=1, name="fluid") # 3D 영역에 "fluid"라는 이름의 물리 그룹 추가
    gmsh.model.addPhysicalGroup(2, In,        tag=1, name="inlet") 
    gmsh.model.addPhysicalGroup(2, Out,       tag=2, name="outlet")
    gmsh.model.addPhysicalGroup(2, Wall,      tag=3, name="wall")
    gmsh.model.addPhysicalGroup(2, mem_surfs, tag=4, name="membrane")

    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.12)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05) # 막 주변은 더 세밀하게 메싱. 너무 세밀하게 하면 시간 오래 걸립니다..
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.model.mesh.generate(3) # 3D 메시 생성
    gmsh.model.mesh.optimize("Netgen")

gdata       = io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
msh, ct, ft = gdata.mesh, gdata.cell_tags, gdata.facet_tags
gmsh.finalize()

gdim = msh.geometry.dim # 3D 메시이므로 gdim=3

# ═══════════════════════════════════════════════════════════════
# § 2. Membrane 2D submesh
#   vmap_m: sub_m geometry node → msh geometry node  (FIX 2)
# ═══════════════════════════════════════════════════════════════
# 4번째 반환값인 geom_map을 받고, numpy 정수형 배열로 명시적 변환
sub_m, emap_m, vmap_m, geom_map_raw = create_submesh( # msh의 함수로 sub_m 생성. 
    # geom_map_raw는 sub_m geometry 노드가 msh geometry 노드 중 어디에 대응되는지를 나타내는 매핑 정보.
    msh, msh.topology.dim - 1, ft.find(4) 
)
geom_map = np.array(geom_map_raw, dtype=np.int32) # sub_m을 행렬 형식으로 변환.
# ═══════════════════════════════════════════════════════════════
# § 3. Function spaces
# ═══════════════════════════════════════════════════════════════
V_f   = functionspace(msh,   belem("Lagrange", msh.basix_cell(),   2, shape=(gdim,))) # 유체 속도 공간: 2차 벡터 요소
Q_f   = functionspace(msh,   belem("Lagrange", msh.basix_cell(),   1)) # 유체 압력 공간: 1차 스칼라 요소
V_ale = functionspace(msh,   belem("Lagrange", msh.basix_cell(),   1, shape=(gdim,)))  # ALE 메시 변위 공간: 1차 벡터 요소
W_m   = functionspace(sub_m, ("Lagrange", 1)) # 막 변위 공간: 1차 스칼라 요소 (sub_m은 2D 막 서브메시)

# ═══════════════════════════════════════════════════════════════
# § 4. Integration measures
# ═══════════════════════════════════════════════════════════════
dxf  = Measure("dx", domain=msh,   subdomain_data=ct)
dS_m = Measure("dS", domain=msh,   subdomain_data=ft, subdomain_id=4) # 막 경계면에서의 표면 적분 (dS) 정의. 
dxm  = Measure("dx", domain=sub_m) # 막의 약형식에서는 sub_m의 체적 적분(dx)을 사용.
n    = FacetNormal(msh)   # dS_m (interior) 에서 사용 → 합법

# ═══════════════════════════════════════════════════════════════
# § 5. Boundary conditions
# ═══════════════════════════════════════════════════════════════
fdim = msh.topology.dim - 1 # 2D 경계면에서의 위상 차원, fdim=2

class InletVelocity:
    def __init__(self): self.t = 0.0
    def __call__(self, x):
        v    = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        r2   = x[1]**2 + x[2]**2
        v[0] = (u_max
                * np.sin(np.pi * self.t / (T_sim * 2))
                * np.maximum(1 - r2 / R**2, 0.)) 
        # 입력 속도는 sin 함수로 시간에 따라 변화하며, 반경 방향으로는 포물면.
        return v

iv   = InletVelocity()
u_in = Function(V_f)
u_in.interpolate(iv)

bcu = [
    dirichletbc(u_in,
        locate_dofs_topological(V_f, fdim, ft.find(1))),
    dirichletbc(np.zeros(gdim, dtype=PETSc.ScalarType),
        locate_dofs_topological(V_f, fdim, ft.find(3)), V_f),
]
bcp = [
    dirichletbc(Constant(msh, PETSc.ScalarType(0.)),
        locate_dofs_topological(Q_f, fdim, ft.find(2)), Q_f),
]
bc_wm = dirichletbc(
    PETSc.ScalarType(0.),
    locate_dofs_geometrical(
        W_m, lambda x: np.isclose(np.sqrt(x[1]**2 + x[2]**2), R, atol=2e-2)
    ), W_m
)
bc_ale_fixed = dirichletbc(
    np.zeros(gdim, dtype=PETSc.ScalarType),
    locate_dofs_geometrical(V_ale, lambda x: (
        np.isclose(x[0], 0., atol=.05) |
        np.isclose(x[0], L,  atol=.05) |
        np.isclose(np.sqrt(x[1]**2 + x[2]**2), R, atol=.05)
    )), V_ale
)

# ═══════════════════════════════════════════════════════════════
# § 6. UFL weak forms  (계수는 고정, 행렬은 루프에서 재조립)
# ═══════════════════════════════════════════════════════════════
def eps_(u): return sym(nabla_grad(u))
def sig_(u, p): return 2 * mu_v * eps_(u) - p * Identity(gdim)

u_t, vf  = TrialFunction(V_f),  TestFunction(V_f) # u_t: NS Step 1에서 구할 중간 속도, vf: NS Step 1의 테스트 함수
p_t, qf  = TrialFunction(Q_f),  TestFunction(Q_f) # p_t: NS Step 2에서 구할 압력, qf: NS Step 2의 테스트 함수
u_n, p_n = Function(V_f), Function(Q_f) # 이전 속도/압력
u_,  p_  = Function(V_f), Function(Q_f) # 최종 속도/압력
u_.name, p_.name = "Velocity", "Pressure"

w_msh = Function(V_f)          # ALE mesh velocity, 상대적 대류속도 계산용
U     = 0.5 * (u_n + u_t)     # Crank-Nicolson, 미분항을 선형항으로.

# NS Step 1: momentum + Darcy surface resistance
F1 = (
    rho_v / dt * inner(u_t - u_n, vf) * dxf
  + rho_v * inner(dot(u_n - w_msh, nabla_grad(u_n)), vf) * dxf
  + inner(sig_(U, p_n), eps_(vf)) * dxf
  + DK * dot(avg(U), n("+")) * dot(avg(vf), n("+")) * dS_m
)
# 전체 ALE-NS + Darcy 저항 항을 포함하는 NS Step 1의 약형식 F1 정의. 7번 슬라이드와 동일.
a1_ufl = form(lhs(F1))
L1_ufl = form(rhs(F1))

# NS Step 2: pressure Poisson
a2_ufl = form(dot(nabla_grad(p_t), nabla_grad(qf)) * dxf)
L2_ufl = form(dot(nabla_grad(p_n), nabla_grad(qf)) * dxf
            - rho_v / dt * div(u_) * qf * dxf)
# 2번 슬라이드의 2. 약형 변환과 동일한 식.

# NS Step 3: velocity correction
a3_ufl = form(rho_v * dot(u_t, vf) * dxf)
L3_ufl = form(rho_v * dot(u_, vf) * dxf
            - dt * dot(nabla_grad(p_ - p_n), vf) * dxf)
# 2번 슬라이드의 속도 보정 수식 약형 변환과 동일한 식.

# Membrane Poisson: -T Δ_Γ w = [[p]], 반력은 압력차에 의해 나타남
w_t2, vm = TrialFunction(W_m), TestFunction(W_m)
w_,  w_pr = Function(W_m), Function(W_m)
w_.name   = "Deflection"
pj_fn     = Function(W_m)
a_m_ufl   = form(T_mem * inner(grad(w_t2), grad(vm)) * dxm)
L_m_ufl   = form(pj_fn * vm * dxm)

# ALE extension with Variable Stiffness (메시 꼬임 방지)
d_t3, e3  = TrialFunction(V_ale), TestFunction(V_ale)
d_ale     = Function(V_ale); d_ale.name = "ALE_disp"
d_mem_fn  = Function(V_ale)
x_f = SpatialCoordinate(msh)
# 🌟 막(mem_pos = 3.75)에 가까울수록 강성이 급격히 커지는 가중치 함수
dist_to_mem = abs(x_f[0] - x_m)
# 분모에 0.05 정도의 오프셋을 줘서 0으로 나누어지는 것을 방지
mesh_stiffness = 1.0 / (dist_to_mem + 0.05)**3 # 차수를 높일수록 엄격히 방지

# 강성이 포함된 새로운 약형식: -∇·(k ∇d) = 0
a_ale_ufl = form(mesh_stiffness * inner(nabla_grad(d_t3), nabla_grad(e3)) * dxf)
L_ale_ufl = form(inner(Constant(msh, PETSc.ScalarType((0., 0., 0.))), e3) * dxf)

# ═══════════════════════════════════════════════════════════════
# § 7. KSP factory
#  - 매 스텝 NS 행렬 재조립 → KSP에 행렬 갱신 전달 필요 (setOperators)
# ═══════════════════════════════════════════════════════════════
PETSc.Options()["ksp_error_if_not_converged"] = False

def make_ksp(A, ktype, pctype, comm):
    s = PETSc.KSP().create(comm)
    s.setOperators(A)
    s.setType(ktype)
    s.getPC().setType(pctype)
    s.setTolerances(rtol=1e-7, max_it=800)
    s.setFromOptions()
    return s

# ═══════════════════════════════════════════════════════════════
# § 8. DOF / geometry mappings  (one-time setup)
# ═══════════════════════════════════════════════════════════════
ale_coords   = V_ale.tabulate_dof_coordinates()   # (N_ale, 3)
mem_dof_c    = W_m.tabulate_dof_coordinates()     # (N_mem, 3)  초기 막 DOF 좌표

# W_m DOF → V_ale DOF  (for ALE BC from membrane displacement)
tree_ale        = cKDTree(ale_coords)
dist_ma, idx_ma = tree_ale.query(mem_dof_c)
valid_ma        = dist_ma < 1e-8
ale_bs          = V_ale.dofmap.index_map_bs         # block size = 3

# V_ale geometry node → msh geometry node
#  (ALE에서 메시 이동 후 역방향 동기화용)
#  geo_orig: 초기 3D geometry 좌표
geo_orig = msh.geometry.x.copy()                   # (N_geo, 3)
tree_geo      = cKDTree(geo_orig)
dist_ag, idx_ag = tree_geo.query(ale_coords)
valid_ag        = dist_ag < 1e-8                    # ale DOF → geo node

mem_ale_dofs = locate_dofs_topological(V_ale, fdim, ft.find(4))

# ═══════════════════════════════════════════════════════════════
# § 9. Dynamic pressure jump
#
#   법선벡터 계산과 샘플링을 완전 벡터화하여 효율성 극대화
# ═══════════════════════════════════════════════════════════════-
def pressure_jump_dynamic(p_fn, w_arr, mem_dof_coords_init,
                           gtree_current, eps=EPS_BASE):
    n_pts    = len(mem_dof_coords_init)
    current_x = mem_dof_coords_init[:, 0] + w_arr
    y_c       = mem_dof_coords_init[:, 1]
    z_c       = mem_dof_coords_init[:, 2]

    max_w   = np.max(np.abs(w_arr)) if w_arr.size > 0 else 0.
    eps_dyn = max(eps, max_w * 1.5 + 1e-4)

    # ── 법선 벡터 계산 완전 벡터화 ───────────────────────────
    yz_coords = np.column_stack([y_c, z_c])
    tree_mem  = cKDTree(yz_coords)
    all_nbrs  = tree_mem.query_ball_point(yz_coords, r=3.0*mem_lc)

    dw_dy = np.zeros(n_pts)
    dw_dz = np.zeros(n_pts)

    for i, raw in enumerate(all_nbrs):
        nbrs = np.array([j for j in raw if j != i])
        if len(nbrs) == 0: continue

        dy = y_c[nbrs] - y_c[i]
        dz = z_c[nbrs] - z_c[i]
        dw = w_arr[nbrs] - w_arr[i]

        # 2x2 최소자승 (행렬 없이 직접 계산), 주변 점으로 법선벡터 계산
        syy = dy @ dy; szz = dz @ dz; syz = dy @ dz
        syw = dy @ dw; szw = dz @ dw
        det = syy*szz - syz*syz

        if abs(det) > 1e-20:
            dw_dy[i] = (szz*syw - syz*szw) / det
            dw_dz[i] = (syy*szw - syz*syw) / det

    nx = np.ones(n_pts); ny = -dw_dy; nz = -dw_dz
    mag = np.maximum(np.sqrt(nx**2 + ny**2 + nz**2), 1e-14)
    nx /= mag; ny /= mag; nz /= mag # 단위벡터(크기 1)로 정규화

    # ── 샘플링 벡터화 ────────────────────────────────────────
    p_minus = np.zeros(n_pts)
    p_plus  = np.zeros(n_pts)

    for sign, arr in [(-1, p_minus), (+1, p_plus)]:
        pts = np.column_stack([
            current_x + sign*eps_dyn*nx,
            y_c       + sign*eps_dyn*ny,
            z_c       + sign*eps_dyn*nz,
        ]) # 각 DOF에서 법선 방향으로 eps_dyn만큼 떨어진 샘플링 점 좌표 (n_pts, 3)
        r = np.sqrt(pts[:,1]**2 + pts[:,2]**2)
        out = r > R*0.92
        pts[out,1] *= R*0.92/r[out]
        pts[out,2] *= R*0.92/r[out]
        pts[:,0] = np.clip(pts[:,0], 0.01, L-0.01)

        coll  = compute_collisions_points(gtree_current, pts)
        col_c = compute_colliding_cells(msh, coll, pts)

        # links를 한 번에 처리
        cells = np.array([
            col_c.links(i)[0] if len(col_c.links(i)) > 0 else -1
            for i in range(n_pts)], dtype=np.int32)
        found = (cells >= 0).astype(np.float64)

        ok = cells >= 0
        if ok.any():
            arr[ok] = p_fn.eval(
                pts[ok].reshape(-1,3), cells[ok]).reshape(-1)

    # MPI
    fg = np.zeros(n_pts); pm = np.zeros(n_pts); pp = np.zeros(n_pts)
    msh.comm.Allreduce(found,   fg, op=MPI.SUM)
    msh.comm.Allreduce(p_minus, pm, op=MPI.SUM)
    msh.comm.Allreduce(p_plus,  pp, op=MPI.SUM)
    d = np.maximum(fg, 1.0)
    return pm/d - pp/d

# ═══════════════════════════════════════════════════════════════
# § 10. One-step solve helper
# ═══════════════════════════════════════════════════════════════
def ksp_solve(solver, b_vec, x_fn, L_f, a_f=None, bcs=None):
    with b_vec.localForm() as loc: loc.set(0)
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
# § 11. Output
# ═══════════════════════════════════════════════════════════════
Path("results").mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(msh.comm,   "results/u.bp",   u_,    engine="BP4")
vtx_p = VTXWriter(msh.comm,   "results/p.bp",   p_,    engine="BP4")
vtx_w = VTXWriter(sub_m.comm, "results/w.bp",   w_,    engine="BP4")
vtx_d = VTXWriter(msh.comm,   "results/d.bp",   d_ale, engine="BP4")

for fn in [u_, u_n, p_, p_n, w_, w_pr, d_ale, w_msh]:
    fn.x.array[:] = 0.

# ═══════════════════════════════════════════════════════════════
# § 12. Time loop
# ═══════════════════════════════════════════════════════════════
A1 = create_matrix(a1_ufl)
A2 = create_matrix(a2_ufl)
A3 = create_matrix(a3_ufl)
A_m = create_matrix(a_m_ufl)
A_a = create_matrix(a_ale_ufl)

b1, b2, b3 = create_vector(V_f), create_vector(Q_f), create_vector(V_f)
b_m = create_vector(W_m)
b_a = create_vector(V_ale)

s1 = make_ksp(A1, "bcgs", "bjacobi",   msh.comm) # 병렬환경이면 bjacobi 아니면 ilu
s2 = make_ksp(A2, "cg",   "hypre", msh.comm)
s3 = make_ksp(A3, "cg",   "jacobi",   msh.comm) # 병렬환경이면 jacobi 아니면 sor
s_m = make_ksp(A_m, "cg", "hypre", sub_m.comm)
s_a = make_ksp(A_a, "cg", "hypre", msh.comm)

gtree = bb_tree(msh, msh.topology.dim)
t = 0.0

for step in range(N):
    t += dt

    # ── 0. Inlet velocity update ────────────────────────────
    iv.t = t
    u_in.interpolate(iv)

    # ════════════════════════════════════════════════════════
    # FIX 1: 매 스텝 NS 행렬 재조립
    #   메시 좌표가 바뀌었으므로 dx, FacetNormal 모두 다시 계산
    # ════════════════════════════════════════════════════════
    A1.zeroEntries(); assemble_matrix(A1, a1_ufl, bcs=bcu); A1.assemble()
    s1.setOperators(A1)  # 🌟 솔버에 행렬 갱신 알림

    A2.zeroEntries(); assemble_matrix(A2, a2_ufl, bcs=bcp); A2.assemble()
    s2.setOperators(A2)  # 🌟 솔버에 행렬 갱신 알림

    A3.zeroEntries(); assemble_matrix(A3, a3_ufl);          A3.assemble()
    s3.setOperators(A3)  # 🌟 솔버에 행렬 갱신 알림

    # ── 1. NS Step 1: intermediate velocity ─────────────────
    ksp_solve(s1, b1, u_, L1_ufl, a1_ufl, bcu)

    # ── 2. NS Step 2: pressure Poisson ──────────────────────
    ksp_solve(s2, b2, p_, L2_ufl, a2_ufl, bcp)

    # ── 3. NS Step 3: velocity correction ───────────────────
    ksp_solve(s3, b3, u_, L3_ufl)
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]

    # ════════════════════════════════════════════════════════
    # FIX 3: 동적 압력 점프 (현재 막 변위 기반)
    # ════════════════════════════════════════════════════════
    pj_fn.x.array[:] = pressure_jump_dynamic(
        p_, w_.x.array, mem_dof_c, gtree
    )

    # ── 4. Membrane Poisson (매 스텝 재조립: sub_m 좌표 변함)
    A_m.zeroEntries(); assemble_matrix(A_m, a_m_ufl, bcs=[bc_wm]); A_m.assemble()
    s_m.setOperators(A_m)
    ksp_solve(s_m, b_m, w_, L_m_ufl, a_m_ufl, [bc_wm])

    # ── 5. ALE harmonic extension ────────────────────────────
    dw = w_.x.array - w_pr.x.array              # 증분 변위

    d_mem_fn.x.array[:] = 0.
    ok = np.where(valid_ma)[0]
    d_mem_fn.x.array[idx_ma[ok] * ale_bs + 0] = dw[ok]
    d_mem_fn.x.scatter_forward()

    bc_ale_dyn  = dirichletbc(d_mem_fn, mem_ale_dofs)
    bcs_ale     = [bc_ale_fixed, bc_ale_dyn]

    A_a.zeroEntries(); assemble_matrix(A_a, a_ale_ufl, bcs=bcs_ale); A_a.assemble()
    s_a.setOperators(A_a)
    ksp_solve(s_a, b_a, d_ale, L_ale_ufl, a_ale_ufl, bcs_ale)

    # ════════════════════════════════════════════════════════
    #   msh + sub_m 동기화 (메시 이동 후 신규 좌표로 업데이트)
    #   d_ale: (N_ale_dofs, 3) 증분 변위
    #   msh.geometry.x: (N_geo, 3)
    #   sub_m.geometry.x[i] = msh.geometry.x[vmap_m[i]]
    # ════════════════════════════════════════════════════════
    d_inc = d_ale.x.array.reshape(-1, ale_bs)   # (N_ale_dofs, 3)

    # 3D 메시 좌표 업데이트 (ALE DOF → geometry node)
    if valid_ag.any():
        msh.geometry.x[idx_ag[valid_ag]] += d_inc[valid_ag]

    # 2D submesh 좌표를 3D 메시에서 동기화
    # vmap_m[i]: sub_m의 i번 geometry 노드 = msh의 vmap_m[i]번 노드
    # 2D submesh 좌표를 3D 메시에서 동기화
    sub_m.geometry.x[:] = msh.geometry.x[geom_map]

    # ── 6. ALE mesh velocity: w_mesh = Δd/Δt ────────────────
    w_msh.interpolate(d_ale)
    w_msh.x.array[:] /= dt

    # ── 7. bb_tree 갱신 (메시 이동 후 필수) ─────────────────
    # bb_tree는 메시가 크게 변할 때만 갱신
    if step % 5 == 0:   # 매 스텝 말고 5스텝마다
        gtree = bb_tree(msh, msh.topology.dim)

    # ── 8. Store w for next increment ───────────────────────
    w_pr.x.array[:] = w_.x.array[:]

    # ── 9. Output ────────────────────────────────────────────
    if step % 10 == 0:
        vtx_u.write(t); vtx_p.write(t)
        vtx_w.write(t); vtx_d.write(t)

    if step % 50 == 0 and msh.comm.rank == 0:
        max_w  = np.max(np.abs(w_.x.array))
        max_u  = np.max(np.abs(u_.x.array))
        max_pj = np.max(np.abs(pj_fn.x.array))
        print(f"t={t:.4f}s | "
              f"max_w={max_w:.3e}  "
              f"max_u={max_u:.3e}  "
              f"max_pj={max_pj:.3e}  "
              f"max_d={np.max(np.abs(d_ale.x.array)):.3e}")

vtx_u.close(); vtx_p.close(); vtx_w.close(); vtx_d.close()
