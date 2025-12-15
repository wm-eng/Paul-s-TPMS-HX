"""
Streamlit UI for TPMS Lattice Visualization

Run with: streamlit run ui/streamlit_app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from skimage import measure
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.tpms_library import (
    TPMSType,
    VariantMode,
    evaluate_tpms,
    get_solid_mask,
    RECOMMENDED_T_RANGES
)

# Page config
st.set_page_config(
    page_title="TPMS Lattice Visualizer",
    page_icon="🔷",
    layout="wide"
)

st.title("🔷 TPMS Lattice Visualizer")
st.markdown("Visualize and export Triply Periodic Minimal Surface (TPMS) structures")

# Sidebar controls
st.sidebar.header("Parameters")

# TPMS type selection
tpms_type = st.sidebar.selectbox(
    "TPMS Type",
    options=list(TPMSType),
    format_func=lambda x: f"{x.value} - {x.name.replace('_', ' ').title()}"
)

# Variant mode
variant = st.sidebar.selectbox(
    "Variant Mode",
    options=list(VariantMode),
    format_func=lambda x: x.value.title()
)

# Get recommended t range
t_min, t_max = RECOMMENDED_T_RANGES[TPMSType(tpms_type)]
t_default = (t_min + t_max) / 2.0

# Threshold slider
t = st.sidebar.slider(
    "Threshold (t)",
    min_value=float(t_min),
    max_value=float(t_max),
    value=float(t_default),
    step=0.1,
    help="Isosurface threshold. Lower values = more solid material"
)

# Domain size
domain_size = st.sidebar.slider(
    "Domain Size (m)",
    min_value=0.01,
    max_value=0.1,
    value=0.05,
    step=0.01,
    help="Size of the cubic domain in meters"
)

# Repeats
nx = st.sidebar.slider("Repeats X", min_value=1, max_value=5, value=2)
ny = st.sidebar.slider("Repeats Y", min_value=1, max_value=5, value=2)
nz = st.sidebar.slider("Repeats Z", min_value=1, max_value=5, value=2)

# Grid resolution
resolution = st.sidebar.slider(
    "Grid Resolution",
    min_value=20,
    max_value=100,
    value=50,
    step=10,
    help="Number of grid points per unit cell (higher = smoother but slower)"
)

# Generate button
if st.sidebar.button("Generate Mesh", type="primary"):
    st.session_state.generate = True

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("3D Visualization")
    
    if 'generate' in st.session_state and st.session_state.generate:
        with st.spinner("Generating mesh..."):
            # Create coordinate grid
            # Scale by domain size and repeats
            Lx = domain_size * nx
            Ly = domain_size * ny
            Lz = domain_size * nz
            
            # Grid points per unit cell
            n_per_cell = resolution
            nx_grid = int(n_per_cell * nx)
            ny_grid = int(n_per_cell * ny)
            nz_grid = int(n_per_cell * nz)
            
            # Create coordinate arrays
            x = np.linspace(0, Lx, nx_grid)
            y = np.linspace(0, Ly, ny_grid)
            z = np.linspace(0, Lz, nz_grid)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Normalize coordinates to [0, 1] for TPMS functions
            # TPMS functions expect unit period
            X_norm = X / domain_size
            Y_norm = Y / domain_size
            Z_norm = Z / domain_size
            
            # Evaluate TPMS function
            f = evaluate_tpms(
                TPMSType(tpms_type),
                X_norm,
                Y_norm,
                Z_norm,
                VariantMode(variant),
                t
            )
            
            # Get solid mask
            solid_mask = get_solid_mask(
                TPMSType(tpms_type),
                X_norm,
                Y_norm,
                Z_norm,
                VariantMode(variant),
                t
            )
            
            # Compute porosity
            porosity = 1.0 - np.sum(solid_mask) / solid_mask.size
            
            # Marching cubes for mesh generation
            try:
                # Use isosurface at threshold
                # Note: scikit-image marching_cubes API may vary by version
                spacing = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
                
                if variant == VariantMode.SINGLE:
                    # For single mode: solid is f <= t, so isosurface at f = t
                    result = measure.marching_cubes(f, level=t, spacing=spacing)
                else:  # DOUBLE
                    # For double mode: solid is |f| <= t, so isosurface at |f| = t
                    f_abs = np.abs(f)
                    result = measure.marching_cubes(f_abs, level=t, spacing=spacing)
                
                # Handle different return formats
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        vertices, faces = result[0], result[1]
                    else:
                        raise ValueError("Unexpected marching_cubes return format")
                else:
                    # Newer API returns object with attributes
                    vertices = result.vertices
                    faces = result.faces
                
                # Compute surface area and volume
                # Surface area: sum of triangle areas
                triangle_areas = []
                for face in faces:
                    v0, v1, v2 = vertices[face]
                    a = np.linalg.norm(v1 - v0)
                    b = np.linalg.norm(v2 - v1)
                    c = np.linalg.norm(v0 - v2)
                    s = (a + b + c) / 2.0
                    area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
                    triangle_areas.append(area)
                surface_area = np.sum(triangle_areas)
                
                # Volume: use solid mask
                voxel_volume = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
                volume = np.sum(solid_mask) * voxel_volume
                
                # Surface area per volume
                A_surf_V = surface_area / volume if volume > 0 else 0.0
                
                # Store in session state
                st.session_state.vertices = vertices
                st.session_state.faces = faces
                st.session_state.porosity = porosity
                st.session_state.surface_area = surface_area
                st.session_state.volume = volume
                st.session_state.A_surf_V = A_surf_V
                st.session_state.mesh_generated = True
                
            except Exception as e:
                st.error(f"Mesh generation failed: {e}")
                st.session_state.mesh_generated = False
    
    if 'mesh_generated' in st.session_state and st.session_state.mesh_generated:
        # Create 3D plot
        vertices = st.session_state.vertices
        faces = st.session_state.faces
        
        # Create mesh plot
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.8,
                color='lightblue',
                flatshading=True
            )
        ])
        
        fig.update_layout(
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode='data'
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click 'Generate Mesh' to create a 3D visualization")

with col2:
    st.subheader("Properties")
    
    if 'mesh_generated' in st.session_state and st.session_state.mesh_generated:
        st.metric("Porosity (ε)", f"{st.session_state.porosity:.3f}")
        st.metric("Surface Area", f"{st.session_state.surface_area:.6f} m²")
        st.metric("Volume", f"{st.session_state.volume:.9f} m³")
        st.metric("A_surf/V", f"{st.session_state.A_surf_V:.2f} m⁻¹")
        
        st.markdown("---")
        st.subheader("Export")
        
        # Export buttons
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("Export STL", use_container_width=True):
                try:
                    from stl import mesh
                    
                    # Create STL mesh
                    num_faces = len(faces)
                    stl_data = np.zeros(num_faces, dtype=mesh.Mesh.dtype)
                    
                    for i, face in enumerate(faces):
                        # Get triangle vertices
                        v0 = vertices[face[0]]
                        v1 = vertices[face[1]]
                        v2 = vertices[face[2]]
                        
                        # Compute normal
                        normal = np.cross(v1 - v0, v2 - v0)
                        norm = np.linalg.norm(normal)
                        if norm > 0:
                            normal = normal / norm
                        
                        # Store in STL format
                        stl_data['vectors'][i][0] = v0
                        stl_data['vectors'][i][1] = v1
                        stl_data['vectors'][i][2] = v2
                        stl_data['normals'][i] = normal
                    
                    stl_mesh = mesh.Mesh(stl_data)
                    
                    # Save to bytes
                    stl_bytes = stl_mesh.data.tobytes()
                    
                    st.download_button(
                        label="Download STL",
                        data=stl_bytes,
                        file_name=f"tpms_{tpms_type.value}_{variant.value}_t{t:.2f}.stl",
                        mime="application/octet-stream"
                    )
                except ImportError:
                    st.error("numpy-stl not installed. Install with: pip install numpy-stl")
                except Exception as e:
                    st.error(f"STL export failed: {e}")
        
        with col_export2:
            if st.button("Export OBJ", use_container_width=True):
                try:
                    # Create OBJ file content
                    obj_lines = []
                    obj_lines.append("# TPMS Lattice OBJ File")
                    obj_lines.append(f"# Type: {tpms_type.value}, Variant: {variant.value}, t: {t:.2f}")
                    obj_lines.append(f"# Porosity: {st.session_state.porosity:.3f}")
                    
                    # Vertices
                    for v in vertices:
                        obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
                    
                    # Faces (OBJ uses 1-indexed)
                    for face in faces:
                        obj_lines.append(f"f {face[0]+1} {face[1]+1} {face[2]+1}")
                    
                    obj_content = "\n".join(obj_lines)
                    
                    st.download_button(
                        label="Download OBJ",
                        data=obj_content,
                        file_name=f"tpms_{tpms_type.value}_{variant.value}_t{t:.2f}.obj",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"OBJ export failed: {e}")
        
    else:
        st.info("Generate a mesh to see properties and export options")

# Reset generate flag
if 'generate' in st.session_state:
    st.session_state.generate = False

