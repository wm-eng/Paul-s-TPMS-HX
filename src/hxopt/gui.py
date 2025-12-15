"""
TPMS Heat Exchanger Optimizer GUI
Frontend for visualizing TPMS structures, 2D models, and optimization results.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
import sys
from typing import Optional, Dict, Any
import threading
import gc

# Import hxopt modules
from .config import Config, GeometryConfig, FluidConfig, OptimizationConfig
from .rve_db import RVEDatabase
from .macro_model import MacroModel, MacroModelResult
from .optimize_mma import optimize
from .flow_paths import FlowPathType
from .metal_properties import MetalProperties


class TPMSOptimizerGUI:
    """Main GUI application for TPMS heat exchanger optimizer."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TPMS Heat Exchanger Optimizer")
        self.root.geometry("1400x900")
        
        # State
        self.config: Optional[Config] = None
        self.rve_db: Optional[RVEDatabase] = None
        self.model: Optional[MacroModel] = None
        self.current_result: Optional[MacroModelResult] = None
        self.opt_result = None
        self.d_field: Optional[np.ndarray] = None
        
        # Store canvas references for cleanup
        self.canvases = []
        
        # TPMS structure options
        self.tpms_types = ["Primitive", "Gyroid", "Diamond", "IWP", "Neovius"]
        self.current_tpms = "Primitive"
        
        # Flow path types
        self.flow_path_types = {
            "Straight": FlowPathType.STRAIGHT,
            "U-Shaped": FlowPathType.U_SHAPED,
            "L-Shaped": FlowPathType.L_SHAPED,
        }
        
        self._create_widgets()
        self._load_default_config()
        
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main container with scrollable left panel
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel: Controls (with scrollbar)
        self._create_control_panel(main_frame)
        
        # Right panel: Visualizations
        self._create_visualization_panel(main_frame)
        
    def _create_control_panel(self, parent):
        """Create control panel with inputs."""
        # Create a frame with scrollbar for the control panel
        control_container = ttk.Frame(parent)
        control_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        control_container.columnconfigure(0, weight=1)
        control_container.rowconfigure(0, weight=1)
        
        # Create canvas for scrolling
        canvas = tk.Canvas(control_container, width=350)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        control_frame = ttk.LabelFrame(scrollable_frame, text="Configuration", padding="10")
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        # Action Buttons - Make them FIRST and most prominent
        button_frame = ttk.LabelFrame(control_frame, text="Actions", padding="10")
        button_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Primary START button - most prominent
        start_button = tk.Button(button_frame, text="▶ START", 
                                command=self._solve, 
                                bg="#4CAF50", fg="white",
                                font=("Arial", 14, "bold"),
                                relief=tk.RAISED, bd=3,
                                cursor="hand2",
                                padx=20, pady=15)
        start_button.pack(pady=10, fill=tk.X)
        
        # Secondary action buttons
        self.solve_button = ttk.Button(button_frame, text="Run Simulation", 
                                       command=self._solve, width=25)
        self.solve_button.pack(pady=5, fill=tk.X)
        
        self.optimize_button = ttk.Button(button_frame, text="⚡ Optimize", 
                                         command=self._optimize, width=25)
        self.optimize_button.pack(pady=5, fill=tk.X)
        
        # Secondary buttons
        ttk.Separator(button_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        ttk.Button(button_frame, text="Load Config", command=self._load_config).pack(pady=3, fill=tk.X)
        
        # Export buttons
        export_frame = ttk.LabelFrame(button_frame, text="Export", padding="5")
        export_frame.pack(pady=3, fill=tk.X)
        ttk.Button(export_frame, text="Export Results (CSV/VTK)", command=self._export_results).pack(pady=2, fill=tk.X)
        ttk.Button(export_frame, text="Export STL", command=self._export_stl).pack(pady=2, fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(button_frame, textvariable=self.status_var, foreground="blue")
        status_label.pack(pady=5)
        
        # TPMS Structure Selection
        ttk.Label(control_frame, text="TPMS Structure:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.tpms_var = tk.StringVar(value=self.current_tpms)
        tpms_combo = ttk.Combobox(control_frame, textvariable=self.tpms_var, 
                                   values=self.tpms_types, state="readonly", width=20)
        tpms_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        tpms_combo.bind("<<ComboboxSelected>>", self._on_tpms_change)
        
        # RVE Table Selection
        ttk.Label(control_frame, text="RVE Table:").grid(row=2, column=0, sticky=tk.W, pady=5)
        rve_frame = ttk.Frame(control_frame)
        rve_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        self.rve_path_var = tk.StringVar(value="data/rve_tables/primitive_default.csv")
        ttk.Entry(rve_frame, textvariable=self.rve_path_var, width=25).pack(side=tk.LEFT)
        ttk.Button(rve_frame, text="Browse", command=self._browse_rve).pack(side=tk.LEFT, padx=5)
        
        # Geometry Parameters
        geom_frame = ttk.LabelFrame(control_frame, text="Geometry", padding="5")
        geom_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.geom_vars = {}
        geom_params = [
            ("Length (m):", "length", 0.5),
            ("Width (m):", "width", 0.1),
            ("Height (m):", "height", 0.1),
            ("Segments:", "n_segments", 50),
        ]
        
        for i, (label, key, default) in enumerate(geom_params):
            ttk.Label(geom_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=default) if key != "n_segments" else tk.IntVar(value=int(default))
            self.geom_vars[key] = var
            ttk.Entry(geom_frame, textvariable=var, width=15).grid(row=i, column=1, sticky=tk.W, pady=2)
        
        # Flow Path Configuration
        flow_frame = ttk.LabelFrame(control_frame, text="Flow Path (2D)", padding="5")
        flow_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.use_2d_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(flow_frame, text="Enable 2D Flow Path", 
                       variable=self.use_2d_var, command=self._toggle_2d).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Label(flow_frame, text="Hot Path:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.hot_path_var = tk.StringVar(value="Straight")
        hot_path_combo = ttk.Combobox(flow_frame, textvariable=self.hot_path_var, 
                    values=list(self.flow_path_types.keys()), state="readonly", width=15)
        hot_path_combo.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(flow_frame, text="Cold Path:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.cold_path_var = tk.StringVar(value="Straight")
        cold_path_combo = ttk.Combobox(flow_frame, textvariable=self.cold_path_var, 
                    values=list(self.flow_path_types.keys()), state="readonly", width=15)
        cold_path_combo.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Fluid Properties
        fluid_frame = ttk.LabelFrame(control_frame, text="Fluid Properties", padding="5")
        fluid_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.fluid_vars = {}
        fluid_params = [
            ("T_hot_in (K):", "T_hot_in", 300.0),
            ("T_cold_in (K):", "T_cold_in", 20.0),
            ("P_hot_in (Pa):", "P_hot_in", 2e5),
            ("P_cold_in (Pa):", "P_cold_in", 1e5),
            ("m_dot_hot (kg/s):", "m_dot_hot", 0.01),
            ("m_dot_cold (kg/s):", "m_dot_cold", 0.05),
        ]
        
        for i, (label, key, default) in enumerate(fluid_params):
            ttk.Label(fluid_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=default)
            self.fluid_vars[key] = var
            ttk.Entry(fluid_frame, textvariable=var, width=15).grid(row=i, column=1, sticky=tk.W, pady=2)
        
        # Real-fluid properties checkbox
        self.use_real_props_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fluid_frame, text="Use Real-Fluid Properties (REFPROP/COOLProp)", 
                       variable=self.use_real_props_var,
                       command=self._toggle_real_properties).grid(row=len(fluid_params), column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Constant properties (shown when real properties disabled)
        const_props_frame = ttk.LabelFrame(fluid_frame, text="Constant Properties (if not using real properties)", padding="3")
        const_props_frame.grid(row=len(fluid_params)+1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.const_prop_vars = {}
        const_props = [
            ("ρ_hot (kg/m³):", "rho_hot", 0.18),
            ("μ_hot (Pa·s):", "mu_hot", 2.0e-5),
            ("cp_hot (J/(kg·K)):", "cp_hot", 5200.0),
            ("k_hot (W/(m·K)):", "k_hot", 0.15),
            ("ρ_cold (kg/m³):", "rho_cold", 71.0),
            ("μ_cold (Pa·s):", "mu_cold", 9.0e-6),
            ("cp_cold (J/(kg·K)):", "cp_cold", 9600.0),
            ("k_cold (W/(m·K)):", "k_cold", 0.1),
        ]
        
        for i, (label, key, default) in enumerate(const_props):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(const_props_frame, text=label).grid(row=row, column=col, sticky=tk.W, pady=1, padx=2)
            var = tk.DoubleVar(value=default)
            self.const_prop_vars[key] = var
            ttk.Entry(const_props_frame, textvariable=var, width=12).grid(row=row, column=col+1, sticky=tk.W, pady=1, padx=2)
        
        # Initially hide constant properties (since real properties is default)
        const_props_frame.grid_remove()
        
        # Metal Properties Selection
        metal_frame = ttk.LabelFrame(control_frame, text="Metal Properties", padding="5")
        metal_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(metal_frame, text="Solid Metal:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.metal_var = tk.StringVar(value="Aluminum (6061)")
        metal_combo = ttk.Combobox(
            metal_frame, 
            textvariable=self.metal_var,
            values=MetalProperties.list_metals(),
            state="readonly",
            width=25
        )
        metal_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        metal_combo.bind("<<ComboboxSelected>>", self._on_metal_change)
        
        # Metal info display
        self.metal_info_var = tk.StringVar(value="k = 167 W/(m·K) at 300K")
        ttk.Label(metal_frame, textvariable=self.metal_info_var, 
                 font=("Arial", 9), foreground="gray").grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Optimization Parameters
        opt_frame = ttk.LabelFrame(control_frame, text="Optimization", padding="5")
        opt_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.opt_vars = {}
        opt_params = [
            ("Max Iterations:", "max_iter", 20),
            ("d_min:", "d_min", 0.1),
            ("d_max:", "d_max", 0.9),
            ("d_init:", "d_init", 0.5),
            ("ΔP_max_hot (Pa):", "delta_P_max_hot", 10e3),
            ("ΔP_max_cold (Pa):", "delta_P_max_cold", 5e3),
        ]
        
        for i, (label, key, default) in enumerate(opt_params):
            ttk.Label(opt_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=default) if key != "max_iter" else tk.IntVar(value=int(default))
            self.opt_vars[key] = var
            ttk.Entry(opt_frame, textvariable=var, width=15).grid(row=i, column=1, sticky=tk.W, pady=2)
        
    def _create_visualization_panel(self, parent):
        """Create visualization panel with plots."""
        viz_frame = ttk.Frame(parent)
        viz_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.rowconfigure(1, weight=1)
        
        # Notebook for tabs
        notebook = ttk.Notebook(viz_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: TPMS Structure Visualization
        self.tab_tpms = ttk.Frame(notebook)
        notebook.add(self.tab_tpms, text="TPMS Structure")
        self._create_tpms_visualization(self.tab_tpms)
        
        # Tab 2: 2D Model Visualization
        self.tab_2d = ttk.Frame(notebook)
        notebook.add(self.tab_2d, text="2D Model")
        self._create_2d_plot(self.tab_2d)
        
        # Tab 3: Temperature Profiles
        self.tab_temp = ttk.Frame(notebook)
        notebook.add(self.tab_temp, text="Temperature Profiles")
        self._create_temp_plot(self.tab_temp)
        
        # Tab 4: Design Variable (d field)
        self.tab_d = ttk.Frame(notebook)
        notebook.add(self.tab_d, text="Design Variable (d)")
        self._create_d_plot(self.tab_d)
        
        # Tab 5: Results Summary
        self.tab_results = ttk.Frame(notebook)
        notebook.add(self.tab_results, text="Results")
        self._create_results_display(self.tab_results)
        
        # Tab 6: Fluid Property Charts
        self.tab_properties = ttk.Frame(notebook)
        notebook.add(self.tab_properties, text="Fluid Properties")
        self._create_property_charts(self.tab_properties)
        
    def _create_tpms_visualization(self, parent):
        """Create TPMS structure visualization."""
        fig = Figure(figsize=(8, 8), dpi=100)
        self.fig_tpms = fig
        self.ax_tpms = fig.add_subplot(111, projection='3d')
        self.ax_tpms.set_title("TPMS Structure Visualization")
        
        canvas = FigureCanvasTkAgg(fig, parent)
        self.canvases.append(canvas)  # Store for cleanup
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
        # Initial visualization
        self._update_tpms_visualization()
    
    def _update_tpms_visualization(self):
        """Update TPMS structure visualization based on current selection."""
        # Clear all artists to free memory
        self.ax_tpms.clear()
        # Force garbage collection of cleared artists
        gc.collect()
        
        tpms_type = self.current_tpms if hasattr(self, 'current_tpms') else "Primitive"
        
        if tpms_type == "Gyroid":
            self._plot_gyroid()
        elif tpms_type == "Primitive":
            self._plot_primitive()
        elif tpms_type == "Diamond":
            self._plot_diamond()
        elif tpms_type == "IWP":
            self._plot_iwp()
        elif tpms_type == "Neovius":
            self._plot_neovius()
        else:
            self._plot_primitive()  # Default
        
        self.ax_tpms.set_xlabel("X")
        self.ax_tpms.set_ylabel("Y")
        self.ax_tpms.set_zlabel("Z")
        self.ax_tpms.set_title(f"{tpms_type} TPMS Structure")
        self.fig_tpms.tight_layout()
        self.fig_tpms.canvas.draw()
    
    def _plot_gyroid(self):
        """Plot Gyroid TPMS structure.
        
        The Gyroid is a triply periodic minimal surface defined by:
        cos(x)*sin(y) + cos(y)*sin(z) + cos(z)*sin(x) = 0
        """
        # Reduced grid size to prevent memory issues (was 50, now 25)
        n = 25
        x = np.linspace(0, 2*np.pi, n)
        y = np.linspace(0, 2*np.pi, n)
        z = np.linspace(0, 2*np.pi, n)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Gyroid level set function
        F = np.cos(X) * np.sin(Y) + np.cos(Y) * np.sin(Z) + np.cos(Z) * np.sin(X)
        
        # Clean up intermediate arrays immediately
        del x, y, z
        
        # Extract isosurface using contour plots at different z-slices
        # This gives a better 3D representation
        threshold = 0.0
        n_slices = 8
        
        for i, z_val in enumerate(np.linspace(0, 2*np.pi, n_slices)):
            z_idx = int(z_val / (2*np.pi) * (n - 1))
            if z_idx < n:
                # Get 2D slice
                X_2d = X[:, :, z_idx]
                Y_2d = Y[:, :, z_idx]
                F_2d = F[:, :, z_idx]
                
                # Plot contour lines at this z-level
                # Use different colors for positive and negative regions
                try:
                    contours = self.ax_tpms.contour(X_2d, Y_2d, F_2d, 
                                                   levels=[threshold], 
                                                   colors='cyan', 
                                                   alpha=0.4 + 0.3 * (i % 2),
                                                   linewidths=1.5,
                                                   zdir='z', 
                                                   offset=z_val)
                except:
                    # Fallback if contour fails
                    pass
        
        # Add a more detailed surface representation using parametric form
        # Gyroid parametric representation (simplified for visualization)
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(0, 2*np.pi, 40)
        U, V = np.meshgrid(u, v)
        
        # Create multiple surface patches to show the periodic structure
        # This is an approximation that captures the Gyroid's characteristic shape
        for offset in [0, np.pi/2, np.pi, 3*np.pi/2]:
            # Rotated and translated patches
            X_patch = U + offset
            Y_patch = V + offset
            Z_patch = np.sin(U + offset) * np.cos(V) + np.cos(U) * np.sin(V + offset)
            
            # Only plot if within bounds
            mask = (X_patch >= 0) & (X_patch <= 2*np.pi) & \
                   (Y_patch >= 0) & (Y_patch <= 2*np.pi) & \
                   (Z_patch >= 0) & (Z_patch <= 2*np.pi)
            
            if np.any(mask):
                self.ax_tpms.plot_surface(X_patch, Y_patch, Z_patch, 
                                         alpha=0.3, 
                                         color='cyan', 
                                         edgecolor='blue', 
                                         linewidth=0.2,
                                         antialiased=True)
        
        # Add wireframe for structure clarity
        # Sample points on the isosurface
        u_wire = np.linspace(0, 2*np.pi, 20)
        v_wire = np.linspace(0, 2*np.pi, 20)
        U_wire, V_wire = np.meshgrid(u_wire, v_wire)
        
        # Wireframe representation
        X_wire = U_wire
        Y_wire = V_wire
        Z_wire = np.sin(U_wire) * np.cos(V_wire) + np.cos(U_wire) * np.sin(V_wire)
        
        self.ax_tpms.plot_wireframe(X_wire, Y_wire, Z_wire, 
                                   alpha=0.5, 
                                   color='blue', 
                                   linewidth=0.5)
        
        self.ax_tpms.set_xlim(0, 2*np.pi)
        self.ax_tpms.set_ylim(0, 2*np.pi)
        self.ax_tpms.set_zlim(0, 2*np.pi)
        
        # Set equal aspect ratio for better visualization
        self.ax_tpms.set_box_aspect([1, 1, 1])
        
        # Clean up large arrays
        del X, Y, Z, F, X_2d, Y_2d, F_2d, U, V, X_patch, Y_patch, Z_patch, U_wire, V_wire, X_wire, Y_wire, Z_wire
        gc.collect()
    
    def _plot_primitive(self):
        """Plot Primitive TPMS structure."""
        # Primitive: cos(x) + cos(y) + cos(z) = 0
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, 2*np.pi, 30)
        U, V = np.meshgrid(u, v)
        
        # Simplified representation
        X = U
        Y = V
        Z = np.cos(U) + np.cos(V)
        
        self.ax_tpms.plot_surface(X, Y, Z, alpha=0.7, color='orange', 
                                  edgecolor='red', linewidth=0.1)
        self.ax_tpms.set_xlim(0, 2*np.pi)
        self.ax_tpms.set_ylim(0, 2*np.pi)
        self.ax_tpms.set_zlim(-2, 2)
    
    def _plot_diamond(self):
        """Plot Diamond TPMS structure."""
        # Diamond: sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + 
        #          cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z) = 0
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, 2*np.pi, 30)
        U, V = np.meshgrid(u, v)
        
        X = U
        Y = V
        Z = np.sin(U) * np.sin(V) + np.cos(U) * np.cos(V)
        
        self.ax_tpms.plot_surface(X, Y, Z, alpha=0.7, color='green', 
                                  edgecolor='darkgreen', linewidth=0.1)
        self.ax_tpms.set_xlim(0, 2*np.pi)
        self.ax_tpms.set_ylim(0, 2*np.pi)
        self.ax_tpms.set_zlim(-2, 2)
    
    def _plot_iwp(self):
        """Plot IWP (I-graph and Wrapped Package) TPMS structure."""
        # IWP: 2*(cos(x)*cos(y) + cos(y)*cos(z) + cos(z)*cos(x)) - 
        #      (cos(2*x) + cos(2*y) + cos(2*z)) = 0
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, 2*np.pi, 30)
        U, V = np.meshgrid(u, v)
        
        X = U
        Y = V
        Z = 2 * (np.cos(U) * np.cos(V)) - (np.cos(2*U) + np.cos(2*V))
        
        self.ax_tpms.plot_surface(X, Y, Z, alpha=0.7, color='purple', 
                                  edgecolor='darkviolet', linewidth=0.1)
        self.ax_tpms.set_xlim(0, 2*np.pi)
        self.ax_tpms.set_ylim(0, 2*np.pi)
        self.ax_tpms.set_zlim(-4, 4)
    
    def _plot_neovius(self):
        """Plot Neovius TPMS structure."""
        # Neovius: 3*(cos(x) + cos(y) + cos(z)) + 4*cos(x)*cos(y)*cos(z) = 0
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, 2*np.pi, 30)
        U, V = np.meshgrid(u, v)
        
        X = U
        Y = V
        Z = 3 * (np.cos(U) + np.cos(V)) + 4 * np.cos(U) * np.cos(V)
        
        self.ax_tpms.plot_surface(X, Y, Z, alpha=0.7, color='red', 
                                  edgecolor='darkred', linewidth=0.1)
        self.ax_tpms.set_xlim(0, 2*np.pi)
        self.ax_tpms.set_ylim(0, 2*np.pi)
        self.ax_tpms.set_zlim(-8, 8)
    
    def _create_2d_plot(self, parent):
        """Create 2D model visualization."""
        fig = Figure(figsize=(8, 6), dpi=100)
        self.fig_2d = fig
        self.ax_2d = fig.add_subplot(111)
        self.ax_2d.set_title("2D Flow Path Model")
        self.ax_2d.set_xlabel("Length (m)")
        self.ax_2d.set_ylabel("Width (m)")
        self.ax_2d.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, parent)
        self.canvases.append(canvas)  # Store for cleanup
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
    def _create_temp_plot(self, parent):
        """Create temperature profile plot."""
        fig = Figure(figsize=(8, 6), dpi=100)
        self.fig_temp = fig
        self.ax_temp = fig.add_subplot(111)
        self.ax_temp.set_title("Temperature Profiles")
        self.ax_temp.set_xlabel("Position (m)")
        self.ax_temp.set_ylabel("Temperature (K)")
        self.ax_temp.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, parent)
        self.canvases.append(canvas)  # Store for cleanup
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
    def _create_d_plot(self, parent):
        """Create design variable plot."""
        fig = Figure(figsize=(8, 6), dpi=100)
        self.fig_d = fig
        self.ax_d = fig.add_subplot(111)
        self.ax_d.set_title("Design Variable d(x)")
        self.ax_d.set_xlabel("Position (m)")
        self.ax_d.set_ylabel("d")
        self.ax_d.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, parent)
        self.canvases.append(canvas)  # Store for cleanup
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
    def _create_results_display(self, parent):
        """Create results summary display."""
        results_frame = ttk.Frame(parent, padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=20, width=60)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def _load_default_config(self):
        """Load default configuration."""
        try:
            geometry = GeometryConfig(
                length=0.5,
                width=0.1,
                height=0.1,
                n_segments=50,
            )
            
            # Default: use real-fluid properties (REFPROP/COOLProp)
            # But provide constant properties as fallback
            fluid = FluidConfig(
                T_hot_in=300.0,
                T_cold_in=20.0,
                P_hot_in=2e5,
                P_cold_in=1e5,
                m_dot_hot=0.01,
                m_dot_cold=0.05,
                use_real_properties=True,  # Default to real properties
                # Provide constant properties as fallback (for helium/hydrogen at cryogenic temps)
                rho_hot=0.18,  # kg/m³, helium at ~40K
                mu_hot=2.0e-5,  # Pa·s
                cp_hot=5200.0,  # J/(kg·K)
                k_hot=0.15,  # W/(m·K)
                rho_cold=71.0,  # kg/m³, liquid hydrogen at ~20K
                mu_cold=9.0e-6,  # Pa·s
                cp_cold=9600.0,  # J/(kg·K)
                k_cold=0.1,  # W/(m·K)
            )
            
            optimization = OptimizationConfig()
            
            # Default metal
            metal_name = "Aluminum (6061)"
            
            self.config = Config(
                geometry=geometry,
                fluid=fluid,
                optimization=optimization,
                rve_table_path="data/rve_tables/primitive_default.csv",
                metal_name=metal_name,
            )
            
            # Update UI with default metal
            if hasattr(self, 'metal_var'):
                self.metal_var.set(metal_name)
                self._update_metal_info()
            
            self._load_rve_db()
            self.status_var.set("Default config loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default config: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_rve_db(self):
        """Load RVE database."""
        try:
            rve_path = self.rve_path_var.get()
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            full_path = os.path.join(project_root, rve_path) if not os.path.isabs(rve_path) else rve_path
            
            # Check if file exists, try fallback if not
            if not os.path.exists(full_path):
                # Try relative path first
                if not os.path.exists(rve_path):
                    # Fall back to primitive_default.csv
                    fallback_path = os.path.join(project_root, "data/rve_tables/primitive_default.csv")
                    if os.path.exists(fallback_path):
                        rve_path = "data/rve_tables/primitive_default.csv"
                        self.rve_path_var.set(rve_path)
                        full_path = fallback_path
                    else:
                        raise FileNotFoundError(f"RVE table not found: {rve_path} and fallback not available")
                else:
                    full_path = rve_path
            else:
                # Use the full path
                if not os.path.isabs(rve_path):
                    # Update to use relative path for display
                    pass
            
            # Get metal selection
            metal_name = self.metal_var.get() if hasattr(self, 'metal_var') else None
            
            # Get cell size from config if available
            cell_size = getattr(self.config, 'rve_cell_size', None) if self.config else None
            
            self.rve_db = RVEDatabase(full_path, cell_size=cell_size, metal_name=metal_name)
            status_msg = f"RVE database loaded: {os.path.basename(full_path)}"
            if metal_name:
                status_msg += f" | Metal: {metal_name}"
            self.status_var.set(status_msg)
            
            # Update metal info display
            if hasattr(self, 'metal_var'):
                self._update_metal_info()
        except FileNotFoundError as e:
            # Try to use primitive as fallback
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                fallback_path = os.path.join(project_root, "data/rve_tables/primitive_default.csv")
                if os.path.exists(fallback_path):
                    metal_name = self.metal_var.get() if hasattr(self, 'metal_var') else None
                    cell_size = getattr(self.config, 'rve_cell_size', None) if self.config else None
                    self.rve_db = RVEDatabase(fallback_path, cell_size=cell_size, metal_name=metal_name)
                    self.rve_path_var.set("data/rve_tables/primitive_default.csv")
                    self.status_var.set(f"RVE database loaded (fallback): primitive_default.csv")
                    messagebox.showwarning(
                        "RVE Table Not Found",
                        f"RVE table not found: {os.path.basename(str(e))}\n"
                        f"Using Primitive RVE properties as fallback."
                    )
                else:
                    raise
            except Exception:
                messagebox.showerror("Error", f"Failed to load RVE database: {e}\n\nNo fallback available.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load RVE database: {e}")
            # Don't raise - gracefully handle error to prevent GUI crash
    
    def _browse_rve(self):
        """Browse for RVE table file."""
        filename = filedialog.askopenfilename(
            title="Select RVE Table",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.rve_path_var.set(filename)
            self._load_rve_db()
    
    def _on_tpms_change(self, event=None):
        """Handle TPMS structure change."""
        self.current_tpms = self.tpms_var.get()
        # Update RVE path based on TPMS type
        default_path = f"data/rve_tables/{self.current_tpms.lower()}_default.csv"
        
        # Check if file exists, fall back to primitive if not
        if not os.path.exists(default_path):
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            full_path = os.path.join(project_root, default_path)
            if not os.path.exists(full_path):
                # Fall back to primitive_default.csv for missing TPMS types
                fallback_path = "data/rve_tables/primitive_default.csv"
                self.rve_path_var.set(fallback_path)
                messagebox.showinfo(
                    "RVE Table Not Found",
                    f"RVE table for {self.current_tpms} not found.\n"
                    f"Using Primitive RVE properties as fallback.\n\n"
                    f"To use {self.current_tpms}-specific properties, create:\n"
                    f"{default_path}"
                )
            else:
                self.rve_path_var.set(default_path)
        else:
            self.rve_path_var.set(default_path)
        
        self._load_rve_db()
        # Update TPMS visualization
        if hasattr(self, 'ax_tpms'):
            self._update_tpms_visualization()
    
    def _toggle_2d(self):
        """Toggle 2D flow path mode."""
        pass  # UI already handles this
    
    def _toggle_real_properties(self):
        """Toggle visibility of constant properties based on real properties checkbox."""
        # Find the constant properties frame
        for widget in self.root.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.Frame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, ttk.LabelFrame) and "Constant Properties" in str(grandchild.cget("text")):
                            if self.use_real_props_var.get():
                                grandchild.grid_remove()
                            else:
                                grandchild.grid()
                            return
    
    def _on_metal_change(self, event=None):
        """Handle metal selection change."""
        self._update_metal_info()
        # Reload RVE database with new metal
        if hasattr(self, 'rve_db'):
            self._load_rve_db()
    
    def _update_metal_info(self):
        """Update metal info display."""
        try:
            metal_name = self.metal_var.get()
            metal = MetalProperties(metal_name)
            k_300 = metal.thermal_conductivity(300.0)
            k_77 = metal.thermal_conductivity(77.0)
            k_20 = metal.thermal_conductivity(20.0)
            info = f"k = {k_300:.1f} W/(m·K) @ 300K | {k_77:.1f} @ 77K | {k_20:.1f} @ 20K"
            self.metal_info_var.set(info)
        except Exception as e:
            self.metal_info_var.set(f"Error: {e}")
    
    def _get_config_from_ui(self) -> Config:
        """Get configuration from UI inputs."""
        # Geometry
        geometry = GeometryConfig(
            length=self.geom_vars["length"].get(),
            width=self.geom_vars["width"].get(),
            height=self.geom_vars["height"].get(),
            n_segments=self.geom_vars["n_segments"].get(),
            use_2d=self.use_2d_var.get(),
            hot_path_type=self.flow_path_types.get(self.hot_path_var.get(), FlowPathType.STRAIGHT),
            cold_path_type=self.flow_path_types.get(self.cold_path_var.get(), FlowPathType.STRAIGHT),
        )
        
        # Fluid
        use_real = self.use_real_props_var.get()
        fluid_kwargs = {
            "T_hot_in": self.fluid_vars["T_hot_in"].get(),
            "T_cold_in": self.fluid_vars["T_cold_in"].get(),
            "P_hot_in": self.fluid_vars["P_hot_in"].get(),
            "P_cold_in": self.fluid_vars["P_cold_in"].get(),
            "m_dot_hot": self.fluid_vars["m_dot_hot"].get(),
            "m_dot_cold": self.fluid_vars["m_dot_cold"].get(),
            "use_real_properties": use_real,
        }
        
        # Add constant properties if not using real properties
        if not use_real:
            fluid_kwargs.update({
                "rho_hot": self.const_prop_vars["rho_hot"].get(),
                "mu_hot": self.const_prop_vars["mu_hot"].get(),
                "cp_hot": self.const_prop_vars["cp_hot"].get(),
                "k_hot": self.const_prop_vars["k_hot"].get(),
                "rho_cold": self.const_prop_vars["rho_cold"].get(),
                "mu_cold": self.const_prop_vars["mu_cold"].get(),
                "cp_cold": self.const_prop_vars["cp_cold"].get(),
                "k_cold": self.const_prop_vars["k_cold"].get(),
            })
        
        fluid = FluidConfig(**fluid_kwargs)
        
        # Optimization
        optimization = OptimizationConfig(
            max_iter=self.opt_vars["max_iter"].get(),
            d_min=self.opt_vars["d_min"].get(),
            d_max=self.opt_vars["d_max"].get(),
            d_init=self.opt_vars["d_init"].get(),
            delta_P_max_hot=self.opt_vars["delta_P_max_hot"].get(),
            delta_P_max_cold=self.opt_vars["delta_P_max_cold"].get(),
        )
        
        # Metal selection
        metal_name = self.metal_var.get() if hasattr(self, 'metal_var') else None
        
        # Config
        config = Config(
            geometry=geometry,
            fluid=fluid,
            optimization=optimization,
            rve_table_path=self.rve_path_var.get(),
            metal_name=metal_name,
        )
        
        return config
    
    def _load_config(self):
        """Load configuration from file."""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            # TODO: Implement config loading from JSON
            messagebox.showinfo("Info", "Config loading from JSON not yet implemented")
    
    
    def _solve(self):
        """Solve the model with current configuration."""
        try:
            self.status_var.set("Solving...")
            self.root.update()
            
            config = self._get_config_from_ui()
            self.config = config
            
            # Load RVE database
            rve_path = config.rve_table_path
            if not os.path.exists(rve_path):
                # Try relative to project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                rve_path = os.path.join(project_root, rve_path)
            # Get metal and cell size
            metal_name = getattr(self.config, 'metal_name', None) if self.config else None
            if not metal_name and hasattr(self, 'metal_var'):
                metal_name = self.metal_var.get()
            cell_size = getattr(self.config, 'rve_cell_size', None) if self.config else None
            self.rve_db = RVEDatabase(rve_path, cell_size=cell_size, metal_name=metal_name)
            
            model = MacroModel(config, self.rve_db)
            d_field = np.full(config.geometry.n_segments, config.optimization.d_init)
            result = model.solve(d_field)
            
            self.current_result = result
            self.d_field = d_field
            self.model = model
            
            self._update_visualizations()
            self.status_var.set("Solve complete ✓")
            
        except Exception as e:
            messagebox.showerror("Error", f"Solve failed: {e}")
            self.status_var.set("Solve failed ✗")
            import traceback
            traceback.print_exc()
    
    def _optimize(self):
        """Run optimization."""
        def optimize_thread():
            try:
                self.root.after(0, lambda: self.status_var.set("Optimizing..."))
                self.root.update()
                
                config = self._get_config_from_ui()
                self.config = config
                
                # Load RVE database
                rve_path = config.rve_table_path
                if not os.path.exists(rve_path):
                    # Try relative to project root
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    rve_path = os.path.join(project_root, rve_path)
                self.rve_db = RVEDatabase(rve_path)
                
                opt_result = optimize(config, self.rve_db, log_file=None)
                
                self.opt_result = opt_result
                if opt_result.d_fields:
                    self.d_field = opt_result.d_fields[-1]
                    model = MacroModel(config, self.rve_db)
                    self.current_result = model.solve(self.d_field)
                    self.model = model
                
                self.root.after(0, self._update_visualizations)
                self.root.after(0, lambda: self.status_var.set("Optimization complete ✓"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Optimization failed: {e}"))
                self.root.after(0, lambda: self.status_var.set("Optimization failed ✗"))
                import traceback
                traceback.print_exc()
        
        thread = threading.Thread(target=optimize_thread)
        thread.daemon = True
        thread.start()
    
    def _update_visualizations(self):
        """Update all visualizations."""
        if not self.current_result or self.d_field is None:
            return
        
        # 2D Model
        self._update_2d_plot()
        
        # Temperature profiles
        self._update_temp_plot()
        
        # Design variable
        self._update_d_plot()
        
        # Results
        self._update_results_display()
        
        # Force garbage collection after updates
        gc.collect()
    
    def _update_2d_plot(self):
        """Update 2D model visualization."""
        self.ax_2d.clear()
        gc.collect()  # Clean up cleared artists
        
        if self.config and self.config.geometry.use_2d and self.model:
            # Plot 2D flow paths
            if hasattr(self.model, 'hot_path') and self.model.hot_path is not None:
                path = self.model.hot_path.coordinates  # Use 'coordinates' property, not 'path_coords'
                if path is not None and len(path) > 0:
                    self.ax_2d.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Hot Path', marker='o', markersize=3)
            
            if hasattr(self.model, 'cold_path') and self.model.cold_path is not None:
                path = self.model.cold_path.coordinates  # Use 'coordinates' property, not 'path_coords'
                if path is not None and len(path) > 0:
                    self.ax_2d.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Cold Path', marker='s', markersize=3)
            
            self.ax_2d.set_xlim(0, self.config.geometry.length)
            self.ax_2d.set_ylim(0, self.config.geometry.width)
        else:
            # 1D model - show as line
            if self.config:
                self.ax_2d.plot([0, self.config.geometry.length], [0, 0], 'k-', linewidth=3, label='Flow Direction')
                self.ax_2d.set_xlim(0, self.config.geometry.length)
                self.ax_2d.set_ylim(-0.01, 0.01)
        
        self.ax_2d.set_title("2D Flow Path Model")
        self.ax_2d.set_xlabel("Length (m)")
        self.ax_2d.set_ylabel("Width (m)")
        self.ax_2d.legend()
        self.ax_2d.grid(True, alpha=0.3)
        self.fig_2d.tight_layout()
        self.fig_2d.canvas.draw()
    
    def _update_temp_plot(self):
        """Update temperature profile plot."""
        self.ax_temp.clear()
        gc.collect()  # Clean up cleared artists
        
        if self.current_result:
            x = self.current_result.x
            self.ax_temp.plot(x, self.current_result.T_hot, 'r-', linewidth=2, label='Hot Fluid')
            self.ax_temp.plot(x, self.current_result.T_cold, 'b-', linewidth=2, label='Cold Fluid')
            if hasattr(self.current_result, 'T_solid') and self.current_result.T_solid is not None:
                self.ax_temp.plot(x, self.current_result.T_solid, 'g--', linewidth=1, label='Solid')
        
        self.ax_temp.set_title("Temperature Profiles")
        self.ax_temp.set_xlabel("Position (m)")
        self.ax_temp.set_ylabel("Temperature (K)")
        self.ax_temp.legend()
        self.ax_temp.grid(True, alpha=0.3)
        self.fig_temp.tight_layout()
        self.fig_temp.canvas.draw()
    
    def _update_d_plot(self):
        """Update design variable plot."""
        self.ax_d.clear()
        gc.collect()  # Clean up cleared artists
        
        if self.d_field is not None and self.config:
            x = np.linspace(0, self.config.geometry.length, len(self.d_field))
            self.ax_d.plot(x, self.d_field, 'k-', linewidth=2, marker='o', markersize=3)
            self.ax_d.axhline(y=self.config.optimization.d_max, color='r', linestyle='--', label='d_max')
            self.ax_d.axhline(y=self.config.optimization.d_min, color='r', linestyle='--', label='d_min')
        
        self.ax_d.set_title("Design Variable d(x)")
        self.ax_d.set_xlabel("Position (m)")
        self.ax_d.set_ylabel("d")
        self.ax_d.legend()
        self.ax_d.grid(True, alpha=0.3)
        self.ax_d.set_ylim(0, 1)
        self.fig_d.tight_layout()
        self.fig_d.canvas.draw()
    
    def _format_number(self, value, unit="", use_scientific=True, max_decimals=3):
        """
        Format number with appropriate precision and scientific notation.
        
        Parameters:
        -----------
        value : float
            Number to format
        unit : str
            Unit string to append
        use_scientific : bool
            Use scientific notation for very large/small numbers
        max_decimals : int
            Maximum decimal places
        """
        import numpy as np
        
        # Check for unphysical values (NaN, Inf, or extremely large)
        if not np.isfinite(value) or abs(value) > 1e50:
            return f"ERROR (unphysical value){' ' + unit if unit else ''}"
        
        # Use scientific notation for very large or very small numbers
        if use_scientific and (abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0)):
            return f"{value:.{max_decimals}e} {unit}".strip()
        
        # Format with appropriate decimal places
        if abs(value) >= 1000:
            return f"{value:.{max_decimals-1}f} {unit}".strip()
        elif abs(value) >= 1:
            return f"{value:.{max_decimals}f} {unit}".strip()
        else:
            return f"{value:.{max_decimals+1}f} {unit}".strip()
    
    def _update_results_display(self):
        """Update results summary display."""
        self.results_text.delete(1.0, tk.END)
        
        if self.current_result:
            result = self.current_result
            
            # Format heat transfer
            Q_mw = result.Q / 1e6
            Q_str = self._format_number(Q_mw, "MW", use_scientific=abs(Q_mw) > 1e6 or abs(Q_mw) < 1e-3)
            
            # Format pressure drops
            dP_hot_kpa = result.delta_P_hot / 1e3
            dP_cold_kpa = result.delta_P_cold / 1e3
            dP_hot_str = self._format_number(dP_hot_kpa, "kPa", use_scientific=abs(dP_hot_kpa) > 1e6)
            dP_cold_str = self._format_number(dP_cold_kpa, "kPa", use_scientific=abs(dP_cold_kpa) > 1e6)
            
            # Format temperatures
            T_hot_in = result.T_hot[0]
            T_hot_out = result.T_hot[-1]
            T_cold_in = result.T_cold[-1]
            T_cold_out = result.T_cold[0]
            
            T_hot_in_str = self._format_number(T_hot_in, "K", use_scientific=False, max_decimals=1)
            T_hot_out_str = self._format_number(T_hot_out, "K", use_scientific=abs(T_hot_out) > 1e6 or abs(T_hot_out) < 1e-3)
            T_cold_in_str = self._format_number(T_cold_in, "K", use_scientific=abs(T_cold_in) > 1e6 or abs(T_cold_in) < 1e-3)
            T_cold_out_str = self._format_number(T_cold_out, "K", use_scientific=False, max_decimals=1)
            
            # Format pressures
            P_hot_in_bar = result.P_hot[0] / 1e5
            P_hot_out_bar = result.P_hot[-1] / 1e5
            P_cold_in_bar = result.P_cold[-1] / 1e5
            P_cold_out_bar = result.P_cold[0] / 1e5
            
            P_hot_in_str = self._format_number(P_hot_in_bar, "bar", use_scientific=False, max_decimals=2)
            P_hot_out_str = self._format_number(P_hot_out_bar, "bar", use_scientific=abs(P_hot_out_bar) > 1e6)
            P_cold_in_str = self._format_number(P_cold_in_bar, "bar", use_scientific=abs(P_cold_in_bar) > 1e6)
            P_cold_out_str = self._format_number(P_cold_out_bar, "bar", use_scientific=False, max_decimals=2)
            
            text = f"""RESULTS SUMMARY
{'='*60}

Heat Transfer:
  Q = {Q_str}

Pressure Drops:
  ΔP_hot = {dP_hot_str}
  ΔP_cold = {dP_cold_str}

Temperatures:
  T_hot_in = {T_hot_in_str}
  T_hot_out = {T_hot_out_str}
  T_cold_in = {T_cold_in_str}
  T_cold_out = {T_cold_out_str}

Pressures:
  P_hot_in = {P_hot_in_str}
  P_hot_out = {P_hot_out_str}
  P_cold_in = {P_cold_in_str}
  P_cold_out = {P_cold_out_str}

"""
            
            if self.opt_result:
                final_Q = self.opt_result.Q_values[-1] / 1e6
                initial_Q = self.opt_result.Q_values[0] / 1e6
                
                # Calculate improvement safely
                if abs(initial_Q) > 1e-9:
                    improvement = ((final_Q - initial_Q) / abs(initial_Q)) * 100
                    improvement_str = self._format_number(improvement, "%", use_scientific=False, max_decimals=1)
                else:
                    improvement_str = "N/A"
                
                final_Q_str = self._format_number(final_Q, "MW", use_scientific=abs(final_Q) > 1e6 or abs(final_Q) < 1e-3)
                
                text += f"""Optimization:
  Iterations: {len(self.opt_result.d_fields)}
  Final Q: {final_Q_str}
  Improvement: {improvement_str}
"""
            
            self.results_text.insert(1.0, text)
    
    def _export_results(self):
        """Export results to file."""
        if not self.current_result:
            messagebox.showwarning("Warning", "No results to export. Run solve or optimize first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("VTK files", "*.vtk"),
                ("STL files", "*.stl"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                from .export_geometry import export_field_csv, export_vtk
                
                if filename.endswith('.csv'):
                    export_field_csv(self.current_result, self.d_field, self.config, filename=filename)
                elif filename.endswith('.vtk'):
                    export_vtk(self.current_result, self.d_field, self.config, filename=filename)
                elif filename.endswith('.stl'):
                    # Export TPMS geometry as STL using memory-safe method
                    # Call _export_stl with the filename already chosen
                    self._export_stl_with_filename(filename)
                    return
                else:
                    messagebox.showwarning("Warning", "Unsupported file format")
                
                if not filename.endswith('.stl'):
                    messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
                import traceback
                traceback.print_exc()
    
    def _export_stl(self, filename=None):
        """Export TPMS geometry as STL file.
        
        Parameters:
        -----------
        filename : str, optional
            Output filename. If None, shows file dialog.
        """
        if not self.current_result or self.d_field is None:
            messagebox.showwarning(
                "Warning",
                "No results to export. Run solve or optimize first."
            )
            return
        
        if filename is None:
            filename = filedialog.asksaveasfilename(
                title="Export STL",
                defaultextension=".stl",
                filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
            )
        
        if filename:
            # Run STL export in background thread to prevent UI blocking and memory issues
            def export_stl_thread():
                try:
                    from .export_geometry import export_tpms_stl_from_optimization
                    from .tpms_library import TPMSType, VariantMode
                    
                    # Get TPMS type from current selection
                    tpms_type_map = {
                        "Primitive": TPMSType.PRIMITIVE,
                        "Gyroid": TPMSType.GYROID,
                        "Diamond": TPMSType.DIAMOND,
                        "IWP": TPMSType.IWP,
                        "Neovius": TPMSType.NEOVIUS,
                    }
                    tpms_type = tpms_type_map.get(self.current_tpms, TPMSType.PRIMITIVE)
                    
                    # Update status
                    self.root.after(0, lambda: self.status_var.set("Generating STL (this may take a while)..."))
                    
                    # Calculate safe resolution based on geometry to prevent memory issues
                    # Limit total grid points to ~50 million to prevent memory exhaustion
                    L = self.config.geometry.length
                    W = self.config.geometry.width
                    H = self.config.geometry.height
                    cell_size = 0.001
                    
                    # Calculate max safe resolution
                    max_total_points = 50_000_000  # 50M points max
                    base_resolution = 30  # Reduced from 50
                    
                    # Calculate grid dimensions
                    n_x = int(L / cell_size * base_resolution)
                    n_y = int(W / cell_size * base_resolution)
                    n_z = int(H / cell_size * base_resolution)
                    total_points = n_x * n_y * n_z
                    
                    # Reduce resolution if too large
                    if total_points > max_total_points:
                        scale_factor = (max_total_points / total_points) ** (1/3)
                        base_resolution = int(base_resolution * scale_factor)
                        self.root.after(0, lambda: self.status_var.set(
                            f"Generating STL (reduced resolution: {base_resolution} to prevent memory issues)..."
                        ))
                    
                    # Force garbage collection before starting
                    gc.collect()
                    
                    export_tpms_stl_from_optimization(
                        result=self.current_result,
                        d_field=self.d_field,
                        config=self.config,
                        filename=os.path.basename(filename),
                        tpms_type=tpms_type,
                        variant=VariantMode.SINGLE,
                        cell_size=cell_size,
                        resolution=base_resolution  # Use calculated safe resolution
                    )
                    
                    # Move file to user's chosen location
                    if os.path.exists(os.path.join(self.config.output_dir, os.path.basename(filename))):
                        import shutil
                        shutil.move(
                            os.path.join(self.config.output_dir, os.path.basename(filename)),
                            filename
                        )
                    
                    # Force cleanup after export
                    gc.collect()
                    
                    self.root.after(0, lambda: self.status_var.set("STL export complete ✓"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"STL exported to {filename}"))
                    
                except ImportError as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error",
                        f"STL export requires additional packages.\n"
                        f"Install with: pip install numpy-stl scikit-image\n\n"
                        f"Error: {e}"
                    ))
                    self.root.after(0, lambda: self.status_var.set("STL export failed ✗"))
                except MemoryError as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Memory Error",
                        f"STL generation requires too much memory.\n"
                        f"Try reducing geometry size or resolution.\n\n"
                        f"Error: {e}"
                    ))
                    self.root.after(0, lambda: self.status_var.set("STL export failed (memory) ✗"))
                    gc.collect()
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"STL export failed: {e}"))
                    self.root.after(0, lambda: self.status_var.set("STL export failed ✗"))
                    import traceback
                    traceback.print_exc()
                    gc.collect()
            
            # Start export in background thread
            thread = threading.Thread(target=export_stl_thread)
            thread.daemon = True
            thread.start()
    
    def _export_stl_with_filename(self, filename):
        """Helper to call _export_stl with a pre-selected filename."""
        self._export_stl(filename=filename)
    
    def _create_property_charts(self, parent):
        """Create fluid property charts tab."""
        # Create figure with subplots
        fig = Figure(figsize=(12, 10), dpi=100)
        self.fig_props = fig
        
        # Create 2x2 subplot layout
        # Top row: Constant pressure charts (T-h diagrams)
        # Bottom row: Constant enthalpy charts (T-P diagrams)
        self.ax_he_p = fig.add_subplot(2, 2, 1)  # Helium constant pressure
        self.ax_h2_p = fig.add_subplot(2, 2, 2)  # Hydrogen constant pressure
        self.ax_he_h = fig.add_subplot(2, 2, 3)  # Helium constant enthalpy
        self.ax_h2_h = fig.add_subplot(2, 2, 4)  # Hydrogen constant enthalpy
        
        canvas = FigureCanvasTkAgg(fig, parent)
        self.canvases.append(canvas)  # Store for cleanup
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        
        # Initial plot
        self._update_property_charts()
    
    def _update_property_charts(self):
        """Update fluid property charts."""
        try:
            from .materials import RealFluidProperties, HAS_REFPROP, HAS_COOLPROP
            
            # Check if we can use real properties
            if not (HAS_REFPROP or HAS_COOLPROP):
                for ax in [self.ax_he_p, self.ax_h2_p, self.ax_he_h, self.ax_h2_h]:
                    ax.clear()
                    ax.text(0.5, 0.5, 'REFPROP/COOLProp not available', 
                           ha='center', va='center', transform=ax.transAxes)
                self.fig_props.canvas.draw()
                return
            
            # Initialize fluid property objects
            try:
                props_he = RealFluidProperties('helium', backend='auto')
                props_h2 = RealFluidProperties('hydrogen', backend='auto')
            except Exception as e:
                for ax in [self.ax_he_p, self.ax_h2_p, self.ax_he_h, self.ax_h2_h]:
                    ax.clear()
                    ax.text(0.5, 0.5, f'Error initializing properties: {e}', 
                           ha='center', va='center', transform=ax.transAxes)
                self.fig_props.canvas.draw()
                return
            
            # Plot constant pressure charts (T-h diagrams)
            self._plot_constant_pressure_chart(props_he, self.ax_he_p, 'Helium')
            self._plot_constant_pressure_chart(props_h2, self.ax_h2_p, 'Hydrogen')
            
            # Plot constant enthalpy charts (T-P diagrams)
            self._plot_constant_enthalpy_chart(props_he, self.ax_he_h, 'Helium')
            self._plot_constant_enthalpy_chart(props_h2, self.ax_h2_h, 'Hydrogen')
            
            self.fig_props.tight_layout()
            self.fig_props.canvas.draw()
            
        except Exception as e:
            import traceback
            error_msg = f"Error updating property charts: {e}\n{traceback.format_exc()}"
            for ax in [self.ax_he_p, self.ax_h2_p, self.ax_he_h, self.ax_h2_h]:
                ax.clear()
                ax.text(0.5, 0.5, error_msg, ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8, wrap=True)
            self.fig_props.canvas.draw()
    
    def _plot_constant_pressure_chart(self, props, ax, fluid_name):
        """Plot constant pressure T-h diagram."""
        ax.clear()
        gc.collect()  # Clean up cleared artists
        
        # Define pressure levels (Pa)
        if 'helium' in fluid_name.lower():
            pressures = [1e5, 2e5, 5e5, 10e5]  # 1, 2, 5, 10 bar
            T_range = np.linspace(2.2, 300, 100)  # Helium: 2.2K to 300K
        else:  # Hydrogen
            pressures = [1e5, 2e5, 5e5, 10e5]  # 1, 2, 5, 10 bar
            T_range = np.linspace(14, 100, 100)  # Hydrogen: 14K to 100K
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(pressures)))
        
        for P, color in zip(pressures, colors):
            h_values = []
            T_valid = []
            
            for T in T_range:
                try:
                    h = props.enthalpy(T, P)
                    if np.isfinite(h) and h > -1e10:  # Valid value
                        h_values.append(h / 1e3)  # Convert to kJ/kg
                        T_valid.append(T)
                except Exception:
                    continue
            
            if len(h_values) > 0:
                ax.plot(h_values, T_valid, '-', color=color, linewidth=2, 
                       label=f'P = {P/1e5:.1f} bar')
            # Clean up intermediate arrays
            del h_values, T_valid
        
        # Clean up loop variables
        del T_range, pressures, colors
        gc.collect()
        
        ax.set_xlabel('Enthalpy (kJ/kg)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title(f'{fluid_name} - Constant Pressure (T-h)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_constant_enthalpy_chart(self, props, ax, fluid_name):
        """Plot constant enthalpy T-P diagram."""
        ax.clear()
        gc.collect()  # Clean up cleared artists
        
        # Define enthalpy levels (J/kg)
        # First, sample actual enthalpy values at reference conditions to get realistic ranges
        if 'helium' in fluid_name.lower():
            # Helium: sample at a reference point to get realistic enthalpy range
            T_ref = 50.0  # K
            P_ref = 2e5  # 2 bar
            try:
                h_ref = props.enthalpy(T_ref, P_ref)
                # Use enthalpy levels around the reference, spanning typical cryogenic range
                h_levels = [h_ref - 20e3, h_ref - 10e3, h_ref, h_ref + 10e3, h_ref + 20e3]
            except:
                # Fallback: typical helium enthalpies at cryogenic temps (0-10 kJ/kg)
                h_levels = [0, 2e3, 5e3, 8e3, 10e3]  # 0, 2, 5, 8, 10 kJ/kg
            T_range = np.linspace(2.2, 300, 100)  # Reduced resolution to save memory
            P_range = np.linspace(1e5, 10e5, 30)  # Reduced from 50 to 30
        else:  # Hydrogen
            # Hydrogen: sample at liquid hydrogen conditions
            T_ref = 20.0  # K (liquid hydrogen)
            P_ref = 2e5  # 2 bar
            try:
                h_ref = props.enthalpy(T_ref, P_ref)
                # Use enthalpy levels around the reference (typically negative for liquid)
                h_levels = [h_ref - 50e3, h_ref - 25e3, h_ref, h_ref + 25e3, h_ref + 50e3]
            except:
                # Fallback: typical liquid hydrogen enthalpies
                h_levels = [-200e3, -100e3, 0, 100e3, 200e3]  # Negative for liquid
            T_range = np.linspace(14, 100, 100)  # Reduced resolution to save memory
            P_range = np.linspace(1e5, 10e5, 30)  # Reduced from 50 to 30
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(h_levels)))
        
        for h_target, color in zip(h_levels, colors):
            T_values = []
            P_values = []
            
            # For each pressure, find temperature that gives target enthalpy
            for P in P_range:
                # Binary search for temperature
                T_low = T_range[0]
                T_high = T_range[-1]
                tolerance = 0.5  # K (relaxed for better convergence)
                max_iter = 100  # Increased iterations
                
                # Check if target enthalpy is achievable at this pressure
                # by checking bounds
                try:
                    h_low = props.enthalpy(T_low, P)
                    h_high = props.enthalpy(T_high, P)
                    
                    # If target is outside range, skip this pressure
                    if not (min(h_low, h_high) <= h_target <= max(h_low, h_high)):
                        continue
                except:
                    continue
                
                for _ in range(max_iter):
                    T_mid = (T_low + T_high) / 2
                    try:
                        h_mid = props.enthalpy(T_mid, P)
                        if not np.isfinite(h_mid):
                            break
                        
                        # Tolerance in J/kg (1000 J/kg = 1 kJ/kg)
                        if abs(h_mid - h_target) < tolerance * 1e3:
                            T_values.append(T_mid)
                            P_values.append(P / 1e5)  # Convert to bar
                            break
                        elif h_mid < h_target:
                            T_low = T_mid
                        else:
                            T_high = T_mid
                    except Exception:
                        break
                    
                    if (T_high - T_low) < tolerance:
                        # If we're close enough, use the midpoint
                        try:
                            h_final = props.enthalpy((T_low + T_high) / 2, P)
                            if np.isfinite(h_final) and abs(h_final - h_target) < 5e3:  # 5 kJ/kg tolerance
                                T_values.append((T_low + T_high) / 2)
                                P_values.append(P / 1e5)
                        except:
                            pass
                        break
            
            if len(T_values) > 0:
                ax.plot(P_values, T_values, '-', color=color, linewidth=2,
                       label=f'h = {h_target/1e3:.0f} kJ/kg')
            # Clean up intermediate arrays
            del T_values, P_values
        
        # Clean up loop variables
        del T_range, P_range, h_levels, colors
        gc.collect()
        
        ax.set_xlabel('Pressure (bar)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title(f'{fluid_name} - Constant Enthalpy (T-P)')
        # Only show legend if there are labeled lines
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)


def main():
    """Launch the GUI application."""
    try:
        root = tk.Tk()
    except Exception as e:
        print(f"Error creating Tk root window: {e}")
        print("This may be due to running in a non-GUI environment.")
        print("Try running from Terminal.app or use: pythonw -m hxopt.gui")
        raise
    
    def on_closing():
        """Handle window closing with proper cleanup."""
        # Clean up matplotlib figures
        import matplotlib.pyplot as plt
        plt.close('all')
        # Force garbage collection
        gc.collect()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = TPMSOptimizerGUI(root)
    root.mainloop()
    
    # Final cleanup
    gc.collect()


if __name__ == "__main__":
    main()

