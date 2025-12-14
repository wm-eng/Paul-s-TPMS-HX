"""
TPMS Heat Exchanger Optimizer GUI
Frontend for visualizing TPMS structures, 2D models, and optimization results.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import os
import sys
from typing import Optional, Dict, Any
import threading

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
        ttk.Button(button_frame, text="Export Results", command=self._export_results).pack(pady=3, fill=tk.X)
        
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
        
        # Action Buttons - Make them more prominent and visible at top
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
        ttk.Button(button_frame, text="Export Results", command=self._export_results).pack(pady=3, fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(button_frame, textvariable=self.status_var, foreground="blue")
        status_label.pack(pady=5)
        
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
        
        # Tab 1: 2D Model Visualization
        self.tab_2d = ttk.Frame(notebook)
        notebook.add(self.tab_2d, text="2D Model")
        self._create_2d_plot(self.tab_2d)
        
        # Tab 2: Temperature Profiles
        self.tab_temp = ttk.Frame(notebook)
        notebook.add(self.tab_temp, text="Temperature Profiles")
        self._create_temp_plot(self.tab_temp)
        
        # Tab 3: Design Variable (d field)
        self.tab_d = ttk.Frame(notebook)
        notebook.add(self.tab_d, text="Design Variable (d)")
        self._create_d_plot(self.tab_d)
        
        # Tab 4: Results Summary
        self.tab_results = ttk.Frame(notebook)
        notebook.add(self.tab_results, text="Results")
        self._create_results_display(self.tab_results)
        
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
            if not os.path.exists(rve_path):
                # Try relative to project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                rve_path = os.path.join(project_root, rve_path)
            
            # Get metal selection
            metal_name = self.metal_var.get() if hasattr(self, 'metal_var') else None
            
            # Get cell size from config if available
            cell_size = getattr(self.config, 'rve_cell_size', None) if self.config else None
            
            self.rve_db = RVEDatabase(rve_path, cell_size=cell_size, metal_name=metal_name)
            status_msg = f"RVE database loaded: {os.path.basename(rve_path)}"
            if metal_name:
                status_msg += f" | Metal: {metal_name}"
            self.status_var.set(status_msg)
            
            # Update metal info display
            if hasattr(self, 'metal_var'):
                self._update_metal_info()
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
        self.rve_path_var.set(default_path)
        self._load_rve_db()
    
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
    
    def _update_2d_plot(self):
        """Update 2D model visualization."""
        self.ax_2d.clear()
        
        if self.config and self.config.geometry.use_2d and self.model:
            # Plot 2D flow paths
            if hasattr(self.model, 'hot_path') and self.model.hot_path is not None:
                path = self.model.hot_path.path_coords
                if path is not None and len(path) > 0:
                    self.ax_2d.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Hot Path', marker='o', markersize=3)
            
            if hasattr(self.model, 'cold_path') and self.model.cold_path is not None:
                path = self.model.cold_path.path_coords
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
    
    def _update_results_display(self):
        """Update results summary display."""
        self.results_text.delete(1.0, tk.END)
        
        if self.current_result:
            result = self.current_result
            text = f"""RESULTS SUMMARY
{'='*60}

Heat Transfer:
  Q = {result.Q/1e6:.3f} MW

Pressure Drops:
  ΔP_hot = {result.delta_P_hot/1e3:.2f} kPa
  ΔP_cold = {result.delta_P_cold/1e3:.2f} kPa

Temperatures:
  T_hot_in = {result.T_hot[0]:.2f} K
  T_hot_out = {result.T_hot[-1]:.2f} K
  T_cold_in = {result.T_cold[-1]:.2f} K
  T_cold_out = {result.T_cold[0]:.2f} K

Pressures:
  P_hot_in = {result.P_hot[0]/1e5:.2f} bar
  P_hot_out = {result.P_hot[-1]/1e5:.2f} bar
  P_cold_in = {result.P_cold[-1]/1e5:.2f} bar
  P_cold_out = {result.P_cold[0]/1e5:.2f} bar

"""
            
            if self.opt_result:
                text += f"""Optimization:
  Iterations: {len(self.opt_result.d_fields)}
  Final Q: {self.opt_result.Q_values[-1]/1e6:.3f} MW
  Improvement: {((self.opt_result.Q_values[-1] - self.opt_result.Q_values[0])/self.opt_result.Q_values[0]*100):.1f}%
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
            filetypes=[("CSV files", "*.csv"), ("VTK files", "*.vtk"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                from .export_geometry import export_field_csv, export_vtk
                
                if filename.endswith('.csv'):
                    export_field_csv(self.current_result, self.d_field, self.config, filename=filename)
                elif filename.endswith('.vtk'):
                    export_vtk(self.current_result, self.d_field, self.config, filename=filename)
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = TPMSOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

