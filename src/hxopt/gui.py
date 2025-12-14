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
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel: Controls
        self._create_control_panel(main_frame)
        
        # Right panel: Visualizations
        self._create_visualization_panel(main_frame)
        
    def _create_control_panel(self, parent):
        """Create control panel with inputs."""
        control_frame = ttk.LabelFrame(parent, text="Configuration", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # TPMS Structure Selection
        ttk.Label(control_frame, text="TPMS Structure:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.tpms_var = tk.StringVar(value=self.current_tpms)
        tpms_combo = ttk.Combobox(control_frame, textvariable=self.tpms_var, 
                                   values=self.tpms_types, state="readonly", width=20)
        tpms_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        tpms_combo.bind("<<ComboboxSelected>>", self._on_tpms_change)
        
        # RVE Table Selection
        ttk.Label(control_frame, text="RVE Table:").grid(row=1, column=0, sticky=tk.W, pady=5)
        rve_frame = ttk.Frame(control_frame)
        rve_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        self.rve_path_var = tk.StringVar(value="data/rve_tables/primitive_default.csv")
        ttk.Entry(rve_frame, textvariable=self.rve_path_var, width=25).pack(side=tk.LEFT)
        ttk.Button(rve_frame, text="Browse", command=self._browse_rve).pack(side=tk.LEFT, padx=5)
        
        # Geometry Parameters
        geom_frame = ttk.LabelFrame(control_frame, text="Geometry", padding="5")
        geom_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
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
        flow_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
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
        fluid_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
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
        
        # Real-fluid properties
        self.use_real_props_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fluid_frame, text="Use Real-Fluid Properties (REFPROP/COOLProp)", 
                       variable=self.use_real_props_var).grid(row=len(fluid_params), column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Optimization Parameters
        opt_frame = ttk.LabelFrame(control_frame, text="Optimization", padding="5")
        opt_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
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
        
        # Action Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Load Config", command=self._load_config).pack(pady=5, fill=tk.X)
        ttk.Button(button_frame, text="Solve", command=self._solve).pack(pady=5, fill=tk.X)
        ttk.Button(button_frame, text="Optimize", command=self._optimize).pack(pady=5, fill=tk.X)
        ttk.Button(button_frame, text="Export Results", command=self._export_results).pack(pady=5, fill=tk.X)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, foreground="blue").grid(row=7, column=0, columnspan=2, pady=5)
        
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
            
            fluid = FluidConfig(
                T_hot_in=300.0,
                T_cold_in=20.0,
                P_hot_in=2e5,
                P_cold_in=1e5,
                m_dot_hot=0.01,
                m_dot_cold=0.05,
            )
            
            optimization = OptimizationConfig()
            
            self.config = Config(
                geometry=geometry,
                fluid=fluid,
                optimization=optimization,
                rve_table_path="data/rve_tables/primitive_default.csv",
            )
            
            self._load_rve_db()
            self.status_var.set("Default config loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default config: {e}")
    
    def _load_rve_db(self):
        """Load RVE database."""
        try:
            rve_path = self.rve_path_var.get()
            if not os.path.exists(rve_path):
                rve_path = os.path.join(os.path.dirname(__file__), "..", "..", rve_path)
            
            self.rve_db = RVEDatabase(rve_path)
            self.status_var.set(f"RVE database loaded: {os.path.basename(rve_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load RVE database: {e}")
    
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
        fluid = FluidConfig(
            T_hot_in=self.fluid_vars["T_hot_in"].get(),
            T_cold_in=self.fluid_vars["T_cold_in"].get(),
            P_hot_in=self.fluid_vars["P_hot_in"].get(),
            P_cold_in=self.fluid_vars["P_cold_in"].get(),
            m_dot_hot=self.fluid_vars["m_dot_hot"].get(),
            m_dot_cold=self.fluid_vars["m_dot_cold"].get(),
            use_real_properties=self.use_real_props_var.get(),
        )
        
        # Optimization
        optimization = OptimizationConfig(
            max_iter=self.opt_vars["max_iter"].get(),
            d_min=self.opt_vars["d_min"].get(),
            d_max=self.opt_vars["d_max"].get(),
            d_init=self.opt_vars["d_init"].get(),
            delta_P_max_hot=self.opt_vars["delta_P_max_hot"].get(),
            delta_P_max_cold=self.opt_vars["delta_P_max_cold"].get(),
        )
        
        # Config
        config = Config(
            geometry=geometry,
            fluid=fluid,
            optimization=optimization,
            rve_table_path=self.rve_path_var.get(),
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
            
            if not self.rve_db:
                self._load_rve_db()
            
            model = MacroModel(config, self.rve_db)
            d_field = np.full(config.geometry.n_segments, config.optimization.d_init)
            result = model.solve(d_field)
            
            self.current_result = result
            self.d_field = d_field
            self.model = model
            
            self._update_visualizations()
            self.status_var.set("Solve complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Solve failed: {e}")
            self.status_var.set("Solve failed")
            import traceback
            traceback.print_exc()
    
    def _optimize(self):
        """Run optimization."""
        def optimize_thread():
            try:
                self.status_var.set("Optimizing...")
                self.root.update()
                
                config = self._get_config_from_ui()
                self.config = config
                
                if not self.rve_db:
                    self._load_rve_db()
                
                opt_result = optimize(config, self.rve_db, log_file=None)
                
                self.opt_result = opt_result
                if opt_result.d_fields:
                    self.d_field = opt_result.d_fields[-1]
                    model = MacroModel(config, self.rve_db)
                    self.current_result = model.solve(self.d_field)
                    self.model = model
                
                self.root.after(0, self._update_visualizations)
                self.root.after(0, lambda: self.status_var.set("Optimization complete"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Optimization failed: {e}"))
                self.root.after(0, lambda: self.status_var.set("Optimization failed"))
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

